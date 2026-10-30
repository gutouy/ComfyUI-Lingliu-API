"""Image matting node (V17.7, console 1.6.15): one image in, a transparent PNG out.

The platform cuts out what the prompt names and returns an RGBA PNG.  The
result is downloaded once and kept byte for byte: IMAGE drops alpha (RGB,
original colours), MASK carries it (1 = transparent, the same convention as
ComfyUI's Load Image), and the original PNG can be written to output/ as is.
The source image goes straight to R2 when it can, otherwise inside a multipart
submit.  Nothing is ever resubmitted automatically, and there is no cancel.
"""

import asyncio
import json
import os
import re
import time
from io import BytesIO
from urllib.parse import quote

import numpy as np
import requests
import torch
from PIL import Image

from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    get_config,
    mask_provider_payload,
    mask_provider_text,
    normalize_provider,
)
from ..utils import blank_image_tensor, tensor2pil
from .lingliu_reference_upload import log, upload_references
from .nodes_depth_video import _hide_upstream_name
from .nodes_jl_video import URL_PATTERN, _origin
from .nodes_minimax_h3 import _redact, _redact_text
from .nodes_openai_compatible import LingliuBaseNode

NODE_ID = "lingliu_matting"
MODEL = "灵流-图片抠图（透明底）"
ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
JSON_PATH = "/v1/images/generations"
MULTIPART_PATH = "/v1/images/edits"
TASK_PATH = "/v1/images/tasks/{task_id}"
MAX_PROMPT_CHARS = 2000
MAX_IMAGE_BYTES = 30 * 1024 * 1024
JPEG_FALLBACK_QUALITY = 95
MAX_IDEMPOTENCY_KEY = 256
TRANSIENT_POLL_LIMIT = 12
DOWNLOAD_ATTEMPTS = 3
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OUTPUT_SUBDIR = "lingliu_matting"
KEY_PATTERN = re.compile(r"^[\x21-\x7e]+$")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
PENDING_STATUSES = {"", "queued", "in_progress", "pending", "submitted", "processing", "running", "unknown"}
FAILED_STATUSES = {"failed", "failure", "error", "cancelled", "canceled"}
SUCCESS_STATUSES = {"completed", "succeeded", "success"}
PRICE_TIP = "按次计费 0.2✦/张（再乘分组倍率），失败全额退；提交后不能取消，一般 1–2 分钟出图。"
EXPIRY_NOTE = "图片URL 24 小时内有效、需带 API 密钥访问；要长期保存请打开「保存透明PNG」（原样保存，不丢透明）。"
QUERY_HINT = "原任务可能仍在生成，请把任务 ID 填进本节点的「续查任务ID」再运行，不要重复提交"
UNCERTAIN_HINT = "提交结果不确定，请先到任务日志确认，不要重复提交"
NOT_CHARGED = "未提交、未扣费"
REFUSED = "未扣费"


class MattingError(RuntimeError):
    """uncertain means the submit may have reached the platform."""

    def __init__(self, message, *, uncertain=False, task_id=""):
        super().__init__(message)
        self.uncertain = uncertain
        self.task_id = str(task_id or "")


class _Transient(RuntimeError):
    pass


def mask_text(text, secrets=(), origin=""):
    """Hide keys (any sk-/Bearer token too) and every URL that is not on the platform's own origin."""
    text = _redact_text(str(text or ""), secrets)

    def replace(match):
        url = match.group(0)
        return url if origin and _origin(url) == origin else "<hidden_upstream_url>"

    return _hide_upstream_name(mask_provider_text(URL_PATTERN.sub(replace, text)))


def mask_json(payload, secrets=(), origin=""):
    masked = mask_provider_payload(_redact(payload, secrets))
    return mask_text(json.dumps(masked, ensure_ascii=False, indent=2), secrets, origin)


def validate_prompt(prompt):
    prompt = str(prompt or "").strip()
    if not prompt:
        raise MattingError(f"请在「抠什么」里写要抠出的主体，例如：人物、左边的杯子（{NOT_CHARGED}）")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise MattingError(f"「抠什么」最多 {MAX_PROMPT_CHARS} 字，当前 {len(prompt)} 字（{NOT_CHARGED}）")
    return prompt


def validate_idempotency_key(key):
    key = str(key or "").strip()
    if key and (len(key) > MAX_IDEMPOTENCY_KEY or not KEY_PATTERN.match(key)):
        raise MattingError(f"防重复提交键只能用英文字母、数字和符号（不含空格），最多 {MAX_IDEMPOTENCY_KEY} 个字符")
    return key


def validate_task_id(task_id):
    task_id = str(task_id or "").strip()
    if task_id and not TASK_ID_PATTERN.match(task_id):
        raise MattingError("续查任务ID 格式不正确，请粘贴本节点输出的任务 ID")
    return task_id


def encode_source(image):
    """Return (bytes, mime, note).  PNG first; JPEG 95 only when PNG is over 30 MiB."""
    if image is None:
        raise MattingError(f"请连接要抠图的图像（{NOT_CHARGED}）")
    frames = tensor2pil(image)
    if not frames:
        raise MattingError(f"输入图像为空（前面的节点可能失败了，{NOT_CHARGED}）")
    note = f"输入了 {len(frames)} 张图，只抠第 1 张。" if len(frames) > 1 else ""
    frame = frames[0]
    buffer = BytesIO()
    frame.save(buffer, format="PNG")
    data = buffer.getvalue()
    if len(data) <= MAX_IMAGE_BYTES:
        return data, "image/png", note
    buffer = BytesIO()
    frame.convert("RGB").save(buffer, format="JPEG", quality=JPEG_FALLBACK_QUALITY)
    data = buffer.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise MattingError(f"图像太大：转成 JPEG 后仍超过 30MB，请先缩小再抠图（{NOT_CHARGED}）")
    return data, "image/jpeg", (note + "PNG 超过 30MB，已改用 JPEG 95 上传。").strip()


def _error_message(response):
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:500]
    return str(body.get("message") or error or "")[:500]


def submit_matting(base_url, api_key, prompt, timeout, *, image_url="", blob=None, mime="image/png",
                   idempotency_key=""):
    """POST exactly once.  Returns (task_id, payload) or raises MattingError."""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    fields = {"model": MODEL, "prompt": prompt, "n": 1, "response_format": "url"}
    try:
        if image_url:
            response = requests.post(build_url(base_url, JSON_PATH), json={**fields, "image": image_url},
                                     headers={**headers, "Content-Type": "application/json"},
                                     timeout=timeout, allow_redirects=False)
        else:
            suffix = "png" if mime == "image/png" else "jpg"
            response = requests.post(build_url(base_url, MULTIPART_PATH),
                                     data={key: str(value) for key, value in fields.items()},
                                     files={"image": (f"input.{suffix}", blob, mime)},
                                     headers=headers, timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        raise MattingError(f"提交时连接中断（{type(exc).__name__}）", uncertain=True) from exc
    status, message = response.status_code, _error_message(response)
    if status in (200, 201, 202):
        try:
            payload = response.json()
        except ValueError as exc:
            raise MattingError(f"创建响应不是 JSON（HTTP {status}）", uncertain=True) from exc
        task_id = str((payload or {}).get("task_id") or (payload or {}).get("id") or "") if isinstance(payload, dict) else ""
        if not task_id:
            raise MattingError("创建响应缺少任务 ID", uncertain=True)
        return task_id, payload
    hints = {
        400: "参数未被接受",
        401: "API 密钥无效",
        403: "余额或令牌额度不足，或没有此模型权限",
        409: "防重复提交键已用于另一张图或另一个主体，请换一个键或留空",
        429: "当前排队已满，请稍后再试",
    }
    if status in hints:
        detail = f"：{message}" if message else ""
        raise MattingError(f"{hints[status]}（HTTP {status}{detail}，{REFUSED}）")
    if status == 503:
        try:
            body = response.json()
        except ValueError:
            body = {}
        task_id = str(body.get("task_id") or "") if isinstance(body, dict) else ""
        if task_id:
            raise MattingError("提交结果待平台核对", uncertain=True, task_id=task_id)
        raise MattingError(f"线路暂不可用：{message or 'HTTP 503'}（{REFUSED}）")
    raise MattingError(f"提交失败（HTTP {status}）{('：' + message) if message else ''}", uncertain=True)


def fetch_task(base_url, api_key, task_id, timeout):
    url = build_url(base_url, TASK_PATH.format(task_id=quote(task_id, safe="")))
    try:
        response = requests.get(url, headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
                                timeout=timeout)
    except requests.RequestException as exc:
        raise _Transient(type(exc).__name__) from exc
    if response.status_code == 404:
        raise MattingError("任务不存在，或不属于当前 API 密钥", task_id=task_id)
    if response.status_code == 429 or response.status_code >= 500:
        raise _Transient(f"HTTP {response.status_code}")
    if response.status_code >= 400:
        raise MattingError(f"查询失败（HTTP {response.status_code}）{_error_message(response)}", task_id=task_id)
    try:
        payload = response.json()
    except ValueError as exc:
        raise _Transient("查询响应不是 JSON") from exc
    return payload if isinstance(payload, dict) else {}


def task_status(payload):
    return str((payload or {}).get("status") or "").strip().lower()


def result_url(payload):
    items = (payload or {}).get("data")
    if not isinstance(items, list):
        result = (payload or {}).get("result")
        items = result.get("data") if isinstance(result, dict) else []
    for item in items or []:
        if isinstance(item, dict) and str(item.get("url") or "").startswith(("http://", "https://")):
            return str(item["url"])
    return ""


async def wait_task(base_url, api_key, task_id, timeout, *, max_polls, interval, first_now=False, notify=None):
    """Poll until completed or failed; never resubmits.  Returns the final payload."""
    transient, last_status = 0, None
    for attempt in range(max_polls):
        if attempt or not first_now:
            await asyncio.sleep(interval)
        try:
            payload = await asyncio.to_thread(fetch_task, base_url, api_key, task_id, timeout)
            transient = 0
        except _Transient:
            transient += 1
            if transient < TRANSIENT_POLL_LIMIT:
                continue
            raise
        status = task_status(payload)
        if notify and status != last_status:
            last_status = status
            notify(status or "pending")
        if status in SUCCESS_STATUSES:
            return payload
        if status in FAILED_STATUSES:
            error = payload.get("error")
            reason = error.get("message") if isinstance(error, dict) else error
            raise MattingError(f"抠图失败：{str(reason or status)[:500]}（失败会全额退回）", task_id=task_id)
    raise MattingError(f"轮询超时；{QUERY_HINT}", task_id=task_id)


def _get_bytes(url, headers, timeout):
    """One authenticated GET; requests drops Authorization on a cross-host redirect."""
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    if response.status_code == 429 or response.status_code >= 500:
        raise _Transient(f"HTTP {response.status_code}")
    if response.status_code >= 400:
        raise MattingError(f"下载结果失败（HTTP {response.status_code}）")
    if not response.content:
        raise _Transient("下载内容为空")
    return response.content


def download_result(base_url, api_key, task_id, payload, timeout):
    """Fetch the PNG through the platform content endpoint; the task's own URL is the fallback."""
    content_url = build_url(base_url, TASK_PATH.format(task_id=quote(task_id, safe="")) + "/content")
    candidates = [(content_url, {"Authorization": f"Bearer {api_key}"})]
    direct = result_url(payload)
    if direct and direct != content_url:
        same_origin = _origin(direct) == _origin(base_url)
        candidates.append((direct, {"Authorization": f"Bearer {api_key}"} if same_origin else {}))
    last = None
    for url, headers in candidates:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                return _get_bytes(url, headers, timeout)
            except (requests.RequestException, _Transient) as exc:
                last = exc
                if attempt + 1 < DOWNLOAD_ATTEMPTS:
                    time.sleep(2)
            except MattingError as exc:
                last = exc
                break
    reason = type(last).__name__ if isinstance(last, requests.RequestException) else str(last)
    raise MattingError(f"结果已生成但下载失败（{reason}）；可把任务 ID 填进「续查任务ID」重新取图，不会重复扣费",
                       task_id=task_id)


def decode_rgba(data, task_id=""):
    """Return (image [1,H,W,3], mask [1,H,W], info).  Never flattens or invents alpha."""
    try:
        picture = Image.open(BytesIO(data))
        picture.load()
        file_format = picture.format or ""
    except Exception as exc:
        raise MattingError(f"结果不是可识别的图片（{type(exc).__name__}）", task_id=task_id) from exc
    mode = picture.mode
    if mode in ("LA", "PA") or (mode in ("P", "L", "RGB") and "transparency" in picture.info):
        picture = picture.convert("RGBA")
    elif mode != "RGBA":
        raise MattingError(f"结果没有透明通道（{file_format or '未知格式'} {mode}），未做任何转换", task_id=task_id)
    array = np.asarray(picture, dtype=np.uint8)
    rgb = torch.from_numpy(array[..., :3].astype(np.float32) / 255.0).unsqueeze(0)
    alpha = array[..., 3]
    mask = torch.from_numpy(1.0 - alpha.astype(np.float32) / 255.0).unsqueeze(0)
    info = {
        "format": file_format,
        "source_mode": mode,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "alpha_min": int(alpha.min()),
        "alpha_max": int(alpha.max()),
        "transparent_ratio": round(float((alpha == 0).mean()), 4),
    }
    return rgb, mask, info


def output_directory():
    try:
        import folder_paths
    except Exception:
        return ""
    getter = getattr(folder_paths, "get_output_directory", None)
    return str(getter() or "") if callable(getter) else ""


def save_png(data, task_id):
    """Write the downloaded bytes unchanged; a non-PNG result is not saved as .png."""
    if not data.startswith(PNG_SIGNATURE):
        raise MattingError("结果不是 PNG，未保存（不做转码以免丢透明）", task_id=task_id)
    root = output_directory()
    if not root:
        raise MattingError("找不到 ComfyUI 输出目录，未保存透明PNG", task_id=task_id)
    folder = os.path.join(root, OUTPUT_SUBDIR)
    os.makedirs(folder, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", task_id)[:64] or "task"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for index in range(1000):
        name = f"matting_{stamp}_{stem}{'' if not index else f'_{index}'}.png"
        path = os.path.join(folder, name)
        try:
            with open(path, "xb") as handle:
                handle.write(data)
            return path
        except FileExistsError:
            continue
    raise MattingError("透明PNG 文件名冲突，未保存", task_id=task_id)


def blank_mask():
    return torch.zeros((1, 1024, 1024), dtype=torch.float32)


def resolve_matting_config(api_key_override=""):
    config = get_config()
    provider = normalize_provider(config.get("api_provider"))
    if provider not in ALLOWED_PROVIDERS:
        raise MattingError("图片抠图只支持灵流线路，请在设置里切换到灵流线路")
    api_key = str(api_key_override or config.get("api_key") or "").strip()
    if not api_key:
        raise MattingError("缺少 API 密钥，请在设置里填写或在节点里输入")
    return {
        "base_url": str(config.get("base_url") or "").rstrip("/"),
        "api_key": api_key,
        "timeout": int(config.get("timeout") or 300),
    }


class LingliuMatting(LingliuBaseNode):
    CATEGORY = "灵流API/工具/图像"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "遮罩", "透明PNG路径", "图片URL", "任务ID", "响应")
    FUNCTION = "cutout"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "抠什么": ("STRING", {"multiline": True, "default": "", "tooltip": "写要保留的主体，例如：人物、左边的杯子、整辆车。"}),
            },
            "optional": {
                "图像": ("IMAGE", {"tooltip": f"1 张图（多张只抠第 1 张），PNG 上传，超过 30MB 改 JPEG 95。{PRICE_TIP}"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True, "tooltip": "关闭后只提交并返回任务 ID，之后填进「续查任务ID」取图。"}),
                "最大轮询次数": ("INT", {"default": 60, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
                "防重复提交键": ("STRING", {
                    "default": "",
                    "tooltip": "留空=每次新建任务。填了以后，同一个键+同一张图+同一个主体会直接返回上次的任务，不再扣费；换图或换主体必须换键。",
                }),
                "保存透明PNG": ("BOOLEAN", {"default": True, "tooltip": "把平台返回的透明 PNG 原样存到 output/lingliu_matting/。"}),
                "续查任务ID": ("STRING", {"default": "", "tooltip": "填了就不再提交，只查询并取回这个任务的结果（不扣费）。"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _status(self, unique_id, task_id, status):
        if task_id:
            self.publish_task_id(unique_id, task_id, append=False)
        self.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status}", append=False)

    async def cutout(self, 抠什么="", 图像=None, API密钥="", 等待完成=True, 最大轮询次数=60, 轮询间隔=5,
                     防重复提交键="", 保存透明PNG=True, 续查任务ID="", unique_id=None):
        secrets = {str(API密钥 or "").strip()}
        envelope = {"node": NODE_ID, "model": MODEL}
        task_id, stage, origin = "", "prepare", ""
        self.publish_task_id(unique_id, clear=True)
        try:
            max_polls, interval = int(最大轮询次数), int(轮询间隔)
            if not 1 <= max_polls <= 720 or not 1 <= interval <= 60:
                raise MattingError("最大轮询次数需在 1–720，轮询间隔需在 1–60 秒")
            resume_id = validate_task_id(续查任务ID)
            config = resolve_matting_config(API密钥)
            base_url, api_key, timeout = config["base_url"], config["api_key"], config["timeout"]
            secrets.add(api_key)
            origin = _origin(base_url)
            if resume_id:
                task_id = resume_id
                envelope.update({"task_id": task_id, "resumed": True, "note": "续查已有任务，未重新提交、不扣费。"})
            else:
                prompt = validate_prompt(抠什么)
                key = validate_idempotency_key(防重复提交键)
                blob, mime, note = await asyncio.to_thread(encode_source, 图像)
                if note:
                    envelope["note"] = note
                envelope.update({"prompt": prompt, "input_bytes": len(blob), "input_mime": mime})
                stage = "upload"
                urls = await upload_references(base_url, api_key, [blob], mime=mime)
                envelope["upload"] = "direct" if urls else "multipart"
                stage = "submit"
                task_id, payload = await asyncio.to_thread(
                    submit_matting, base_url, api_key, prompt, (30, max(timeout, 120)),
                    image_url=(urls or [""])[0], blob=blob, mime=mime, idempotency_key=key)
                envelope.update({"task_id": task_id, "submit_response": payload})
                billing = payload.get("billing") if isinstance(payload, dict) else None
                if billing:
                    envelope["billing"] = billing
                log(f"抠图任务已提交：{task_id}（{envelope['upload']}）")
            self._status(unique_id, task_id, "resuming" if resume_id else "submitted")
            if not 等待完成:
                envelope["message"] = "任务已提交，未等待完成；之后把任务 ID 填进「续查任务ID」取图。"
                return blank_image_tensor(), blank_mask(), "", "", task_id, mask_json(envelope, secrets, origin)
            stage = "poll"
            final = await wait_task(base_url, api_key, task_id, (30, timeout), max_polls=max_polls, interval=interval,
                                    first_now=bool(resume_id),
                                    notify=lambda status: self._status(unique_id, task_id, status))
            stage = "download"
            data = await asyncio.to_thread(download_result, base_url, api_key, task_id, final, (30, timeout))
            image, mask, info = await asyncio.to_thread(decode_rgba, data, task_id)
            content_url = build_url(base_url, TASK_PATH.format(task_id=quote(task_id, safe="")) + "/content")
            path = ""
            if 保存透明PNG:
                stage = "save"
                path = await asyncio.to_thread(save_png, data, task_id)
            envelope.update({"status": "completed", "image_url": content_url, "result": info,
                             "result_bytes": len(data), "saved_png": path, "expiry": EXPIRY_NOTE,
                             "query_response": final})
            self._status(unique_id, task_id, "completed")
            return image, mask, path, content_url, task_id, mask_json(envelope, secrets, origin)
        except MattingError as exc:
            task_id = exc.task_id or task_id
            message = str(exc).rstrip("。")
            if exc.uncertain:
                message += f"；{QUERY_HINT if task_id else UNCERTAIN_HINT}"
        except _Transient as exc:
            message = f"查询连续失败（{exc}）；{QUERY_HINT}"
        except Exception as exc:
            message = str(exc).rstrip("。") or type(exc).__name__
            if stage in ("poll", "download") and task_id:
                message += f"；{QUERY_HINT}"
        message = mask_text(message, secrets, origin)
        if task_id:
            self.publish_task_id(unique_id, task_id, append=False)
        self.publish_live_message(unique_id, f"Task ID: {task_id}\n错误: {message}" if task_id else f"错误: {message}",
                                  append=False)
        envelope.update({"status": "error", "task_id": task_id, "error": {"stage": stage, "message": message}})
        return blank_image_tensor(), blank_mask(), "", "", task_id, mask_json(envelope, secrets, origin)
