"""Soul 2 image and Soul ID nodes (V17.6, console 1.6.14).

Generation uses the platform's async image task: one JSON POST to
/v1/images/generations, then /v1/images/tasks/{id}.  Soul accepts references
only as URLs, so the reference and training images go straight to R2 through
reference-presign; an upload failure stops before anything is submitted.
Training a Soul ID costs 25✦, so the trainer reuses a same-name Soul ID unless
asked to retrain.  Nothing is ever resubmitted automatically.
"""

import asyncio
import json
import math
import os
import re
import threading
import time
from io import BytesIO

import requests
from PIL import Image, ImageOps

from ..lingliu_config import LINGLIU_PROVIDER, build_url, mask_provider_text, normalize_provider
from ..utils import blank_image_tensor, tensor2pil
from .lingliu_image_channels import ImageChannelError, ImageTaskFailed, wait_flat_image_task
from .lingliu_reference_upload import ReferenceUploadError, upload_references_strict
from .nodes_google import LingliuGoogleBase

ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
MODEL = "soul-2"
GENERATE_PATH = "/v1/images/generations"
STYLES_PATH = "/v1/soul/styles"
CHARACTERS_PATH = "/v1/soul/characters"
RESOLUTIONS = ("1K", "2K")
RATIOS = ("3:4", "1:1", "2:3", "3:2", "4:3", "9:16", "16:9")
COUNTS = (1, 4)
DEFAULT_STYLE = "General"
STYLE_CACHE_SECONDS = 3600
REFERENCE_MAX_EDGE = 4096
TRAIN_MAX_EDGE = 2048
JPEG_QUALITY = 92
TRAIN_MIN_IMAGES, TRAIN_MAX_IMAGES = 1, 100
TRAIN_POLL_SECONDS = 20
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SOUL_ID_PATTERN = re.compile(r"^soul_[A-Za-z0-9]{8,64}$")
UUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
EXPIRY_NOTE = "图像URL 为临时链接，7 天后失效；需要长期保存请接「保存图像」。"
QUERY_HINT = "原任务可能仍在生成，请用任务查询节点查询该任务 ID，不要重复生成"
UNCERTAIN_HINT = "提交结果不确定，请先到任务日志确认，不要重复提交"
PRICE_TIP = "按次计费：1K×1 0.1✦、2K×1 0.2✦、1K×4 0.4✦、2K×4 0.8✦（再乘分组倍率）；失败、违规、取消全额退。"

# Offline snapshot of GET /v1/soul/styles (10-27).  The dropdown shows these
# names; the id actually sent is looked up live by name at run time.
STYLE_SNAPSHOT = {
    "General": "3db34ab5-3439-4317-9e03-08dc30852e69",
    "Y2K studio": "7ee16ff3-c0b0-4f5f-a0e1-e7fc46144230",
    "Street photography": "95151de0-e0e5-4e04-bd45-c58c8a4ac023",
    "Editorial street style": "3d5584b2-4d15-48d2-8a09-c1073259f4c6",
    "Warm ambient": "fafd3087-0d0f-4fb1-9af6-91b7d304687c",
    "Drain": "8261de31-258b-4a58-997c-b65bbd6d367f",
    "Neutral pastel film": "adbca234-bfff-4c19-835f-f8647655d295",
    "Digital camera": "74abc530-cec8-4c13-88a6-2b3f78bfd0ff",
    "Old smartphone": "a7cb654b-eba7-4e9b-af41-8eaa482bf7c9",
    "Asian nostalgia": "99716f25-c133-438a-829a-ec53258980b4",
    "Flash editorial": "5e87c5bd-86b6-491a-94ab-046891d1e35e",
    "Frutiger aero": "599d96ce-be5b-4071-9ee0-5b3494f6c757",
    "Muted cool film": "a15510bf-2439-4964-a2cc-0e35d1b5e70f",
    "Double exposure": "7a42d44e-0dab-4dd1-be49-c83fec827251",
    "Disposable camera": "7ef8747e-c925-4851-b9e9-6400ddb19df9",
    "Surreal solarization": "26bb19d1-25cc-4455-9a80-522eb136a4de",
    "Underwater": "04022c41-6164-47d9-b33f-cf241f0b5e15",
    "Siren": "cfd92ced-4801-42c4-b8ca-e096a405ff1a",
    "Candy pop": "cd55643f-812c-461d-9837-baa9a88dff2e",
    "Mystique city": "d6371f60-a70c-4a43-9393-6e3d8206590c",
    "Subtle flash": "e62e75c2-b1bb-433d-9192-f1196faed74e",
    "80s horror": "313f3e05-0d79-4d2f-a56a-d9d289321fd8",
    "New Indie": "184401c9-65d8-4d4b-b2e1-fd6a87a5b60c",
    "Extraterrestrial": "82e55816-3b41-4f01-8500-d36f235e08e6",
    "Theatrical light": "7876c2a5-aa83-4530-ad09-2b368bbb5e95",
    "Y2K street": "d7e58c26-b74e-4a72-8c78-d061d600c9a5",
    "Warm contrast film": "2c109e42-bf29-47a2-ae16-aa6cbbba8b8a",
    "Retro BW": "069f5a5f-8c4b-4591-a3dd-bde488672228",
    "Swag era": "6dffca48-f9fe-43f2-8589-da58a172fa3e",
    "Nature light": "dbe816eb-c651-4361-aee3-9386f8372121",
    "2000s band": "330789e8-cb50-410c-a31f-b1ba10188176",
    "Warm vivid film": "dbaabc25-7fee-438c-bd65-ae4d1775d4ab",
    "BW film": "3ece570b-a9a3-4b86-8c20-8eaa7c58cc4f",
}
STYLE_NAMES = tuple(STYLE_SNAPSHOT)

_UPSTREAM_NAME = re.compile(r"(?i)higgs\s*field")
_style_cache = {}
_style_lock = threading.Lock()


class SoulRequestError(RuntimeError):
    """A request failed.  uncertain means it may have reached the platform."""

    def __init__(self, message, *, uncertain=False, task_id=""):
        super().__init__(message)
        self.uncertain = uncertain
        self.task_id = str(task_id or "")


def clear_style_cache():
    with _style_lock:
        _style_cache.clear()


def mask(text, secrets=()):
    text = str(text or "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return _UPSTREAM_NAME.sub("上游", text)


def _error_message(response):
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return str(body.get("message") or error or "")


def _request(method, url, api_key, timeout, **kwargs):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    if "json" in kwargs:
        headers["Content-Type"] = "application/json"
    return requests.request(method, url, headers=headers, timeout=timeout, **kwargs)


def _json(response):
    try:
        payload = response.json()
    except ValueError as exc:
        raise SoulRequestError(f"平台返回格式不正确（HTTP {response.status_code}）") from exc
    return payload if isinstance(payload, dict) else {"data": payload}


def fetch_styles(base_url, api_key, *, now=None):
    """Return {name: id} from the live list, cached 1 hour per line; raises on failure."""
    key = str(base_url or "").rstrip("/")
    current = time.monotonic() if now is None else now
    with _style_lock:
        cached = _style_cache.get(key)
        if cached and current - cached[0] < STYLE_CACHE_SECONDS:
            return cached[1]
    response = _request("GET", build_url(key, STYLES_PATH), api_key, (10, 30))
    if response.status_code != 200:
        raise SoulRequestError(f"风格列表获取失败（HTTP {response.status_code}）")
    styles = {}
    for item in _json(response).get("data") or []:
        if isinstance(item, dict) and item.get("name") and UUID_PATTERN.match(str(item.get("id") or "")):
            styles[str(item["name"])] = str(item["id"])
    if not styles:
        raise SoulRequestError("风格列表为空")
    with _style_lock:
        _style_cache[key] = (current, styles)
    return styles


def resolve_style_id(base_url, api_key, name, override=""):
    """Return (style_id, source).  General sends no style_id."""
    override = str(override or "").strip()
    if override:
        if not UUID_PATTERN.match(override):
            raise ImageChannelError("风格ID 格式不正确，应为 UUID；不确定时请清空，改用下拉选择。")
        return override, "override"
    name = str(name or DEFAULT_STYLE).strip() or DEFAULT_STYLE
    if name == DEFAULT_STYLE:
        return "", "default"
    try:
        live = fetch_styles(base_url, api_key)
    except Exception:
        if name not in STYLE_SNAPSHOT:
            raise ImageChannelError(f"未知风格：{name}")
        return STYLE_SNAPSHOT[name], "snapshot"
    if name not in live:
        raise ImageChannelError(f"风格「{name}」已下线，请换一个风格（未提交、未扣费）。")
    return live[name], "live"


def _jpeg(pil_image, max_edge):
    """RGB JPEG with the long edge capped; EXIF is not carried over."""
    pil_image = pil_image.convert("RGB")
    if max(pil_image.size) > max_edge:
        pil_image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def _frames(image):
    if image is None:
        return []
    return tensor2pil(image)


def reference_blob(image):
    frames = _frames(image)
    if not frames:
        return None
    if len(frames) > 1:
        raise ImageChannelError(f"Soul 2 只支持 1 张参考图，当前接入 {len(frames)} 张。")
    return _jpeg(frames[0], REFERENCE_MAX_EDGE)


def _folder_files(folder):
    folder = str(folder or "").strip().strip('"')
    if not folder:
        return []
    if not os.path.isdir(folder):
        raise ImageChannelError(f"图片文件夹不存在：{folder}")
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if name.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(folder, name))]


def training_blobs(images=None, folder=""):
    """Encode every training image as a 2048px JPEG, one at a time to keep memory low."""
    files = _folder_files(folder)
    frame_count = 0 if images is None else (int(images.shape[0]) if len(images.shape) == 4 else 1)
    total = frame_count + len(files)
    if not TRAIN_MIN_IMAGES <= total <= TRAIN_MAX_IMAGES:
        raise ImageChannelError(f"训练图需要 {TRAIN_MIN_IMAGES}–{TRAIN_MAX_IMAGES} 张，当前 {total} 张。")
    blobs = [_jpeg(frame, TRAIN_MAX_EDGE) for frame in _frames(images)]
    for path in files:
        try:
            with Image.open(path) as opened:
                opened.draft("RGB", (TRAIN_MAX_EDGE, TRAIN_MAX_EDGE))
                blobs.append(_jpeg(ImageOps.exif_transpose(opened), TRAIN_MAX_EDGE))
        except (OSError, ValueError) as exc:
            raise ImageChannelError(f"无法读取训练图：{os.path.basename(path)}（{type(exc).__name__}）") from exc
    return blobs


def build_generation_body(prompt, resolution, ratio, count, enhance, style_id, seed, image_url, soul_id, strength):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "resolution": resolution.lower(),
        "aspect_ratio": ratio,
        "n": count,
        "enhance_prompt": enhance,
        "response_format": "url",
    }
    if style_id:
        body["style_id"] = style_id
    if seed:
        body["seed"] = seed
    if image_url:
        body["image"] = image_url
    if soul_id:
        body["soul_id"] = soul_id
        body["soul_strength"] = strength
    return body


def submit_generation(base_url, api_key, body, timeout):
    """POST once.  Returns the task id or raises SoulRequestError."""
    try:
        response = _request("POST", build_url(base_url, GENERATE_PATH), api_key, timeout, json=body)
    except requests.RequestException as exc:
        raise SoulRequestError(f"提交时连接中断（{type(exc).__name__}）", uncertain=True) from exc
    status, message = response.status_code, _error_message(response)
    if status in (200, 201, 202):
        payload = _json(response)
        task_id = str(payload.get("task_id") or payload.get("id") or "")
        if not task_id:
            raise SoulRequestError("创建响应缺少任务 ID", uncertain=True)
        return task_id, payload
    if status == 503:
        try:
            body_json = response.json()
        except ValueError:
            body_json = {}
        task_id = str((body_json or {}).get("task_id") or "") if isinstance(body_json, dict) else ""
        if task_id and str(body_json.get("status") or "").lower() == "unknown":
            raise SoulRequestError("提交结果待确认，请勿重复生成", uncertain=True, task_id=task_id)
        raise SoulRequestError(f"渠道暂不可用：{message or 'HTTP 503'}（未扣费）")
    if status == 400:
        raise SoulRequestError(f"参数未被接受：{message or 'HTTP 400'}（未扣费）")
    if status in (401, 403):
        raise SoulRequestError(f"API 密钥无效或没有 Soul 2 权限（HTTP {status}，未扣费）")
    if status == 429:
        raise SoulRequestError(f"{message or '排队中，请稍后再试'}（未扣费，请稍后手动重试）")
    if status == 502:
        raise SoulRequestError(f"上游暂不可用：{message or 'HTTP 502'}（未扣费）")
    raise SoulRequestError(f"提交失败（HTTP {status}）：{message}", uncertain=True)


def _created(item):
    try:
        return int(item.get("created_at") or 0)
    except (TypeError, ValueError):
        return 0


def describe_character(item):
    line = (f"{item.get('name', '')} | {item.get('id', '')} | {item.get('status', '')} | "
            f"{item.get('image_count', '?')} 张 | "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(_created(item))) if _created(item) else ''}")
    if item.get("fail_reason"):
        line += f" | 失败原因：{item['fail_reason']}"
    return line


class LingliuSoulBase(LingliuGoogleBase):
    CATEGORY = "Lingliu/Soul"

    def soul_config(self, api_key_override=""):
        config = self.load_config(api_key_override=str(api_key_override or ""))
        provider = normalize_provider(config.get("api_provider"))
        if provider not in ALLOWED_PROVIDERS:
            raise ImageChannelError("Soul 只支持灵流线路，请在 lingliu_api_set 里切换到灵流。")
        if not str(config.get("api_key") or "").strip():
            raise ImageChannelError("缺少 API 密钥，请先运行 lingliu_api_set 或在节点里填写。")
        config["api_key"] = str(config["api_key"]).strip()
        config["api_provider"] = provider
        return config

    def out(self, envelope, secrets):
        return mask(self.safe_json(envelope), secrets)

    def call(self, config, method, path, timeout=(10, 60), **kwargs):
        try:
            response = _request(method, build_url(config["base_url"], path), config["api_key"], timeout, **kwargs)
        except requests.RequestException as exc:
            raise SoulRequestError(f"连接平台失败（{type(exc).__name__}）", uncertain=True) from exc
        if response.status_code >= 400:
            message = _error_message(response) or f"HTTP {response.status_code}"
            error = SoulRequestError(f"{message}（HTTP {response.status_code}）")
            error.status_code = response.status_code
            raise error
        return _json(response)

    def list_characters(self, config):
        data = self.call(config, "GET", CHARACTERS_PATH).get("data")
        return [item for item in (data or []) if isinstance(item, dict)]

    def status(self, unique_id, label, value, state):
        self.publish_live_message(unique_id, f"{label}: {value}\n状态: {state}", append=False)


class LingliuSoul2Image(LingliuSoulBase):
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图像URL", "任务ID", "响应")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "清晰度": (list(RESOLUTIONS), {"default": "1K"}),
                "比例": (list(RATIOS), {"default": "3:4"}),
                "张数": ([str(n) for n in COUNTS], {"default": "1", "tooltip": PRICE_TIP}),
                "风格": (list(STYLE_NAMES), {"default": DEFAULT_STYLE}),
                "提示词增强": ("BOOLEAN", {"default": True, "tooltip": "选了 General 以外的风格时会自动打开，风格只有在增强开启时才生效。"}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "0 = 随机。"}),
            },
            "optional": {
                "参考图": ("IMAGE", {"tooltip": "可选 1 张，有就是图生图；会先直传到图床再提交。不带 Soul ID 时不保留人脸。"}),
                "Soul_ID": ("STRING", {"default": "", "tooltip": "可连 Soul ID 训练节点的输出；只能用自己训练好的 Soul ID。"}),
                "Soul_ID强度": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05}),
                "风格ID": ("STRING", {"default": "", "tooltip": "高级：填了就优先用它，用来选列表里还没有的新风格。"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True, "tooltip": "关闭后只提交并返回任务 ID，可用任务查询节点取图。"}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    async def generate(self, 提示词, 清晰度="1K", 比例="3:4", 张数="1", 风格=DEFAULT_STYLE, 提示词增强=True, 种子=0,
                       参考图=None, Soul_ID="", Soul_ID强度=1.0, 风格ID="", API密钥="", 等待完成=True,
                       最大轮询次数=120, 轮询间隔=5, unique_id=None):
        secrets = {str(API密钥 or "").strip()}
        envelope = {"node": "lingliu_soul2_image", "model": MODEL}
        task_id, stage = "", "prepare"
        self.publish_task_id(unique_id, clear=True)
        try:
            prompt = str(提示词 or "").strip()
            if not prompt:
                raise ImageChannelError("提示词不能为空。")
            resolution = str(清晰度 or "").strip().upper()
            if resolution not in RESOLUTIONS:
                raise ImageChannelError(f"清晰度只支持 {'/'.join(RESOLUTIONS)}，当前 {resolution or '空'}")
            ratio = str(比例 or "").strip()
            if ratio not in RATIOS:
                raise ImageChannelError(f"比例只支持 {'/'.join(RATIOS)}，当前 {ratio or '空'}")
            count = int(str(张数).strip()) if str(张数).strip().isdigit() else 0
            if count not in COUNTS:
                raise ImageChannelError("张数只能是 1 或 4。")
            seed = int(种子 or 0)
            if not 0 <= seed <= 1000000:
                raise ImageChannelError("种子须在 0–1000000 之间（0 = 随机）。")
            soul_id = str(Soul_ID or "").strip()
            if soul_id and not SOUL_ID_PATTERN.match(soul_id):
                raise ImageChannelError("Soul ID 格式不正确，应以 soul_ 开头。")
            strength = float(Soul_ID强度)
            if not 0 < strength <= 1:
                raise ImageChannelError("Soul ID 强度须大于 0、不超过 1。")
            max_polls, interval = int(最大轮询次数), int(轮询间隔)
            if not 1 <= max_polls <= 720:
                raise ImageChannelError("最大轮询次数须在 1–720 之间")
            if not 1 <= interval <= 60:
                raise ImageChannelError("轮询间隔须在 1–60 秒之间")
            config = self.soul_config(API密钥)
            secrets.add(config["api_key"])
            style_name = str(风格 or DEFAULT_STYLE).strip() or DEFAULT_STYLE
            style_id, style_source = await asyncio.to_thread(
                resolve_style_id, config["base_url"], config["api_key"], style_name, 风格ID)
            enhance = bool(提示词增强)
            if style_id and not enhance:
                enhance = True
                envelope["note"] = "选了风格，提示词增强已自动打开（风格只在增强开启时生效）。"
            image_url = ""
            blob = await asyncio.to_thread(reference_blob, 参考图)
            if blob:
                stage = "upload"
                try:
                    image_url = (await upload_references_strict(config["base_url"], config["api_key"], [blob]))[0]
                except ReferenceUploadError as exc:
                    raise ImageChannelError(f"参考图{exc}（未提交、未扣费）") from exc
            body = build_generation_body(prompt, resolution, ratio, count, enhance, style_id, seed, image_url,
                                         soul_id, strength)
            envelope.update({"request": body, "style": {"name": "风格ID" if style_source == "override" else style_name,
                                                        "source": style_source}})
            stage = "submit"
            task_id, payload = await asyncio.to_thread(
                submit_generation, config["base_url"], config["api_key"], body, (30, self.timeout(config)))
            envelope["submit_response"] = payload
            stage = "poll"
            self.publish_task_id(unique_id, task_id, append=False)
            self.status(unique_id, "Task ID", task_id, "submitted")
            envelope.update({"task_id": task_id, "status": "submitted"})
            if not 等待完成:
                envelope["message"] = "任务已提交，未等待完成；可用任务查询节点按任务 ID 取图。"
                return blank_image_tensor(), "", task_id, self.out(envelope, secrets)
            items, final = await wait_flat_image_task(
                self, config, task_id, max_poll_attempts=max_polls, poll_interval=interval)
            tensor, _first_url = await self.decode_image_items(items, timeout=self.timeout(config))
            urls = [str(item["url"]) for item in items if item.get("url")]
            envelope.update({"status": "completed", "images": int(tensor.shape[0]),
                             "expires_at": (final or {}).get("expires_at"), "expiry": EXPIRY_NOTE})
            self.status(unique_id, "Task ID", task_id, "completed")
            return tensor, "\n".join(urls), task_id, self.out(envelope, secrets)
        except SoulRequestError as exc:
            task_id = exc.task_id or task_id
            message = str(exc).rstrip("。")
            if exc.uncertain:
                message += f"；{UNCERTAIN_HINT}"
        except Exception as exc:
            message = str(mask_provider_text(exc)).rstrip("。")
            if stage == "poll" and not isinstance(exc, (ImageTaskFailed, TimeoutError)):
                message += f"；{QUERY_HINT}"
        message = mask(message, secrets)
        if task_id:
            self.publish_task_id(unique_id, task_id, append=False)
        self.publish_live_message(unique_id, f"Task ID: {task_id}\n错误: {message}" if task_id else f"错误: {message}",
                                  append=False)
        envelope.update({"status": "error", "task_id": task_id, "error": {"stage": stage, "message": message}})
        return blank_image_tensor(), "", task_id, self.out(envelope, secrets)


TRAIN_REUSE, TRAIN_RETRAIN = "复用", "重新训练"
ACTIVE_STATES = {"training", "ready", "unknown"}


class LingliuSoulIDTrain(LingliuSoulBase):
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("Soul ID", "状态", "响应")
    FUNCTION = "train"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "名称": ("STRING", {"default": "", "tooltip": "1–100 字；同名时按「已有同名时」处理。"}),
                "我已获得照片中人物的授权": ("BOOLEAN", {"default": False}),
                "已有同名时": ([TRAIN_REUSE, TRAIN_RETRAIN], {"default": TRAIN_REUSE,
                                "tooltip": "训练一次 25✦。复用：已有同名的训练中/已就绪 Soul ID 时直接用它，不再扣费。"}),
                "等待训练完成": ("BOOLEAN", {"default": True}),
                "最长等待分钟": ("INT", {"default": 30, "min": 1, "max": 60}),
            },
            "optional": {
                "训练图": ("IMAGE", {"tooltip": "1–100 张，和图片文件夹合并计算；会缩到长边 2048 并去掉 EXIF。"}),
                "图片文件夹": ("STRING", {"default": "", "tooltip": "本地文件夹，读取 jpg/png/webp（不含子文件夹）。"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def create_character(self, config, name, urls):
        try:
            response = _request("POST", build_url(config["base_url"], CHARACTERS_PATH), config["api_key"],
                                (30, self.timeout(config)), json={"name": name, "images": urls, "consent": True})
        except requests.RequestException as exc:
            raise SoulRequestError(f"提交训练时连接中断（{type(exc).__name__}）", uncertain=True) from exc
        message = _error_message(response)
        if response.status_code in (200, 201, 202):
            payload = _json(response)
            if not str(payload.get("id") or "").startswith("soul_"):
                raise SoulRequestError("训练响应缺少 Soul ID", uncertain=True)
            return payload
        if response.status_code == 400:
            raise SoulRequestError(f"参数未被接受：{message or 'HTTP 400'}（未扣费）")
        if response.status_code in (401, 403):
            raise SoulRequestError(f"API 密钥无效或没有 Soul ID 权限（HTTP {response.status_code}，未扣费）")
        if response.status_code == 429:
            raise SoulRequestError(f"{message or '训练排队中，请稍后再试'}（未扣费，请稍后手动重试）")
        raise SoulRequestError(f"提交训练失败（HTTP {response.status_code}）：{message}", uncertain=True)

    async def wait_character(self, config, soul_id, minutes):
        polls = max(1, math.ceil(minutes * 60 / TRAIN_POLL_SECONDS))
        item, transient = {}, 0
        for _ in range(polls):
            await asyncio.sleep(TRAIN_POLL_SECONDS)
            try:
                item = await asyncio.to_thread(self.call, config, "GET", f"{CHARACTERS_PATH}/{soul_id}")
                transient = 0
            except SoulRequestError as exc:
                if exc.uncertain and transient < 5:
                    transient += 1
                    continue
                raise
            if str(item.get("status") or "").lower() in ("ready", "failed"):
                return item, False
        return item or {"id": soul_id, "status": "training"}, True

    async def train(self, 名称, 我已获得照片中人物的授权=False, 已有同名时=TRAIN_REUSE, 等待训练完成=True, 最长等待分钟=30,
                    训练图=None, 图片文件夹="", API密钥="", unique_id=None):
        secrets = {str(API密钥 or "").strip()}
        envelope = {"node": "lingliu_soul_id_train", "model": "soul-id"}
        soul_id, state, stage = "", "", "prepare"
        try:
            name = str(名称 or "").strip()
            if not 1 <= len(name) <= 100:
                raise ImageChannelError("名称需要 1–100 字。")
            if not 我已获得照片中人物的授权:
                raise ImageChannelError("请先勾选「我已获得照片中人物的授权」（未上传、未扣费）。")
            if 已有同名时 not in (TRAIN_REUSE, TRAIN_RETRAIN):
                raise ImageChannelError(f"「已有同名时」只能选 {TRAIN_REUSE} 或 {TRAIN_RETRAIN}。")
            minutes = int(最长等待分钟)
            if not 1 <= minutes <= 60:
                raise ImageChannelError("最长等待分钟须在 1–60 之间")
            config = self.soul_config(API密钥)
            secrets.add(config["api_key"])
            envelope["name"] = name
            if 已有同名时 == TRAIN_REUSE:
                stage = "lookup"
                try:
                    existing = await asyncio.to_thread(self.list_characters, config)
                except SoulRequestError as exc:
                    raise ImageChannelError(f"无法确认是否已有同名 Soul ID，为避免重复扣费已停止（未扣费）：{exc}") from exc
                same = [item for item in existing if str(item.get("name") or "").strip() == name
                        and str(item.get("status") or "").lower() in ACTIVE_STATES]
                if same:
                    found = max(same, key=_created)
                    soul_id, state = str(found.get("id") or ""), str(found.get("status") or "").lower()
                    envelope.update({"reused": True, "soul_id": soul_id, "status": state,
                                     "message": "已有同名 Soul ID，直接复用，未重新训练、未扣费。"})
            if not soul_id:
                stage = "upload"
                blobs = await asyncio.to_thread(training_blobs, 训练图, 图片文件夹)
                envelope["image_count"] = len(blobs)
                try:
                    urls = await upload_references_strict(config["base_url"], config["api_key"], blobs, label="训练图")
                except ReferenceUploadError as exc:
                    raise ImageChannelError(f"训练图{exc}（未提交训练、未扣费）") from exc
                stage = "submit"
                created = await asyncio.to_thread(self.create_character, config, name, urls)
                soul_id, state = str(created["id"]), str(created.get("status") or "training").lower()
                envelope.update({"reused": False, "soul_id": soul_id, "status": state})
            self.status(unique_id, "Soul ID", soul_id, state)
            if 等待训练完成 and state not in ("ready", "failed"):
                stage = "poll"
                item, timed_out = await self.wait_character(config, soul_id, minutes)
                state = str(item.get("status") or state).lower()
                envelope["status"] = state
                if timed_out:
                    envelope["message"] = "等待超时，训练可能还在进行；请稍后用 Soul ID 管理节点查询，不要重新训练。"
                if item.get("fail_reason"):
                    envelope["fail_reason"] = item["fail_reason"]
            if state == "failed":
                envelope.setdefault("message", "训练失败（已退款）；可换一组照片后选「重新训练」。")
            self.status(unique_id, "Soul ID", soul_id, state)
            return soul_id, state, self.out(envelope, secrets)
        except SoulRequestError as exc:
            message = str(exc).rstrip("。")
            if exc.uncertain and stage == "submit":
                message += "；训练可能已提交，请先用 Soul ID 管理节点查看列表，不要重复训练"
        except Exception as exc:
            message = str(mask_provider_text(exc)).rstrip("。")
        message = mask(message, secrets)
        self.publish_live_message(unique_id, f"错误: {message}", append=False)
        envelope.update({"status": "error", "soul_id": soul_id, "error": {"stage": stage, "message": message}})
        return soul_id, "error", self.out(envelope, secrets)


MANAGE_LIST, MANAGE_QUERY, MANAGE_DELETE = "列表", "查询", "删除"


class LingliuSoulIDManage(LingliuSoulBase):
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("结果", "Soul ID", "响应")
    FUNCTION = "manage"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "操作": ([MANAGE_LIST, MANAGE_QUERY, MANAGE_DELETE], {"default": MANAGE_LIST}),
            },
            "optional": {
                "Soul_ID": ("STRING", {"default": "", "tooltip": "查询、删除时必填；请手动填写要删除的 Soul ID。"}),
                "确认删除": ("BOOLEAN", {"default": False, "tooltip": "删除后不能恢复。只有「操作=删除」且这里打开才会删除。"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    async def manage(self, 操作=MANAGE_LIST, Soul_ID="", 确认删除=False, API密钥=""):
        secrets = {str(API密钥 or "").strip()}
        envelope = {"node": "lingliu_soul_id_manage", "operation": 操作}
        soul_id = str(Soul_ID or "").strip()
        try:
            if 操作 not in (MANAGE_LIST, MANAGE_QUERY, MANAGE_DELETE):
                raise ImageChannelError(f"操作只能是 {MANAGE_LIST}/{MANAGE_QUERY}/{MANAGE_DELETE}。")
            if 操作 != MANAGE_LIST and not SOUL_ID_PATTERN.match(soul_id):
                raise ImageChannelError("请填写正确的 Soul ID（以 soul_ 开头）。")
            if 操作 == MANAGE_DELETE and not 确认删除:
                raise ImageChannelError("删除后不能恢复：请打开「确认删除」后再运行。")
            config = self.soul_config(API密钥)
            secrets.add(config["api_key"])
            if 操作 == MANAGE_LIST:
                items = sorted(await asyncio.to_thread(self.list_characters, config), key=_created, reverse=True)
                envelope["data"] = items
                ready = [item for item in items if str(item.get("status") or "").lower() == "ready"]
                result = "\n".join(describe_character(item) for item in items) or "还没有 Soul ID。"
                return result, str(ready[0]["id"]) if ready else "", self.out(envelope, secrets)
            path = f"{CHARACTERS_PATH}/{soul_id}"
            if 操作 == MANAGE_QUERY:
                item = await asyncio.to_thread(self.call, config, "GET", path)
                envelope["data"] = item
                return describe_character(item), soul_id, self.out(envelope, secrets)
            await asyncio.to_thread(self.call, config, "DELETE", path)
            envelope.update({"soul_id": soul_id, "deleted": True})
            return f"已删除 {soul_id}", "", self.out(envelope, secrets)
        except SoulRequestError as exc:
            message = ("Soul ID 不存在，或不属于当前 API 密钥。" if getattr(exc, "status_code", None) == 404
                       else str(exc))
        except Exception as exc:
            message = str(mask_provider_text(exc))
        message = mask(message, secrets)
        envelope["error"] = message
        return f"错误：{message}", soul_id, self.out(envelope, secrets)
