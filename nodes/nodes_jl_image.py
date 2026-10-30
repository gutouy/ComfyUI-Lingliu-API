"""JL image node: image models the platform serves only through its JL group.

JL_IMAGE_CONTRACTS mirrors the platform's per-model image contract, so an
unsupported combination stops locally before any upload or paid submit.
Submit is a single async POST; polling never resubmits.
"""

import asyncio
import re

from ..upstream_mask import mask_upstream
from ..lingliu_config import LINGLIU_PROVIDER, get_config, mask_provider_text, normalize_provider
from ..utils import blank_image_tensor
from .lingliu_image_channels import (
    JL_MAX_REFERENCE_BYTES,
    ImageChannelError,
    ImageSubmitUncertain,
    ImageTaskFailed,
    image_tensor_size,
    jl_reference_bytes,
    jl_live,
    nearest_ratio,
    submit_once,
    wait_flat_image_task,
)
from .nodes_google import LingliuGoogleBase

NODE_ID = "lingliu_jl_image"
ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
MAX_REFERENCE_SLOTS = 3
QUERY_HINT = "原任务可能仍在生成，请用任务查询节点查询该任务 ID，不要重复提交生成"
UNCERTAIN_HINT = "提交结果不确定，请先到任务日志确认，不要重复提交"

# The platform's JL image contract: only these combinations are accepted.
JL_IMAGE_CONTRACTS = {
    "grok-image": {
        "resolutions": ("1K", "2K"),
        "ratios": ("1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"),
        "max_refs": 3,
    },
}
JL_IMAGE_MODELS = tuple(JL_IMAGE_CONTRACTS)
RATIO_CHOICES = ("auto", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9")
RESOLUTION_CHOICES = ("1K", "2K")
RESPONSE_FORMATS = ("url", "b64_json")


def resolve_controls(model, ratio, resolution, response_format):
    """Check the widgets against the model contract before anything is sent."""
    model = str(model or "").strip()
    contract = JL_IMAGE_CONTRACTS.get(model)
    if contract is None:
        raise ImageChannelError(f"不支持的 JL 图像模型：{model}")
    resolution = str(resolution or "").strip().upper()
    if resolution not in contract["resolutions"]:
        raise ImageChannelError(f"{model} 清晰度只支持 {'/'.join(contract['resolutions'])}，当前 {resolution or '空'}")
    ratio = str(ratio or "").strip() or "auto"
    if ratio != "auto" and ratio not in contract["ratios"]:
        raise ImageChannelError(f"{model} 比例只支持 auto/{'/'.join(contract['ratios'])}，当前 {ratio}")
    response_format = str(response_format or "").strip() or "url"
    if response_format not in RESPONSE_FORMATS:
        raise ImageChannelError(f"响应格式只支持 {'/'.join(RESPONSE_FORMATS)}，当前 {response_format}")
    return contract, ratio, resolution, response_format


def collect_reference_frames(model, contract, images):
    """Split connected IMAGE batches into single frames; empty slots are skipped."""
    frames = []
    for image in images:
        if image is None:
            continue
        shape = getattr(image, "shape", None)
        count = int(shape[0]) if shape is not None and len(shape) == 4 else 1
        frames.extend(image[index : index + 1] if count > 1 else image for index in range(count))
    if len(frames) > contract["max_refs"]:
        raise ImageChannelError(f"{model} 最多支持 {contract['max_refs']} 张参考图，当前 {len(frames)} 张。")
    return frames


def encode_references(frames):
    payloads = [jl_reference_bytes(frame) for frame in frames]
    for index, data in enumerate(payloads, start=1):
        if not data or len(data) > JL_MAX_REFERENCE_BYTES:
            raise ImageChannelError(f"第 {index} 张参考图为空或超过 {JL_MAX_REFERENCE_BYTES >> 20} MiB。")
    return payloads


def resolve_ratio(contract, ratio, frames):
    """The platform treats a missing ratio as 1:1 and rejects auto above 1K.

    auto therefore follows the first reference image, or 1:1 without one.
    """
    if ratio != "auto":
        return ratio
    width, height = image_tensor_size(frames[0]) if frames else (1, 1)
    return nearest_ratio(width, height, list(contract["ratios"]))


def build_fields(model, prompt, ratio, resolution, response_format):
    # No size or n: the platform derives the pixel size from resolution and
    # ratio, and grok-image accepts only one image per request.
    return {
        "model": model,
        "prompt": str(prompt or "").strip(),
        "response_format": response_format,
        "resolution": resolution,
        "aspect_ratio": ratio,
    }


def resolve_jl_config(api_key_override=""):
    config = get_config()
    provider = normalize_provider(config.get("api_provider"))
    if provider not in ALLOWED_PROVIDERS:
        raise ImageChannelError("JL 图像只支持灵流线路，请在设置里切换到灵流线路。")
    raw_key = str(api_key_override or config.get("api_key") or "").strip()
    if not raw_key:
        raise ImageChannelError("缺少 API 密钥，请在设置里填写或在节点里输入。")
    config["api_key"] = raw_key
    config["api_provider"] = provider
    return config


def _mask(text, secrets):
    text = str(text or "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return mask_upstream(text)


def _summarize_items(items):
    """Keep URLs, drop inline image bytes from the response text."""
    summary = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        entry = {key: value for key, value in item.items() if key != "b64_json"}
        if item.get("b64_json"):
            entry["b64_json"] = f"<{len(str(item['b64_json']))} chars>"
        summary.append(entry)
    return summary


def _submit_uncertain(exc):
    """A 4xx is a definite refusal; anything else may have reached the platform."""
    status_code = getattr(exc, "status_code", None)
    return not (isinstance(exc, ImageChannelError) or (status_code and 400 <= int(status_code) < 500))


class LingliuJLImage(LingliuGoogleBase):
    CATEGORY = "Lingliu/JL"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图像URL", "任务ID", "响应")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "模型": (list(JL_IMAGE_MODELS), {"default": JL_IMAGE_MODELS[0]}),
                "比例": (list(RATIO_CHOICES), {"default": "1:1", "tooltip": "auto：有参考图时按第 1 张参考图取最接近的比例，没有参考图时按 1:1。"}),
                "清晰度": (list(RESOLUTION_CHOICES), {"default": "1K"}),
                "响应格式": (list(RESPONSE_FORMATS), {"default": "url"}),
            },
            "optional": {
                "参考图1": ("IMAGE", {"tooltip": "最多 3 张参考图；接了参考图走图生图。"}),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True, "tooltip": "关闭后只提交并返回任务 ID，可用任务查询节点取图。"}),
                "最大轮询次数": ("INT", {"default": 360, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 2, "min": 1, "max": 60}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _json(self, envelope, secrets):
        return _mask(self.safe_json(envelope), secrets)

    def _status(self, unique_id, task_id, status):
        self.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status}", append=False)

    async def _ensure_live(self, config, model):
        try:
            live = await asyncio.to_thread(jl_live, config.get("base_url"), model)
        except Exception as exc:
            raise ImageChannelError(f"无法确认当前线路的 JL 渠道状态，已停止提交（未扣费）：{type(exc).__name__}") from exc
        if not live:
            raise ImageChannelError(f"该线路 JL 尚未上线（模型 {model}），已停止提交（未扣费）。")

    async def generate(self, 提示词, 模型, 比例, 清晰度, 响应格式, 参考图1=None, 参考图2=None, 参考图3=None,
                       API密钥="", 等待完成=True, 最大轮询次数=360, 轮询间隔=2, unique_id=None):
        model = str(模型 or "").strip()
        secrets = {str(API密钥 or "").strip()}
        envelope = {"node": NODE_ID, "model": model, "channel": "JL"}
        task_id = ""
        stage = "prepare"
        self.publish_task_id(unique_id, clear=True)
        try:
            contract, ratio, resolution, response_format = resolve_controls(model, 比例, 清晰度, 响应格式)
            max_poll_attempts, poll_interval = int(最大轮询次数), int(轮询间隔)
            if not 1 <= max_poll_attempts <= 720:
                raise ImageChannelError("最大轮询次数须在 1–720 之间")
            if not 1 <= poll_interval <= 60:
                raise ImageChannelError("轮询间隔须在 1–60 秒之间")
            prompt = str(提示词 or "").strip()
            if not prompt:
                raise ImageChannelError("提示词不能为空。")
            frames = collect_reference_frames(model, contract, (参考图1, 参考图2, 参考图3))
            config = resolve_jl_config(API密钥)
            secrets.add(config["api_key"])
            await self._ensure_live(config, model)
            references = await asyncio.to_thread(encode_references, frames)
            fields = build_fields(model, prompt, resolve_ratio(contract, ratio, frames), resolution, response_format)
            envelope.update({"request": fields, "reference_images": len(references)})
            stage = "submit"
            if references:
                files = [("image", (f"image_{index}.jpg", data, "image/jpeg")) for index, data in enumerate(references)]
                payload = await submit_once(
                    self, _async_url(self.endpoint(config, "images_edit_path")), config,
                    timeout=self.timeout(config), headers=self.auth_headers(config), data=fields, files=files,
                )
            else:
                payload = await submit_once(
                    self, _async_url(self.endpoint(config, "images_generate_path")), config,
                    timeout=self.timeout(config), headers=self.auth_headers(config, json_mode=True), json=fields,
                )
            envelope["submit_response"] = payload
            task_id = self.extract_task_id(payload)
            if not task_id:
                raise RuntimeError("创建响应缺少任务 ID")
            stage = "poll"
            self.publish_task_id(unique_id, task_id, append=False)
            self._status(unique_id, task_id, "submitted")
            envelope.update({"task_id": task_id, "status": "submitted"})
            if not 等待完成:
                envelope["message"] = "任务已提交，未等待完成；可用任务查询节点按任务 ID 取图。"
                return blank_image_tensor(), "", task_id, self._json(envelope, secrets)
            items, _payload = await wait_flat_image_task(
                self, config, task_id, max_poll_attempts=max_poll_attempts, poll_interval=poll_interval,
            )
            tensor, image_url = await self.decode_image_items(items, timeout=self.timeout(config))
            envelope.update({"status": "completed", "data": _summarize_items(items)})
            self._status(unique_id, task_id, "completed")
            return tensor, image_url, task_id, self._json(envelope, secrets)
        except ImageSubmitUncertain as exc:
            task_id = exc.task_id
            message = str(exc)
        except Exception as exc:
            message = str(mask_provider_text(exc)).rstrip("。")
            if stage == "submit" and _submit_uncertain(exc):
                message += f"；{UNCERTAIN_HINT}"
            elif stage == "poll" and not isinstance(exc, (ImageTaskFailed, TimeoutError)):
                message += f"；{QUERY_HINT}"
        message = _mask(message, secrets)
        if task_id:
            self.publish_task_id(unique_id, task_id, append=False)
        self.publish_live_message(unique_id, f"Task ID: {task_id}\n错误: {message}" if task_id else f"错误: {message}", append=False)
        envelope.update({"status": "error", "task_id": task_id, "error": {"stage": stage, "message": message}})
        return blank_image_tensor(), "", task_id, self._json(envelope, secrets)


def _async_url(url):
    return f"{url}{'&' if '?' in url else '?'}async=true"
