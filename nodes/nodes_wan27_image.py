"""Wan2.7 image node: wan2.7-image / wan2.7-image-pro on the platform's image route.

The platform answers these models synchronously and returns no task ID, so a
lost response cannot be looked up from the node: the submit is one POST and is
never retried.  Every boundary the platform or upstream would reject is checked
locally first, so a refused request is never sent or charged.
"""

import asyncio
import base64
import math
import re

from ..upstream_mask import mask_upstream
from ..lingliu_config import LINGLIU_PROVIDER, get_config, mask_provider_text, normalize_provider
from ..utils import blank_image_tensor
from .lingliu_image_channels import ImageChannelError, ImageSubmitUncertain, image_tensor_size, jl_reference_bytes, submit_once
from .nodes_google import LingliuGoogleBase

NODE_ID = "lingliu_wan27_image"
MODELS = ("wan2.7-image", "wan2.7-image-pro")
PRO_MODEL = "wan2.7-image-pro"
SIZE_CHOICES = ("1K", "2K", "4K")
TIER_EDGES = {"1K": 1024, "2K": 2048, "4K": 4096}
RATIO_CHOICES = ("auto", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9")
ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
REFERENCE_SLOTS = 3
MAX_REFERENCES = 9
MAX_PROMPT_CHARS = 5000
MAX_SEED = 2147483647
MIN_REFERENCE_EDGE = 240
MAX_REFERENCE_RATIO = 8
MAX_REFERENCE_BYTES = 20 << 20
MIN_SUBMIT_TIMEOUT = 600
UNCERTAIN_HINT = "提交结果不确定，请先到任务日志确认，不要重复生成"
EMPTY_HINT = "平台返回成功但没有图像，请先到任务日志确认是否扣费，不要重复生成"
DOWNLOAD_HINT = "图像已生成并按张扣费，但下载失败；图像URL 输出是原图地址（24 小时内有效），不要重复生成"


def resolve_size(model, size, ratio, reference_count):
    """auto sends the tier, which follows the last reference image (square without one).

    A ratio becomes an explicit W*H inside the tier's pixel budget, rounded down
    to multiples of 16 so it never exceeds the budget.
    """
    size = str(size or "").strip().upper()
    if size not in TIER_EDGES:
        raise ImageChannelError(f"尺寸只支持 {'/'.join(SIZE_CHOICES)}，当前 {size or '空'}")
    if size == "4K" and (model != PRO_MODEL or reference_count):
        raise ImageChannelError("4K 只支持 wan2.7-image-pro 文生图（不接参考图），请改用 1K/2K")
    ratio = str(ratio or "").strip() or "auto"
    if ratio not in RATIO_CHOICES:
        raise ImageChannelError(f"比例只支持 {'/'.join(RATIO_CHOICES)}，当前 {ratio}")
    if ratio == "auto":
        return size
    width_part, height_part = (int(part) for part in ratio.split(":"))
    budget = TIER_EDGES[size] ** 2
    width = math.isqrt(budget * width_part // height_part) // 16 * 16
    height = math.isqrt(budget * height_part // width_part) // 16 * 16
    return f"{width}*{height}"


def collect_reference_frames(images):
    """Split connected IMAGE batches into single frames, in slot then batch order."""
    frames = []
    for image in images:
        if image is None:
            continue
        shape = getattr(image, "shape", None)
        count = int(shape[0]) if shape is not None and len(shape) == 4 else 1
        frames.extend(image[index : index + 1] if count > 1 else image for index in range(count))
    if len(frames) > MAX_REFERENCES:
        raise ImageChannelError(f"最多支持 {MAX_REFERENCES} 张参考图，当前 {len(frames)} 张。")
    for index, frame in enumerate(frames, start=1):
        width, height = image_tensor_size(frame)
        if min(width, height) < MIN_REFERENCE_EDGE:
            raise ImageChannelError(f"第 {index} 张参考图 {width}×{height}，宽和高都不能小于 {MIN_REFERENCE_EDGE} 像素。")
        if max(width, height) > MAX_REFERENCE_RATIO * min(width, height):
            raise ImageChannelError(f"第 {index} 张参考图 {width}×{height}，宽高比须在 1:8 到 8:1 之间。")
    return frames


def encode_references(frames):
    """JPEG data URLs: upstream rejects PNG alpha, and JPEG keeps the inline body small."""
    urls = []
    for index, frame in enumerate(frames, start=1):
        data = jl_reference_bytes(frame)
        if not data or len(data) > MAX_REFERENCE_BYTES:
            raise ImageChannelError(f"第 {index} 张参考图为空或超过 {MAX_REFERENCE_BYTES >> 20} MiB。")
        urls.append(f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}")
    return urls


def build_fields(model, prompt, size, count, seed, watermark):
    # No response_format: asking for b64_json makes the platform download the
    # result itself and silently drop a failed download while still billing it.
    fields = {"model": model, "prompt": prompt, "size": size, "n": count, "watermark": bool(watermark)}
    if seed >= 0:
        fields["seed"] = seed
    return fields


def resolve_wan27_config(api_key_override=""):
    config = get_config()
    provider = normalize_provider(config.get("api_provider"))
    if provider not in ALLOWED_PROVIDERS:
        raise ImageChannelError("Wan2.7 图像只支持灵流线路，请在设置里切换到灵流线路。")
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


def _submit_uncertain(exc):
    """A 4xx is a definite refusal; anything else may have been generated and billed."""
    status_code = getattr(exc, "status_code", None)
    return not (isinstance(exc, ImageChannelError) or (status_code and 400 <= int(status_code) < 500))


def _result_summary(payload, items):
    """Image URLs stay in the 图像URL output; the raw upstream metadata is dropped."""
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    revised = [str(item.get("revised_prompt")) for item in items if item.get("revised_prompt")]
    summary = {"images": len(items)}
    if revised:
        summary["revised_prompt"] = revised
    if isinstance(usage, dict):
        summary["usage"] = usage
    return summary


class LingliuWan27Image(LingliuGoogleBase):
    CATEGORY = "Lingliu/Wan"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图像URL", "响应")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": "", "tooltip": f"最多 {MAX_PROMPT_CHARS} 字；接了参考图时可以留空。"}),
                "模型": (list(MODELS), {"default": MODELS[0]}),
                "尺寸": (list(SIZE_CHOICES), {"default": "2K", "tooltip": "4K 只支持 wan2.7-image-pro 文生图（不接参考图）。"}),
                "比例": (list(RATIO_CHOICES), {"default": "auto", "tooltip": "auto：接参考图时跟随最后一张参考图，不接时为正方形；选比例时按所选尺寸的像素总量换算宽高。"}),
                "张数": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "1–4 张，按成功生成的张数计费。"}),
            },
            "optional": {
                "参考图1": ("IMAGE", {"tooltip": f"三个参考图口合计最多 {MAX_REFERENCES} 张，可接图像批次；顺序按参考图1→3、批次内顺序。"}),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": MAX_SEED, "tooltip": "-1 为随机（不发送种子）。"}),
                "水印": ("BOOLEAN", {"default": False, "tooltip": "开启后在右下角加「AI生成」字样。"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _json(self, envelope, secrets):
        return _mask(self.safe_json(envelope), secrets)

    async def generate(self, 提示词, 模型, 尺寸, 比例, 张数, 参考图1=None, 参考图2=None, 参考图3=None,
                       API密钥="", 随机种子=-1, 水印=False, unique_id=None):
        model = str(模型 or "").strip()
        secrets = {str(API密钥 or "").strip()}
        envelope = {"node": NODE_ID, "model": model}
        image_url = ""
        stage = "prepare"
        try:
            if model not in MODELS:
                raise ImageChannelError(f"不支持的 Wan2.7 图像模型：{model}")
            count, seed = int(张数), int(随机种子)
            if not 1 <= count <= 4:
                raise ImageChannelError("张数须在 1–4 之间")
            if not -1 <= seed <= MAX_SEED:
                raise ImageChannelError(f"随机种子须为 -1（随机）或 0–{MAX_SEED}")
            prompt = str(提示词 or "").strip()
            if len(prompt) > MAX_PROMPT_CHARS:
                raise ImageChannelError(f"提示词最多 {MAX_PROMPT_CHARS} 字（超出部分会被截掉），当前 {len(prompt)} 字。")
            frames = collect_reference_frames((参考图1, 参考图2, 参考图3))
            if not prompt and not frames:
                raise ImageChannelError("提示词和参考图至少要有一个。")
            size = resolve_size(model, 尺寸, 比例, len(frames))
            config = resolve_wan27_config(API密钥)
            secrets.add(config["api_key"])
            references = await asyncio.to_thread(encode_references, frames)
            fields = build_fields(model, prompt, size, count, seed, 水印)
            envelope.update({"request": dict(fields), "reference_images": len(references)})
            if references:
                fields["images"] = references
            stage = "submit"
            timeout = max(self.timeout(config), MIN_SUBMIT_TIMEOUT)
            self.publish_live_message(unique_id, f"生成中：{model} {size} ×{count}（同步等待，最长约 {timeout} 秒，请不要重复提交）", append=False)
            payload = await submit_once(
                self, self.endpoint(config, "images_generate_path"), config,
                timeout=timeout, headers=self.auth_headers(config, json_mode=True), json=fields,
            )
            stage = "result"
            data = payload.get("data") if isinstance(payload, dict) else None
            items = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
            if not any(item.get("url") or item.get("b64_json") for item in items):
                raise RuntimeError(EMPTY_HINT)
            image_url = next((str(item["url"]) for item in items if item.get("url")), "")
            envelope.update(_result_summary(payload, items))
            stage = "download"
            tensor, image_url = await self.decode_image_items(items, timeout=self.timeout(config))
            envelope["status"] = "completed"
            self.publish_live_message(unique_id, f"完成：{int(tensor.shape[0])} 张", append=False)
            return tensor, image_url, self._json(envelope, secrets)
        except ImageSubmitUncertain as exc:
            message = str(exc)
        except Exception as exc:
            message = str(mask_provider_text(exc)).rstrip("。")
            if stage == "submit" and _submit_uncertain(exc):
                message += f"；{UNCERTAIN_HINT}"
            elif stage == "download":
                message += f"；{DOWNLOAD_HINT}"
        message = _mask(message, secrets)
        self.publish_live_message(unique_id, f"错误: {message}", append=False)
        envelope.update({"status": "error", "error": {"stage": stage, "message": message}})
        return blank_image_tensor(), image_url, self._json(envelope, secrets)
