"""Shared LL/JL channel routing for the Lingliu image nodes (V16.3, references V16.4).

LL is the original channel and JL is the second channel.  A request selects a
channel with the platform's key suffix ``sk-xxx-<channel id>``.  Channel IDs
are internal: the UI only shows the LL/JL abbreviations.

Both Lingliu lines share one channel table.  ``image_channel_overrides`` in
lingliu_api.json can replace any ID without a code change, either globally
(``{"model": {"JL": 89}}``) or per line (``{"lingliu": {"model": {...}}}``).

Before a request is pinned to a channel, the line's public /api/pricing is
read.  JL is refused when the model is not published in the JL group, and
LL keeps the historical unsuffixed request until JL is live on that line.
This keeps production unchanged until the platform release, and makes the
release usable without changing the plugin.
"""

from __future__ import annotations

from ..upstream_mask import JL, is_group

import asyncio
import json
import re
import threading
import time
from io import BytesIO

import requests
from PIL import Image

from ..lingliu_config import LINGLIU_PROVIDER, _read_config_file, build_url
from ..utils import tensor2pil

CHANNEL_LL = "LL"
CHANNEL_JL = "JL"
CHANNELS = [CHANNEL_LL, CHANNEL_JL]
CHANNEL_DEFAULT_NEW_NODE = CHANNEL_JL
CHANNEL_DEFAULT_LEGACY = CHANNEL_LL
CHANNEL_PROVIDERS = {LINGLIU_PROVIDER}

# Local platform channel IDs.  Production must use the same IDs, otherwise
# set image_channel_overrides in lingliu_api.json.
DEFAULT_CHANNEL_IDS = {
    "nano-banana-2": {CHANNEL_LL: 4, CHANNEL_JL: 85},
    "nano-banana-2-2k": {CHANNEL_LL: 4, CHANNEL_JL: 85},
    "nano-banana-2-4k": {CHANNEL_LL: 4, CHANNEL_JL: 85},
    "nano-banana-2.1": {CHANNEL_LL: 91, CHANNEL_JL: 89},
    "nano-banana-pro": {CHANNEL_LL: 4, CHANNEL_JL: 68},
    "nano-banana-pro-2k": {CHANNEL_LL: 4, CHANNEL_JL: 68},
    "nano-banana-pro-4k": {CHANNEL_LL: 4, CHANNEL_JL: 68},
    "gpt-image-2": {CHANNEL_LL: 3, CHANNEL_JL: 86},
    "gpt-image-2-1k": {CHANNEL_JL: 83},
    "gpt-image-2.5-flare": {CHANNEL_LL: 3, CHANNEL_JL: 77},
    "gpt-image-2.5-sunburst": {CHANNEL_LL: 3, CHANNEL_JL: 75},
    "gpt-image-2.5-flare-1k": {CHANNEL_JL: 78},
    "gpt-image-2.5-sunburst-1k": {CHANNEL_JL: 79},
    "gpt-image-2.5-1k": {CHANNEL_JL: 80},
}

# JL public capabilities intersected with the platform's published pixel
# table (ratios outside that table are rejected by the platform).
_BANANA2_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
JL_MODEL_CONTRACTS = {
    "nano-banana-2": {"ratios": _BANANA2_RATIOS, "resolutions": ["1K", "2K", "4K"], "max_refs": 6},
    "nano-banana-2-2k": {"ratios": _BANANA2_RATIOS, "resolutions": ["2K"], "max_refs": 6},
    "nano-banana-2-4k": {"ratios": _BANANA2_RATIOS, "resolutions": ["4K"], "max_refs": 6},
    "nano-banana-pro": {"ratios": _BANANA2_RATIOS, "resolutions": ["1K", "2K", "4K"], "max_refs": 6},
    "nano-banana-pro-2k": {"ratios": _BANANA2_RATIOS, "resolutions": ["2K"], "max_refs": 6},
    "nano-banana-pro-4k": {"ratios": _BANANA2_RATIOS, "resolutions": ["4K"], "max_refs": 6},
    "nano-banana-2.1": {
        "ratios": ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9"],
        "resolutions": ["1K", "2K", "4K"],
        "max_refs": 6,
    },
    "gpt-image-2": {
        "ratios": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        "resolutions": ["1K", "2K", "4K"],
        "max_refs": 10,
    },
    "gpt-image-2-1k": {"ratios": ["1:1", "16:9", "9:16", "4:3", "3:4"], "resolutions": ["1K"], "max_refs": 15},
    "gpt-image-2.5-flare": {"ratios": ["1:1", "16:9", "9:16", "4:3", "3:2", "3:4", "2:3"], "resolutions": ["1K", "2K", "4K"], "max_refs": 10},
    "gpt-image-2.5-sunburst": {"ratios": ["1:1", "16:9", "9:16", "5:4", "4:3", "3:2", "3:4"], "resolutions": ["1K", "2K", "4K"], "max_refs": 10},
    "gpt-image-2.5-flare-1k": {"ratios": ["1:1", "16:9", "9:16", "5:4", "4:3", "3:2", "4:5", "3:4"], "resolutions": ["1K"], "max_refs": 15},
    "gpt-image-2.5-sunburst-1k": {"ratios": ["1:1", "16:9", "9:16", "4:3", "21:9", "4:5", "3:4"], "resolutions": ["1K"], "max_refs": 15},
    "gpt-image-2.5-1k": {
        "ratios": ["1:1", "16:9", "9:16", "5:4", "4:3", "3:2", "21:9", "4:5", "3:4", "2:3"],
        "resolutions": ["1K"],
        "max_refs": 15,
    },
    # V17.4 优惠渠道2 (GPT Image 2.5 node): the alias selects the channel, so these have no channel ID.
    "gpt-image-2.5-flare-all": {"ratios": ["1:1", "16:9", "9:16", "4:3", "3:2", "21:9", "3:4", "2:3"], "resolutions": ["1K", "2K", "4K"], "max_refs": 10},
    "gpt-image-2.5-sunburst-all": {"ratios": ["1:1", "16:9", "9:16", "3:2", "4:3", "3:4", "2:3"], "resolutions": ["1K", "2K", "4K"], "max_refs": 10},
}
JL_MAX_REFERENCE_BYTES = 25 << 20
JL_REFERENCE_MAX_EDGE = 4096
JL_REFERENCE_JPEG_QUALITY = 92
JL_QUALITIES = ["auto", "low", "medium", "high"]

PRICING_CACHE_SECONDS = 300
_pricing_cache = {}
_pricing_lock = threading.Lock()

FLAT_PENDING = {"queued", "in_progress", "submitted", "pending", "processing", "running", "unknown"}
FLAT_SUCCESS = {"completed"}
FLAT_FAILURE = {"failed", "cancelled", "canceled"}

UNKNOWN_SUBMIT_MESSAGE = "提交结果待确认，请勿重复生成"


class ImageChannelError(RuntimeError):
    """A local, pre-submit refusal.  Nothing was sent or charged."""


class ImageSubmitUncertain(RuntimeError):
    """The platform accepted funds but could not confirm the upstream submit."""

    def __init__(self, task_id, message=UNKNOWN_SUBMIT_MESSAGE):
        self.task_id = str(task_id or "")
        super().__init__(f"{message}（Task ID: {self.task_id}）" if self.task_id else message)


class ImageTaskFailed(RuntimeError):
    """The image task reached a terminal failure."""


def normalize_channel(value, default=CHANNEL_DEFAULT_LEGACY):
    text = str(value or "").strip().upper()
    return text if text in CHANNELS else default


def channel_applies(provider):
    return str(provider or "").strip() in CHANNEL_PROVIDERS


def effective_channel(provider, channel):
    """Non-Lingliu lines are always LL; the channel input is ignored there."""
    if not channel_applies(provider):
        return CHANNEL_LL
    return normalize_channel(channel)


def _load_overrides(provider):
    try:
        loaded = _read_config_file().get("image_channel_overrides") or {}
    except Exception:
        loaded = {}
    if not isinstance(loaded, dict):
        return {}
    merged = {}
    for key, value in loaded.items():
        if key not in CHANNEL_PROVIDERS and isinstance(value, dict):
            merged[key] = value
    scoped = loaded.get(provider)
    if isinstance(scoped, dict):
        for key, value in scoped.items():
            if isinstance(value, dict):
                merged[key] = {**merged.get(key, {}), **value}
    return merged


def channel_id_for(provider, model, channel):
    entry = dict(DEFAULT_CHANNEL_IDS.get(str(model or "").strip(), {}))
    override = _load_overrides(provider).get(str(model or "").strip())
    if isinstance(override, dict):
        for key, value in override.items():
            name = normalize_channel(key, default="")
            if name:
                entry[name] = value
    value = entry.get(channel)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def force_channel_key(api_key, channel_id):
    key = str(api_key or "").strip()
    channel = str(channel_id or "").strip()
    if not key or not channel.isdigit():
        return key
    base_key = re.sub(r"-\d+$", "", key)
    return f"{base_key}-{channel}"


def config_with_channel(config, channel_id):
    pinned = dict(config or {})
    if channel_id:
        pinned["api_key"] = force_channel_key(pinned.get("api_key"), channel_id)
    return pinned


def _fetch_pricing_groups(base_url, timeout=15):
    response = requests.get(build_url(base_url, "/api/pricing"), timeout=timeout, headers={"Accept": "application/json"})
    response.raise_for_status()
    payload = response.json()
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("pricing response has no model list")
    groups = {}
    for item in items:
        if isinstance(item, dict) and item.get("model_name"):
            groups[str(item["model_name"])] = {str(group) for group in (item.get("enable_groups") or [])}
    return groups


def pricing_groups(base_url, *, fetcher=None, now=None):
    """Return {model: enable_groups} for a line, cached per process."""
    key = str(base_url or "").rstrip("/")
    current = time.monotonic() if now is None else now
    with _pricing_lock:
        cached = _pricing_cache.get(key)
        if cached and current - cached[0] < PRICING_CACHE_SECONDS:
            return cached[1]
    groups = (fetcher or _fetch_pricing_groups)(key)
    with _pricing_lock:
        _pricing_cache[key] = (current, groups)
    return groups


def clear_pricing_cache():
    with _pricing_lock:
        _pricing_cache.clear()


def _is_jl_group(group):
    # The platform splits JL channels into sub-groups named <group>-<suffix>.
    return is_group(group, JL)


def jl_live(base_url, model, *, fetcher=None):
    groups = pricing_groups(base_url, fetcher=fetcher).get(str(model or "").strip(), set())
    return any(_is_jl_group(group) for group in groups)


async def resolve_route(config, model, channel, *, fetcher=None):
    """Resolve (channel, channel_id) for one request; raise before any upload or submit.

    Returns channel_id None when the request must stay unsuffixed.
    """
    provider = str((config or {}).get("api_provider") or "").strip()
    channel = effective_channel(provider, channel)
    if not channel_applies(provider):
        return channel, None
    model = str(model or "").strip()
    base_url = (config or {}).get("base_url")
    channel_id = channel_id_for(provider, model, channel)
    if channel == CHANNEL_JL:
        if model not in JL_MODEL_CONTRACTS:
            raise ImageChannelError(f"JL 渠道不支持模型 {model}，请切换到 LL 或更换模型。")
        if not channel_id:
            raise ImageChannelError("当前线路暂未开通 JL 渠道。")
        try:
            live = await asyncio.to_thread(jl_live, base_url, model, fetcher=fetcher)
        except Exception as exc:
            raise ImageChannelError(f"无法确认当前线路的 JL 渠道状态，已停止提交（未扣费）：{type(exc).__name__}") from exc
        if not live:
            raise ImageChannelError(f"该线路 JL 尚未上线（模型 {model}），已停止提交（未扣费）。请切换到 LL。")
        return channel, channel_id
    if model not in DEFAULT_CHANNEL_IDS and not channel_id:
        return channel, None
    if not channel_id:
        raise ImageChannelError(f"模型 {model} 只能使用 JL 渠道。")
    # Pin LL only once JL is live on this line; before that an unsuffixed
    # request already reaches the original channel, exactly as in V16.2.
    try:
        live = await asyncio.to_thread(jl_live, base_url, model, fetcher=fetcher)
    except Exception:
        live = False
    return channel, (channel_id if live else None)


def jl_contract(model):
    contract = JL_MODEL_CONTRACTS.get(str(model or "").strip())
    if not contract:
        raise ImageChannelError(f"JL 渠道不支持模型 {model}。")
    return contract


def validate_jl_ratio(model, ratio):
    ratio = str(ratio or "").strip()
    if ratio and ratio != "auto" and ratio not in jl_contract(model)["ratios"]:
        raise ImageChannelError(f"JL 渠道的 {model} 不支持比例 {ratio}，可用：{'、'.join(jl_contract(model)['ratios'])}。")


def validate_jl_resolution(model, resolution):
    resolution = str(resolution or "").strip().upper()
    if resolution not in jl_contract(model)["resolutions"]:
        raise ImageChannelError(f"JL 渠道的 {model} 不支持清晰度 {resolution}，可用：{'、'.join(jl_contract(model)['resolutions'])}。")


def validate_jl_references(model, payloads):
    limit = jl_contract(model)["max_refs"]
    if len(payloads) > limit:
        raise ImageChannelError(f"JL 渠道的 {model} 最多支持 {limit} 张参考图，当前 {len(payloads)} 张。")
    for index, data in enumerate(payloads, start=1):
        if not data or len(data) > JL_MAX_REFERENCE_BYTES:
            raise ImageChannelError(f"第 {index} 张参考图为空或超过 25 MiB。")


def jl_reference_bytes(image):
    """Encode the first frame of an IMAGE as a JL reference: JPEG, long edge <= 4096.

    The platform forwards JL references to the JL channel inline before it returns a
    task ID, so upload size dominates the wait (a 4096px PNG is about 14 MiB,
    the same JPEG about 1.6 MiB).
    """
    pil_image = tensor2pil(image)[0].convert("RGB")
    if max(pil_image.size) > JL_REFERENCE_MAX_EDGE:
        pil_image.thumbnail((JL_REFERENCE_MAX_EDGE, JL_REFERENCE_MAX_EDGE), Image.LANCZOS)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=JL_REFERENCE_JPEG_QUALITY)
    return buffer.getvalue()


def nearest_ratio(width, height, allowed):
    width, height = float(width or 0), float(height or 0)
    if width <= 0 or height <= 0 or not allowed:
        return allowed[0] if allowed else "1:1"
    target = width / height

    def distance(ratio):
        numerator, denominator = (float(part) for part in ratio.split(":"))
        return abs(target - numerator / denominator)

    return min(allowed, key=distance)


def image_tensor_size(image):
    """Return (width, height) of the first frame of an IMAGE tensor."""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        return 0, 0
    if len(shape) == 4:
        return int(shape[2]), int(shape[1])
    return int(shape[1]), int(shape[0])


def parse_flat_image_task(payload):
    """Parse the platform's flat image task view.

    Returns None for other envelopes (for example the nested
    ``{"data": {"status": "SUCCESS", ...}}``), otherwise
    ``(state, items, error)`` with state in success/failure/pending.
    """
    if not isinstance(payload, dict) or isinstance(payload.get("data"), dict):
        return None
    status = payload.get("status")
    if not isinstance(status, str):
        return None
    status = status.strip().lower()
    if status in FLAT_SUCCESS:
        items = payload.get("data")
        if not isinstance(items, list):
            result = payload.get("result")
            items = result.get("data") if isinstance(result, dict) else []
        return "success", [item for item in (items or []) if isinstance(item, dict)], ""
    if status in FLAT_FAILURE:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        return "failure", [], str(message or "图像任务失败")
    if status in FLAT_PENDING:
        return "pending", [], ""
    return None


def _uncertain_task_id(response):
    try:
        body = response.json()
    except Exception:
        return None
    if isinstance(body, dict) and str(body.get("status") or "").lower() == "unknown" and body.get("task_id"):
        return str(body["task_id"])
    return None


def _submit_once_sync(node, url, config, timeout, kwargs):
    response = requests.request("POST", url, timeout=timeout, **kwargs)
    if response.status_code == 503:
        task_id = _uncertain_task_id(response)
        if task_id:
            raise ImageSubmitUncertain(task_id)
    node._raise_for_status_with_details(response, url)
    return node._decode_json_response(response, url)


async def submit_once(node, url, config, *, timeout, **kwargs):
    """POST exactly once.  A 503 with status unknown raises ImageSubmitUncertain."""
    return await asyncio.to_thread(_submit_once_sync, node, url, config, timeout, kwargs)


async def wait_flat_image_task(node, config, task_id, *, max_poll_attempts, poll_interval, transient_limit=12):
    """Poll /v1/images/tasks/{id} until a terminal flat status; never resubmits."""
    url = build_url(config.get("base_url"), f"/v1/images/tasks/{task_id}")
    last_payload = None
    transient_errors = 0
    for _ in range(max(1, int(max_poll_attempts))):
        await asyncio.sleep(max(1, int(poll_interval)))
        try:
            payload = await node.async_request_json("GET", url, config, headers=node.headers(config))
            transient_errors = 0
        except Exception as exc:
            if node.is_transient_request_error(exc):
                transient_errors += 1
                if transient_errors < transient_limit:
                    continue
            raise
        last_payload = payload
        parsed = parse_flat_image_task(payload)
        if parsed is None:
            continue
        state, items, error = parsed
        if state == "success":
            if not items:
                raise ImageTaskFailed(f"任务 {task_id} 已完成，但响应中没有图像。")
            return items, payload
        if state == "failure":
            raise ImageTaskFailed(f"任务 {task_id} 失败：{error}")
    raise TimeoutError(f"任务 {task_id} 轮询超时，请稍后用任务查询节点查询，不要重复生成。最后状态：{json.dumps(node.public_payload(last_payload), ensure_ascii=False)[:500]}")


__all__ = [
    "CHANNELS",
    "CHANNEL_DEFAULT_LEGACY",
    "CHANNEL_DEFAULT_NEW_NODE",
    "CHANNEL_JL",
    "CHANNEL_LL",
    "DEFAULT_CHANNEL_IDS",
    "ImageChannelError",
    "ImageSubmitUncertain",
    "ImageTaskFailed",
    "JL_MODEL_CONTRACTS",
    "JL_QUALITIES",
    "channel_applies",
    "channel_id_for",
    "clear_pricing_cache",
    "config_with_channel",
    "effective_channel",
    "force_channel_key",
    "image_tensor_size",
    "jl_contract",
    "nearest_ratio",
    "normalize_channel",
    "parse_flat_image_task",
    "pricing_groups",
    "resolve_route",
    "submit_once",
    "validate_jl_ratio",
    "validate_jl_references",
    "validate_jl_resolution",
    "wait_flat_image_task",
]
