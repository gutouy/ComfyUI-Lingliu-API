"""JL video node: models the platform serves only through its JL group.

JL_VIDEO_CONTRACTS mirrors the platform's per-model capability table, so an
unsupported combination stops locally before any upload or paid submit.
Upload, strict single submit, polling and download reuse MiniMaxH3Workflow.
"""

import asyncio
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

from comfy.comfy_types import IO

from ..upstream_mask import mask_upstream
from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    get_config,
    mask_provider_payload,
    mask_provider_text,
    normalize_provider,
)
from ..utils import LingliuNullVideoAdapter
from .lingliu_image_channels import JL_MAX_REFERENCE_BYTES, jl_reference_bytes, jl_live
from .nodes_minimax_h3 import (
    H3Error,
    H3Result,
    MiniMaxH3Workflow,
    MTH3VideoAdapter,
    PreparedAsset,
    _extract_task_id,
    _read_media_input,
    _redact,
    _redact_text,
    _waveform_to_wav,
)
from .nodes_openai_compatible import LingliuBaseNode

NODE_ID = "lingliu_jl_video"
ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
MAX_REFERENCE_IMAGES = 30
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 10
# The platform accepts reference files up to 50 MiB.
MAX_MEDIA_BYTES = 50 << 20


@dataclass(frozen=True)
class JLVideoContract:
    min_seconds: int
    max_seconds: int
    excluded_seconds: int
    resolutions: tuple
    ratios: tuple
    max_images: int
    max_videos: int
    max_audios: int
    frames: bool
    # "supported": the audio switch is sent; "fixed"/"": model default audio.
    audio: str


JL_VIDEO_CONTRACTS = {
    "seedance-2.5": JLVideoContract(4, 25, 0, ("480p", "720p"), ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"), 30, 0, 10, False, "fixed"),
    "seedance-2.0": JLVideoContract(15, 15, 0, ("1080p",), ("16:9", "9:16"), 9, 3, 3, False, ""),
    "seedance-dola-2.5": JLVideoContract(30, 30, 0, ("720p",), ("16:9", "9:16"), 9, 0, 0, False, ""),
    "veo-3.1": JLVideoContract(8, 8, 0, ("720p",), ("16:9", "9:16"), 3, 0, 0, True, "supported"),
    "veo-3.1-fast": JLVideoContract(8, 8, 0, ("720p",), ("16:9", "9:16"), 3, 0, 0, True, ""),
    "veo-3.1-1080p": JLVideoContract(8, 8, 0, ("1080p",), ("16:9", "9:16"), 3, 0, 0, True, ""),
    "veo-3.1-fast-1080p": JLVideoContract(8, 8, 0, ("1080p",), ("16:9", "9:16"), 3, 0, 0, True, ""),
    "grok-imagine-video": JLVideoContract(1, 15, 7, ("480p", "720p"), ("1:1", "16:9", "9:16", "3:2", "4:3", "3:4", "2:3"), 7, 0, 0, False, ""),
    "minimax-h3-optimized": JLVideoContract(4, 15, 0, ("768p",), ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"), 9, 3, 3, True, ""),
    "grok-video": JLVideoContract(6, 6, 0, ("720p",), ("2:3", "3:2", "1:1", "9:16", "16:9"), 0, 0, 0, False, ""),
}
JL_VIDEO_MODELS = tuple(JL_VIDEO_CONTRACTS)
RATIO_CHOICES = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "3:2", "2:3")
RESOLUTION_CHOICES = ("480p", "720p", "768p", "1080p")

PENDING_STATES = {"", "unknown", "not_start", "submitted", "queued", "pending", "running", "processing", "in_progress"}
FAILED_STATES = {"failed", "failure", "error", "rejected", "cancelled", "canceled", "expired"}
SUCCESS_STATES = {"completed", "complete", "succeeded", "success"}
QUERY_HINT = "原任务可能仍在生成，请使用任务 ID 查询原任务，不要重复提交生成"
TIMEOUT_MESSAGE = "轮询超时。 原任务可能仍在生成，请使用任务 ID 查询原任务，不要重复提交生成。"

ROLE_LABELS = {
    "start_frame": "首帧图",
    "end_frame": "尾帧图",
    "reference_image": "参考图",
    "reference_video": "参考视频",
    "audio_reference": "参考音频",
}
ROLE_MEDIA = {
    "start_frame": "image",
    "end_frame": "image",
    "reference_image": "image",
    "reference_video": "video",
    "audio_reference": "audio",
}


def resolve_controls(model, duration, ratio, resolution):
    """Check the widgets against the model contract before anything is sent."""
    contract = JL_VIDEO_CONTRACTS.get(str(model or "").strip())
    if contract is None:
        raise H3Error(f"不支持的 JL 视频模型：{model}")
    notices = []
    try:
        seconds = int(duration)
    except (TypeError, ValueError) as exc:
        raise H3Error(f"时长必须是整数秒：{duration}") from exc
    if contract.min_seconds == contract.max_seconds:
        if seconds != contract.min_seconds:
            notices.append(f"该模型固定 {contract.min_seconds} 秒，已按 {contract.min_seconds} 秒提交")
        seconds = contract.min_seconds
    elif not contract.min_seconds <= seconds <= contract.max_seconds:
        raise H3Error(f"{model} 时长须在 {contract.min_seconds}–{contract.max_seconds} 秒之间，当前 {seconds} 秒")
    elif contract.excluded_seconds and seconds == contract.excluded_seconds:
        raise H3Error(
            f"{model} 不支持 {seconds} 秒，请选 {contract.min_seconds}–{contract.max_seconds} 秒中的其他时长"
        )
    selected_resolution = str(resolution or "").strip().lower()
    if len(contract.resolutions) == 1:
        if selected_resolution != contract.resolutions[0]:
            notices.append(f"该模型固定 {contract.resolutions[0]}，已按 {contract.resolutions[0]} 提交")
        selected_resolution = contract.resolutions[0]
    elif selected_resolution not in contract.resolutions:
        raise H3Error(f"{model} 清晰度只支持 {'/'.join(contract.resolutions)}，当前 {resolution}")
    selected_ratio = str(ratio or "").strip()
    if selected_ratio not in contract.ratios:
        raise H3Error(f"{model} 比例只支持 {'/'.join(contract.ratios)}，当前 {ratio}")
    return contract, seconds, selected_resolution, selected_ratio, notices


def plan_assets(model, contract, prompt, raw_assets):
    """Check connected slots against the contract before encoding anything.

    raw_assets is [(slot, role, value)]; empty slots are skipped and do not
    use up a reference position. Returns (mode, [(slot, role, value)]).
    """
    planned = [(slot, role, value) for slot, role, value in raw_assets if value is not None]
    counts = {role: 0 for role in ROLE_LABELS}
    for _slot, role, _value in planned:
        counts[role] += 1
    has_frames = counts["start_frame"] or counts["end_frame"]
    has_refs = counts["reference_image"] or counts["reference_video"] or counts["audio_reference"]
    if has_frames:
        if not contract.frames:
            hint = "请改接到参考图接口" if contract.max_images else "请断开首帧图/尾帧图"
            raise H3Error(f"{model} 不支持首尾帧，{hint}")
        if not counts["start_frame"]:
            raise H3Error("接了尾帧图时必须同时接首帧图")
        if has_refs:
            raise H3Error("首尾帧不能和参考图/参考视频/参考音频同时使用，请只保留一种")
        return "frame", planned
    if has_refs and not (contract.max_images or contract.max_videos or contract.max_audios):
        raise H3Error(f"{model} 不支持参考素材，只能文生视频，请断开所有素材接口")
    limits = (
        ("reference_image", contract.max_images),
        ("reference_video", contract.max_videos),
        ("audio_reference", contract.max_audios),
    )
    for role, limit in limits:
        if counts[role] > limit:
            if limit == 0:
                raise H3Error(f"{model} 不支持{ROLE_LABELS[role]}，请断开{ROLE_LABELS[role]}接口")
            raise H3Error(f"{model} 最多 {limit} 个{ROLE_LABELS[role]}，当前接了 {counts[role]} 个")
    if counts["audio_reference"] and not (counts["reference_image"] or counts["reference_video"]):
        raise H3Error("参考音频需要同时接参考图或参考视频")
    mode = "asset" if has_refs else "text"
    if not str(prompt or "").strip() and not counts["reference_image"]:
        raise H3Error("提示词不能为空（只接参考图或首帧图时可以留空）")
    return mode, planned


def _encode_asset(slot, role, value, position):
    """Encode one connected input. Filenames stay ASCII for the upload form."""
    media_type = ROLE_MEDIA[role]
    base = role if role in {"start_frame", "end_frame"} else f"{role}_{position}"
    if media_type == "image":
        batch = MiniMaxH3Workflow._image_batch_size(value)
        if batch != 1:
            raise H3Error(f"{slot} 只能接 1 张图片，当前批次 {batch} 张")
        data = jl_reference_bytes(value)
        if len(data) > JL_MAX_REFERENCE_BYTES:
            raise H3Error(f"{slot} 编码后超过 {JL_MAX_REFERENCE_BYTES >> 20}MB")
        return PreparedAsset(slot, "image", role, f"{base}.jpg", "image/jpeg", data)
    if media_type == "audio" and isinstance(value, Mapping) and "waveform" in value:
        data, source_name = _waveform_to_wav(value)
    else:
        default_suffix = ".mp4" if media_type == "video" else ".wav"
        data, source_name = _read_media_input(value, default_suffix)
    suffix = os.path.splitext(str(source_name or ""))[1].lower()
    if media_type == "video":
        allowed = {".mp4": "video/mp4", ".mov": "video/quicktime"}
    else:
        allowed = {".wav": "audio/wav", ".mp3": "audio/mpeg"}
    if suffix not in allowed:
        raise H3Error(f"{slot} 只支持 {'/'.join(ext.lstrip('.') for ext in allowed)} 格式，当前 {suffix or '未知'}")
    if not data:
        raise H3Error(f"{slot} 内容为空")
    if len(data) > MAX_MEDIA_BYTES:
        raise H3Error(f"{slot} 超过 {MAX_MEDIA_BYTES >> 20}MB")
    return PreparedAsset(slot, media_type, role, f"{base}{suffix}", allowed[suffix], data)


def prepare_assets(planned):
    prepared = []
    positions = {}
    for slot, role, value in planned:
        positions[role] = positions.get(role, 0) + 1
        try:
            prepared.append(_encode_asset(slot, role, value, positions[role]))
        except H3Error:
            raise
        except Exception as exc:
            raise H3Error(f"{slot} 读取失败：{exc}") from exc
    return prepared


def build_body(contract, model, prompt, seconds, resolution, ratio, mode, prepared, urls, generate_audio):
    # No size field: the platform derives size from resolution and ratio.
    body = {
        "model": model,
        "prompt": str(prompt or "").strip(),
        "seconds": seconds,
        "resolution": resolution,
        "aspect_ratio": ratio,
        "response_format": "url",
    }
    by_role = {}
    for asset, url in zip(prepared, urls):
        by_role.setdefault(asset.role, []).append(url)
    if mode == "frame":
        body["start_frame"] = by_role["start_frame"][0]
        if by_role.get("end_frame"):
            body["end_frame"] = by_role["end_frame"][0]
        body["reference_mode"] = "frame"
    elif mode == "asset":
        for role, field in (
            ("reference_image", "reference_images"),
            ("reference_video", "reference_videos"),
            ("audio_reference", "audio_references"),
        ):
            if by_role.get(role):
                body[field] = by_role[role]
        body["reference_mode"] = "asset"
    if contract.audio == "supported":
        body["audio"] = bool(generate_audio)
    return body


def resolve_jl_config(api_key_override=""):
    config = get_config()
    provider = normalize_provider(config.get("api_provider"))
    if provider not in ALLOWED_PROVIDERS:
        raise H3Error("JL 视频只支持灵流线路，请在设置里切换到灵流线路", stage="route")
    raw_key = str(api_key_override or config.get("api_key") or "").strip()
    if not raw_key:
        raise H3Error("缺少 API 密钥，请在设置里填写或在节点里输入", stage="auth")
    return {
        # Selects the strict single-submit transport: create once, 2xx only, no redirects.
        "channel": "MT",
        "provider": provider,
        "base_url": str(config.get("base_url") or "").rstrip("/"),
        "timeout": int(config.get("timeout") or 300),
        "raw_key": raw_key,
        "files_key": raw_key,
        "h3_key": raw_key,
    }


def _origin(url):
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return ""
    return f"{parts.scheme}://{parts.netloc}".lower() if parts.scheme and parts.netloc else ""


def _upstream_message(payload):
    if not isinstance(payload, dict):
        return ""
    task = MiniMaxH3Workflow._task_payload(payload)
    for source in (task, payload):
        error = source.get("error")
        if isinstance(error, dict) and str(error.get("message") or "").strip():
            return str(error["message"]).strip()[:500]
        if isinstance(error, str) and error.strip():
            return error.strip()[:500]
        for key in ("fail_reason", "message"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
    return ""


URL_PATTERN = re.compile(r"https?://[^\s\"'<>()\[\]]+")


class JLVideoWorkflow(MiniMaxH3Workflow):
    """Contract check, live-group check, upload, single create, poll, download."""

    def __init__(self, http=None, sleep=None, pricing_fetcher=None):
        # This node does not probe media; skip building the PyAV probe.
        super().__init__(http=http, media_probe=object(), sleep=sleep)
        self.pricing_fetcher = pricing_fetcher
        self._origin = ""
        self._uploaded = {}

    def _remember_uploads(self, summaries):
        for summary in summaries or []:
            url = summary.get("url") if isinstance(summary, dict) else ""
            if url and url not in self._uploaded:
                self._uploaded[url] = f"<uploaded_reference_{len(self._uploaded) + 1}>"

    def _hide_urls(self, value):
        """Keep platform URLs; replace uploaded and third-party URLs with labels."""
        if isinstance(value, dict):
            return {key: self._hide_urls(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._hide_urls(item) for item in value]
        if not isinstance(value, str):
            return value

        def replace(match):
            url = match.group(0)
            if url in self._uploaded:
                return self._uploaded[url]
            if self._origin and _origin(url) == self._origin:
                return url
            return "<hidden_upstream_url>"

        return URL_PATTERN.sub(replace, value)

    def _mask_text(self, text, secrets):
        masked = mask_provider_text(_redact_text(self._hide_urls(str(text or "")), secrets))
        return mask_upstream(str(masked))

    def _json(self, payload, secrets):
        masked = mask_provider_payload(_redact(self._hide_urls(payload), secrets))
        return mask_upstream(json.dumps(masked, ensure_ascii=False, indent=2))

    async def _ensure_live(self, config, model):
        try:
            live = await asyncio.to_thread(jl_live, config["base_url"], model, fetcher=self.pricing_fetcher)
        except Exception as exc:
            raise H3Error(
                f"无法确认当前线路的 JL 渠道状态，已停止提交（未扣费）：{type(exc).__name__}",
                stage="route",
            ) from exc
        if not live:
            raise H3Error(f"该线路 JL 尚未上线（模型 {model}），已停止提交（未扣费）。", stage="route")

    async def execute(
        self,
        *,
        model,
        prompt,
        duration,
        ratio,
        resolution,
        raw_assets,
        api_key="",
        wait_for_completion=True,
        max_poll_attempts=180,
        poll_interval=10,
        generate_audio=True,
        notify=None,
    ):
        model = str(model or "").strip()
        secrets = {str(api_key or "").strip()}
        envelope = {"node": NODE_ID, "model": model}
        task_id = ""
        try:
            contract, seconds, resolution, ratio, notices = resolve_controls(model, duration, ratio, resolution)
            max_poll_attempts = int(max_poll_attempts)
            poll_interval = int(poll_interval)
            if not 1 <= max_poll_attempts <= 720:
                raise H3Error("最大轮询次数须在 1–720 之间")
            if not 2 <= poll_interval <= 60:
                raise H3Error("轮询间隔须在 2–60 秒之间")
            mode, planned = plan_assets(model, contract, prompt, raw_assets)
            if notices:
                envelope["notice"] = "；".join(notices)
            config = resolve_jl_config(api_key)
            secrets.add(config["raw_key"])
            self._origin = _origin(config["base_url"])
            envelope["provider"] = config["provider"]
            await self._ensure_live(config, model)
            prepared = await asyncio.to_thread(prepare_assets, planned)
            urls, uploads = await self.upload_all(prepared, config)
            self._remember_uploads(uploads)
            envelope["uploads"] = uploads
            body = build_body(contract, model, prompt, seconds, resolution, ratio, mode, prepared, urls, generate_audio)
            envelope["request"] = body
            create_response = await self._request_json(
                "POST",
                build_url(config["base_url"], "/v1/videos"),
                config,
                stage="create",
                key=config["h3_key"],
                retry_network=False,
                retry_status={429},
                extra_headers={"Content-Type": "application/json"},
                json=body,
            )
            envelope["create_response"] = create_response
            task_id = _extract_task_id(create_response)
            if not task_id:
                raise H3Error(
                    "创建响应缺少任务 ID；提交结果不确定，请先到任务日志确认，不要重复提交",
                    stage="create",
                    payload=create_response,
                )
            envelope.update({"task_id": task_id, "status": "submitted"})
            if notify:
                notify(task_id, "submitted")
            if not wait_for_completion:
                envelope["message"] = "任务已提交，未等待完成。"
                return H3Result(LingliuNullVideoAdapter("任务已提交，未等待完成。"), task_id, self._json(envelope, secrets), "")
            query_url = build_url(config["base_url"], f"/v1/videos/{quote(task_id, safe='')}")
            for _attempt in range(max_poll_attempts):
                query_response = await self._request_json(
                    "GET",
                    query_url,
                    config,
                    stage="query",
                    key=config["h3_key"],
                    retry_network=True,
                    retry_status={429},
                )
                envelope["query_response"] = query_response
                status = self._status(query_response)
                envelope["status"] = status or "pending"
                if status in FAILED_STATES:
                    reason = _upstream_message(query_response) or status
                    message = self._mask_text(f"生成失败：{reason}", secrets)
                    envelope["error"] = {"stage": "query", "message": message}
                    if notify:
                        notify(task_id, status)
                    return H3Result(LingliuNullVideoAdapter(message), task_id, self._json(envelope, secrets), "")
                if status in SUCCESS_STATES:
                    # Download through the platform's authenticated content endpoint.
                    video_url = f"{query_url}/content"
                    envelope["video_url"] = video_url
                    if notify:
                        notify(task_id, status)
                    adapter = MTH3VideoAdapter(video_url, config, self.http)
                    return H3Result(adapter, task_id, self._json(envelope, secrets), video_url)
                # Pending and unrecognised states keep polling until the budget runs out.
                await self.sleep(poll_interval)
            envelope.update({"status": "timeout", "message": TIMEOUT_MESSAGE})
            if notify:
                notify(task_id, "timeout")
            return H3Result(LingliuNullVideoAdapter(TIMEOUT_MESSAGE), task_id, self._json(envelope, secrets), "")
        except Exception as exc:
            stage = getattr(exc, "stage", "internal")
            payload = getattr(exc, "payload", None)
            details = dict(getattr(exc, "details", {}) or {})
            status_code = getattr(exc, "status_code", None)
            self._remember_uploads(details.get("completed_uploads"))
            message = str(exc).rstrip("。")
            upstream = _upstream_message(payload)
            if upstream and upstream not in message:
                message = f"{message}：{upstream}"
            if stage == "create" and status_code and int(status_code) >= 500:
                message += "；提交结果不确定，请先到任务日志确认，不要重复提交"
            if stage == "query" and task_id:
                message += f"；{QUERY_HINT}"
            message = self._mask_text(message, secrets)
            error_payload = {
                **envelope,
                "status": "error",
                "task_id": task_id,
                "error": {"stage": stage, "message": message, "status_code": status_code, "upstream": payload, "details": details},
            }
            if details.get("completed_uploads") is not None:
                error_payload["uploads"] = details["completed_uploads"]
            if details.get("failed_slot"):
                error_payload["failed_slot"] = details["failed_slot"]
            if notify and task_id:
                notify(task_id, "error")
            return H3Result(LingliuNullVideoAdapter(message), task_id, self._json(error_payload, secrets), "")


class LingliuJLVideo(LingliuBaseNode):
    CATEGORY = "Lingliu/JL"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL")
    FUNCTION = "generate_video"

    @classmethod
    def INPUT_TYPES(cls):
        slot_tip = "连接后自动出现下一个接口；空接口不占素材名额。"
        optional = {
            "API密钥": ("STRING", {"default": "", "password": True}),
            "等待完成": ("BOOLEAN", {"default": True}),
            "最大轮询次数": ("INT", {"default": 180, "min": 1, "max": 720}),
            "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60}),
            "生成音频": ("BOOLEAN", {"default": True, "tooltip": "仅 veo-3.1 可关闭；其他模型按模型默认。"}),
            "首帧图": ("IMAGE", {"tooltip": "仅 veo-3.1 系列与 minimax-h3-optimized 支持；不能和参考素材同时使用。"}),
            "尾帧图": ("IMAGE", {"tooltip": "需要同时接首帧图。"}),
        }
        for index in range(1, MAX_REFERENCE_IMAGES + 1):
            optional[f"参考图{index}"] = ("IMAGE", {"tooltip": slot_tip})
        for index in range(1, MAX_REFERENCE_VIDEOS + 1):
            optional[f"参考视频{index}"] = (IO.VIDEO, {"tooltip": slot_tip})
        for index in range(1, MAX_REFERENCE_AUDIOS + 1):
            optional[f"参考音频{index}"] = (IO.AUDIO, {"tooltip": slot_tip})
        return {
            "required": {
                "提示词": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "seedance-2.5 可用 @image1、@image2… 指代参考图，按已连接的参考图顺序编号。",
                    },
                ),
                "模型": (list(JL_VIDEO_MODELS), {"default": JL_VIDEO_MODELS[0], "tooltip": "各模型的时长/比例/清晰度/素材上限不同，提交前会先校验。"}),
                "时长": ("INT", {"default": 5, "min": 1, "max": 30, "tooltip": "固定时长的模型会按模型时长提交。"}),
                "比例": (list(RATIO_CHOICES), {"default": "16:9"}),
                "清晰度": (list(RESOLUTION_CHOICES), {"default": "720p", "tooltip": "固定清晰度的模型会按模型清晰度提交。"}),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _notify(self, unique_id):
        def notify(task_id, status):
            if task_id:
                self.publish_task_id(unique_id, task_id, append=False)
            self.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status}", append=False)

        return notify

    async def generate_video(self, **kwargs):
        raw_assets = [("首帧图", "start_frame", kwargs.get("首帧图")), ("尾帧图", "end_frame", kwargs.get("尾帧图"))]
        raw_assets.extend((f"参考图{i}", "reference_image", kwargs.get(f"参考图{i}")) for i in range(1, MAX_REFERENCE_IMAGES + 1))
        raw_assets.extend((f"参考视频{i}", "reference_video", kwargs.get(f"参考视频{i}")) for i in range(1, MAX_REFERENCE_VIDEOS + 1))
        raw_assets.extend((f"参考音频{i}", "audio_reference", kwargs.get(f"参考音频{i}")) for i in range(1, MAX_REFERENCE_AUDIOS + 1))
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)
        result = await JLVideoWorkflow().execute(
            model=kwargs.get("模型", JL_VIDEO_MODELS[0]),
            prompt=kwargs.get("提示词", ""),
            duration=kwargs.get("时长", 5),
            ratio=kwargs.get("比例", "16:9"),
            resolution=kwargs.get("清晰度", "720p"),
            raw_assets=raw_assets,
            api_key=kwargs.get("API密钥", ""),
            wait_for_completion=kwargs.get("等待完成", True),
            max_poll_attempts=kwargs.get("最大轮询次数", 180),
            poll_interval=kwargs.get("轮询间隔", 10),
            generate_audio=kwargs.get("生成音频", True),
            notify=self._notify(unique_id),
        )
        return result.video, result.task_id, result.response, result.video_url
