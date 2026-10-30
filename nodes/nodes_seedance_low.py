import asyncio
import io
import json
import mimetypes
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from comfy.comfy_types import IO

from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    get_config,
    mask_provider_payload,
    normalize_provider,
    provider_defaults,
)
from ..utils import (
    LingliuNullVideoAdapter,
    blank_image_tensor,
    pil2tensor,
    save_video_natively,
    tensor2pil,
    video_save_options,
)
from .nodes_openai_compatible import LingliuBaseNode
from .nodes_seedance25 import (
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_MEDIA,
    MAX_REFERENCE_VIDEOS,
    PreparedAsset,
    Seedance25Error,
    _probe_video_duration,
    _read_media_input,
    _redact,
    _redact_text,
    _waveform_to_wav_bytes,
)


LOW_CALL_SUFFIX = "-按次计费"
LOW_DEFAULT_RESOLUTION = "720p"
LOW_ALLOWED_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
LOW_FIXED_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
LOW_WIDE_TALL_RATIOS = ("16:9", "9:16")
LOW_ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
UPLOAD_CONCURRENCY = 4


@dataclass(frozen=True)
class LowModelContract:
    """Per-alias limits mirrored from the console, so bad inputs fail locally with a plain message."""

    resolution: str = LOW_DEFAULT_RESOLUTION
    min_duration: int = 4
    max_duration: int = 30
    fixed_duration: int = 0
    billing_mode: str = "video_seconds"
    ratios: tuple = LOW_ALLOWED_RATIOS
    max_images: int = 9
    max_videos: int = 3
    max_audios: int = 3
    max_total: int = MAX_REFERENCE_MEDIA
    max_video_seconds: float = 30
    requires_image: bool = False
    audio_requires_image: bool = False
    audio_requires_visual: bool = False
    generate_audio: bool = False
    unstable: bool = False


def _call_fixed(**kwargs):
    return LowModelContract(fixed_duration=15, billing_mode="call", **kwargs)


def _deal_20(resolution):
    return LowModelContract(
        resolution=resolution, max_duration=15, ratios=LOW_FIXED_RATIOS, max_total=15, max_video_seconds=15,
        audio_requires_visual=True, generate_audio=True, unstable=resolution == "1080p",
    )


def _deal_25(resolution):
    return LowModelContract(
        resolution=resolution, ratios=LOW_FIXED_RATIOS, max_images=30, max_videos=10, max_audios=10, max_total=40,
        generate_audio=True, unstable=resolution == "1080p",
    )


# Order is the 模型 dropdown order; the first six are the original public aliases.
LOW_MODEL_CONTRACTS = {
    "seedance2.0-低价": LowModelContract(ratios=LOW_FIXED_RATIOS),
    "seedance2.0-低价-按次计费": _call_fixed(ratios=LOW_FIXED_RATIOS),
    "seedance2.0-fast-低价": LowModelContract(),
    "seedance2.0-fast-低价-按次计费": _call_fixed(ratios=LOW_FIXED_RATIOS),
    "seedance2.0-mini-低价": LowModelContract(),
    "seedance2.5-低价": LowModelContract(),
    "seedance2.5-低价-按次计费": LowModelContract(min_duration=5, billing_mode="call", ratios=LOW_WIDE_TALL_RATIOS, max_images=30, max_videos=10, max_audios=10),
    "seedance2.0-特价-按次计费": _call_fixed(ratios=LOW_WIDE_TALL_RATIOS, max_videos=0),
    "seedance2.5-低价-480p": LowModelContract(resolution="480p", min_duration=5, ratios=LOW_WIDE_TALL_RATIOS, max_images=30, max_videos=0, max_audios=0),
    "seedance2.5-低价-720p": LowModelContract(min_duration=5, max_duration=29, ratios=LOW_FIXED_RATIOS, max_images=30, max_videos=0, max_audios=10, audio_requires_image=True),
    "seedance2.0-图生-低价": LowModelContract(max_duration=15, ratios=LOW_WIDE_TALL_RATIOS, max_videos=0, requires_image=True),
    "seedance2.0-低价-480p": LowModelContract(resolution="480p", min_duration=5, max_duration=15, ratios=LOW_WIDE_TALL_RATIOS),
    "seedance2.0-特惠-480p": _deal_20("480p"),
    "seedance2.0-特惠-720p": _deal_20("720p"),
    "seedance2.0-特惠-1080p": _deal_20("1080p"),
    "seedance2.0-mini-特惠-480p": _deal_20("480p"),
    "seedance2.0-mini-特惠-720p": _deal_20("720p"),
    "seedance2.0-fast-特惠-480p": _deal_20("480p"),
    "seedance2.0-fast-特惠-720p": _deal_20("720p"),
    "seedance2.5-特惠-480p": _deal_25("480p"),
    "seedance2.5-特惠-720p": _deal_25("720p"),
    "seedance2.5-特惠-1080p": _deal_25("1080p"),
}
LOW_MODELS = list(LOW_MODEL_CONTRACTS)


def low_contract(model):
    return LOW_MODEL_CONTRACTS.get(str(model or "").strip()) or LowModelContract()


@dataclass(frozen=True)
class LowSeedanceResult:
    video: object
    task_id: str
    response: str
    video_url: str
    last_frame: object


class LowSeedanceError(Seedance25Error):
    pass


class ProxyVideoAdapter:
    """VIDEO adapter for the authenticated platform content proxy."""

    def __init__(self, video_url, api_key):
        self.video_url = str(video_url or "")
        self.api_key = str(api_key or "").strip()
        self.video_path = None
        self.mime_type = "video/mp4"

    def get_dimensions(self):
        return 1280, 720

    def _response(self):
        if not self.video_url.startswith(("http://", "https://")):
            raise RuntimeError("视频地址无效。")
        return requests.get(
            self.video_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            stream=True,
            timeout=300,
            allow_redirects=False,
        )

    def read_bytes(self):
        response = self._response()
        try:
            response.raise_for_status()
            return b"".join(chunk for chunk in response.iter_content(chunk_size=1024 * 1024) if chunk)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def save_to(self, output, format="auto", codec="auto", metadata=None,
                bit_depth=None, crf=None, color_space=None, preset=None, **kwargs):
        del metadata, kwargs
        options = video_save_options(output, format, codec, bit_depth, crf, color_space, preset)
        if options is not None:
            return save_video_natively(self.save_to, output, options)
        response = self._response()
        try:
            response.raise_for_status()
            if hasattr(output, "write"):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                if hasattr(output, "seek"):
                    output.seek(0)
                return True
            with open(output, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            return True
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


class _RequestsHTTPAdapter:
    def request(self, method, url, **kwargs):
        return requests.request(method, url, **kwargs)


class LowSeedanceWorkflow:
    """Deep module for the low-price public aliases and the /v1/videos contract."""

    def __init__(self, http=None, sleep=None, allowed_models=None, model_resolutions=None, upload_concurrency=UPLOAD_CONCURRENCY):
        self.http = http or _RequestsHTTPAdapter()
        self.sleep = sleep or asyncio.sleep
        self.allowed_models = tuple(allowed_models or LOW_MODELS)
        self.model_resolutions = dict(model_resolutions or {})
        self.upload_concurrency = max(1, int(upload_concurrency))

    @staticmethod
    def effective_duration(model, duration):
        contract = low_contract(model)
        low, high = contract.min_duration, contract.max_duration
        if isinstance(duration, bool) or int(duration) != duration or not (low <= int(duration) <= high):
            raise LowSeedanceError(f"时长必须是 {low}–{high} 的整数。")
        requested = int(duration)
        return requested, contract.fixed_duration or requested, contract.billing_mode

    @staticmethod
    def _image_batch_size(image):
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) == 4:
            return int(shape[0])
        if isinstance(image, (list, tuple)):
            return len(image)
        return 1

    def prepare_image(self, slot, role, image):
        if self._image_batch_size(image) != 1:
            raise LowSeedanceError(f"{slot} 每个插槽只能连接 1 张图片。")
        try:
            pil_image = image if isinstance(image, Image.Image) else tensor2pil(image)[0]
        except Exception as exc:
            raise LowSeedanceError(f"{slot} 不是可识别的 IMAGE：{exc}") from exc
        buffer = BytesIO()
        pil_image.convert("RGB").save(buffer, format="PNG")
        return PreparedAsset(slot, "image", role, f"{slot}.png", "image/png", buffer.getvalue())

    def prepare_video(self, slot, role, video):
        data, filename = _read_media_input(video, ".mp4")
        filename = filename or f"{slot}.mp4"
        duration = _probe_video_duration(data, filename)
        return PreparedAsset(slot, "video", role, filename, mimetypes.guess_type(filename)[0] or "video/mp4", data, duration)

    def prepare_audio(self, slot, role, audio):
        if isinstance(audio, Mapping) and audio.get("waveform") is not None:
            data, filename, content_type = _waveform_to_wav_bytes(audio)
        else:
            data, filename = _read_media_input(audio, ".wav")
            filename = filename or f"{slot}.wav"
            content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
        return PreparedAsset(slot, "audio", role, filename, content_type, data)

    def validate_and_prepare(self, raw_assets, ratio, contract=None):
        contract = contract or LowModelContract()
        prepared = []
        for slot, media_type, role, value in raw_assets:
            if value is None:
                continue
            if media_type == "image":
                prepared.append(self.prepare_image(slot, role, value))
            elif media_type == "video":
                prepared.append(self.prepare_video(slot, role, value))
            elif media_type == "audio":
                prepared.append(self.prepare_audio(slot, role, value))
            else:
                raise LowSeedanceError(f"不支持的素材类型：{media_type}")

        first_frames = [item for item in prepared if item.role == "first_frame"]
        last_frames = [item for item in prepared if item.role == "last_frame"]
        references = [item for item in prepared if item.role.startswith("reference_")]
        image_count = sum(item.role == "reference_image" for item in prepared)
        video_count = sum(item.role == "reference_video" for item in prepared)
        audio_count = sum(item.role == "reference_audio" for item in prepared)
        if last_frames and not first_frames:
            raise LowSeedanceError("连接尾帧图时必须同时连接首帧图。")
        if (first_frames or last_frames) and references:
            raise LowSeedanceError("首尾帧模式不能与参考图、参考视频或参考音频混用。")
        if (first_frames or last_frames) and ratio != "adaptive":
            raise LowSeedanceError("首尾帧模式的比例必须为 adaptive。")
        if image_count > MAX_REFERENCE_IMAGES or video_count > MAX_REFERENCE_VIDEOS or audio_count > MAX_REFERENCE_AUDIOS:
            raise LowSeedanceError("参考素材数量超过节点上限。")
        if len(references) > MAX_REFERENCE_MEDIA:
            raise LowSeedanceError(f"参考媒体总数最多 {MAX_REFERENCE_MEDIA} 个。")
        # The console counts first/last frames as images.
        all_images = image_count + len(first_frames) + len(last_frames)
        for count, limit, unit, label in (
            (image_count, contract.max_images, "张", "参考图"),
            (video_count, contract.max_videos, "个", "参考视频"),
            (audio_count, contract.max_audios, "个", "参考音频"),
        ):
            if count and not limit:
                raise LowSeedanceError(f"该模型不支持{label}。")
            if count > limit:
                raise LowSeedanceError(f"该模型最多 {limit} {unit}{label}。")
        if len(references) > contract.max_total:
            raise LowSeedanceError(f"该模型参考素材总数最多 {contract.max_total} 个。")
        if contract.requires_image and not all_images:
            raise LowSeedanceError("该模型必须至少连接 1 张参考图。")
        if audio_count and contract.audio_requires_image and not all_images:
            raise LowSeedanceError("该模型使用参考音频时必须同时连接参考图。")
        if audio_count and contract.audio_requires_visual and not (all_images or video_count):
            raise LowSeedanceError("该模型使用参考音频时必须同时连接参考图或参考视频。")
        reference_video_seconds = sum(item.duration for item in prepared if item.role == "reference_video")
        if video_count and reference_video_seconds < 2 * video_count:
            raise LowSeedanceError(f"参考视频总时长至少为 {2 * video_count} 秒。")
        if reference_video_seconds > 30:
            raise LowSeedanceError("参考视频总时长不能超过 30 秒。")
        if reference_video_seconds > contract.max_video_seconds:
            raise LowSeedanceError(f"该模型参考视频总时长不能超过 {contract.max_video_seconds:g} 秒。")
        return prepared, reference_video_seconds

    def resolve_config(self, api_key_override=""):
        runtime = get_config()
        provider = normalize_provider(runtime.get("api_provider"))
        if provider not in LOW_ALLOWED_PROVIDERS:
            raise LowSeedanceError("低价视频节点只支持灵流线路。", stage="route")
        raw_key = str(api_key_override or runtime.get("api_key") or "").strip()
        if not raw_key:
            raise LowSeedanceError("API 密钥为空。", stage="auth")
        defaults = provider_defaults(provider)
        base_url = str(runtime.get("base_url") or defaults.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise LowSeedanceError("API 地址为空。", stage="route")
        return {"provider": provider, "base_url": base_url, "timeout": int(runtime.get("timeout") or 300), "api_key": raw_key}

    @staticmethod
    def build_content(prompt, prepared, uploaded_urls):
        content = [{"type": "text", "text": str(prompt or "").strip()}]
        for asset, url in zip(prepared, uploaded_urls):
            field = f"{asset.media_type}_url"
            content.append({"type": field, "role": asset.role, field: {"url": url}})
        return content

    def build_payload(self, model, prompt, content, duration, ratio, reference_video_seconds=0, generate_audio=True):
        model = str(model or "").strip()
        contract = low_contract(model)
        body = {
            "model": model,
            "prompt": str(prompt or "").strip(),
            "content": content,
            "duration": int(duration),
            "resolution": self.model_resolutions.get(model) or contract.resolution,
            "ratio": str(ratio or "").strip(),
        }
        # The console rejects reference videos without their total length.
        if reference_video_seconds > 0:
            body["reference_video_seconds"] = round(float(reference_video_seconds), 3)
        if contract.generate_audio:
            body["generate_audio"] = bool(generate_audio)
        return body

    @staticmethod
    def _extract_value(payload, keys):
        if not isinstance(payload, dict):
            return ""
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("data", "result", "output", "metadata"):
            nested = payload.get(key)
            value = LowSeedanceWorkflow._extract_value(nested, keys)
            if value:
                return value
        return ""

    @classmethod
    def _extract_task_id(cls, payload):
        return cls._extract_value(payload, ("task_id", "taskId", "id", "video_id"))

    @classmethod
    def _extract_status(cls, payload):
        return cls._extract_value(payload, ("status", "state", "task_status")).lower()

    @staticmethod
    def _json(payload, secrets):
        return json.dumps(mask_provider_payload(_redact(payload, secrets)), ensure_ascii=False, indent=2)

    async def _request_json(self, method, url, config, *, stage, retry=False, **kwargs):
        attempts = 3 if retry else 1
        last_error = None
        for attempt in range(attempts):
            try:
                response = await asyncio.to_thread(
                    self.http.request,
                    method,
                    url,
                    headers={"Accept": "application/json", "Authorization": f"Bearer {config['api_key']}", **kwargs.pop("extra_headers", {})},
                    timeout=config["timeout"],
                    **kwargs,
                )
                payload = response.json()
                status_code = int(getattr(response, "status_code", 200) or 200)
                if status_code < 400:
                    return payload
                last_error = LowSeedanceError(f"{stage} HTTP {status_code}", stage=stage, status_code=status_code, payload=payload)
            except LowSeedanceError:
                raise
            except Exception as exc:
                last_error = LowSeedanceError(f"{stage} 网络错误：{exc}", stage=stage)
            if attempt + 1 < attempts:
                await self.sleep(min(2 ** attempt, 4))
        raise last_error or LowSeedanceError(f"{stage} 请求失败。", stage=stage)

    async def _upload_one(self, asset, config):
        payload = await self._request_json(
            "POST",
            build_url(config["base_url"], "/v1/files"),
            config,
            stage="upload",
            retry=True,
            files={"file": (asset.filename, asset.data, asset.content_type)},
        )
        url = self._extract_value(payload, ("url", "file_url", "fileUrl"))
        if not url.startswith(("http://", "https://")):
            raise LowSeedanceError(f"{asset.slot} 上传响应缺少有效 URL。", stage="upload", payload=payload)
        return url, {"slot": asset.slot, "type": asset.media_type, "role": asset.role, "bytes": len(asset.data), "url": url}

    async def upload_all(self, prepared, config):
        urls, summaries = [], []
        for start in range(0, len(prepared), self.upload_concurrency):
            batch = prepared[start : start + self.upload_concurrency]
            results = await asyncio.gather(*(self._upload_one(asset, config) for asset in batch), return_exceptions=True)
            for asset, result in zip(batch, results):
                if isinstance(result, Exception):
                    raise LowSeedanceError(str(result), stage=getattr(result, "stage", "upload"), payload=getattr(result, "payload", None)) from result
                url, summary = result
                urls.append(url)
                summaries.append(summary)
        return urls, summaries

    @staticmethod
    def _content_url(config, task_id):
        return build_url(config["base_url"], f"/v1/videos/{task_id}/content")

    @staticmethod
    def _extract_last_frame(video_bytes):
        temp_path = ""
        capture = None
        try:
            import cv2

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
                temp_path = handle.name
                handle.write(video_bytes)
            capture = cv2.VideoCapture(temp_path)
            if not capture.isOpened():
                return blank_image_tensor(width=1, height=1, color="black")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(count - 1, 0))
            ok, frame = capture.read()
            if not ok or frame is None:
                return blank_image_tensor(width=1, height=1, color="black")
            return pil2tensor(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except Exception:
            return blank_image_tensor(width=1, height=1, color="black")
        finally:
            if capture is not None:
                capture.release()
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    async def execute(
        self,
        *,
        prompt,
        model,
        duration,
        ratio,
        raw_assets,
        api_key="",
        wait_for_completion=True,
        max_poll_attempts=180,
        poll_interval=10,
        return_last_frame=False,
        generate_audio=True,
        notify=None,
    ):
        blank_last_frame = blank_image_tensor(width=1, height=1, color="black")
        task_id = ""
        secrets = [api_key]
        envelope = {"node": "lingliu_seedance_low", "model": model}
        try:
            model = str(model or "").strip()
            if model not in self.allowed_models:
                raise LowSeedanceError("模型不在当前低价视频模型列表中。")
            contract = low_contract(model)
            prompt = str(prompt or "").strip()
            if not prompt:
                raise LowSeedanceError("提示词不能为空。")
            requested_duration, effective_duration, billing_mode = self.effective_duration(model, duration)
            if ratio not in LOW_ALLOWED_RATIOS:
                raise LowSeedanceError("比例不受支持。")
            notices = []
            submit_ratio = ratio
            # Frames need adaptive on the console, so fixed-ratio models cannot take them.
            has_frames = any(role in {"first_frame", "last_frame"} and value is not None for _, _, role, value in raw_assets)
            if has_frames and "adaptive" not in contract.ratios:
                raise LowSeedanceError("该模型不支持首尾帧，请把图片接到参考图接口。")
            if ratio not in contract.ratios:
                if ratio != "adaptive":
                    raise LowSeedanceError(f"该模型比例只支持 {'、'.join(contract.ratios)}。")
                # Saved workflows default to adaptive; fixed-ratio models submit 16:9 instead of failing.
                submit_ratio = "16:9"
                notices.append("该模型不支持 adaptive 比例，已按 16:9 提交。")
            if contract.unstable:
                notices.append("1080p 特惠模型暂不稳定，失败时可改用 720p。")
            if not (1 <= int(max_poll_attempts) <= 720) or not (2 <= int(poll_interval) <= 60):
                raise LowSeedanceError("轮询参数超出范围。")
            config = self.resolve_config(api_key)
            secrets.append(config["api_key"])
            prepared, reference_video_seconds = await asyncio.to_thread(self.validate_and_prepare, raw_assets, ratio, contract)
            uploaded_urls, upload_summaries = await self.upload_all(prepared, config)
            content = self.build_content(prompt, prepared, uploaded_urls)
            body = self.build_payload(model, prompt, content, effective_duration, submit_ratio, reference_video_seconds, generate_audio)
            if notices:
                envelope["notice"] = "；".join(notices)
            envelope["provider"] = config["provider"]
            envelope["request"] = {
                **body,
                "requested_duration": requested_duration,
                "effective_duration": effective_duration,
                "billing_mode": billing_mode,
                "reference_video_seconds": reference_video_seconds,
            }
            envelope["uploads"] = upload_summaries
            create_payload = await self._request_json(
                "POST",
                build_url(config["base_url"], "/v1/videos"),
                config,
                stage="create",
                extra_headers={"Content-Type": "application/json"},
                json=body,
            )
            envelope["create_response"] = create_payload
            task_id = self._extract_task_id(create_payload)
            if not task_id:
                raise LowSeedanceError("创建响应缺少任务 ID。", stage="create", payload=create_payload)
            envelope["task_id"] = task_id
            if notify:
                notify(task_id, "submitted")
            if not wait_for_completion:
                envelope["status"] = "submitted"
                return LowSeedanceResult(LingliuNullVideoAdapter("任务已提交，未等待完成。"), task_id, self._json(envelope, secrets), "", blank_last_frame)

            final_payload = None
            for attempt in range(int(max_poll_attempts)):
                query_payload = await self._request_json(
                    "GET",
                    build_url(config["base_url"], f"/v1/videos/{task_id}"),
                    config,
                    stage="query",
                    retry=True,
                )
                final_payload = query_payload
                envelope["query_response"] = query_payload
                status = self._extract_status(query_payload)
                if notify:
                    notify(task_id, status or "unknown")
                if status in {"", "unknown", "submitted", "queued", "pending", "running", "processing", "in_progress"}:
                    if attempt + 1 < int(max_poll_attempts):
                        await self.sleep(int(poll_interval))
                    continue
                if status in {"failed", "failure", "error", "rejected", "cancelled", "canceled", "expired"}:
                    reason = self._extract_value(query_payload, ("message", "error", "reason", "fail_reason"))
                    raise LowSeedanceError(reason or f"任务终态为 {status}。", stage="query", payload=query_payload)
                if status in {"completed", "complete", "succeeded", "success"}:
                    video_url = self._content_url(config, task_id)
                    envelope["status"] = "succeeded"
                    last_frame = blank_last_frame
                    if return_last_frame:
                        adapter = ProxyVideoAdapter(video_url, config["api_key"])
                        video_bytes = await asyncio.to_thread(adapter.read_bytes)
                        last_frame = await asyncio.to_thread(self._extract_last_frame, video_bytes)
                        envelope["last_frame"] = {"source": "video_last_frame_extracted"}
                    return LowSeedanceResult(ProxyVideoAdapter(video_url, config["api_key"]), task_id, self._json(envelope, secrets), video_url, last_frame)
                raise LowSeedanceError(f"查询返回未知状态：{status or '<empty>'}。", stage="query", payload=query_payload)

            envelope["status"] = "timeout"
            envelope["query_response"] = final_payload
            return LowSeedanceResult(LingliuNullVideoAdapter("轮询超时。"), task_id, self._json(envelope, secrets), "", blank_last_frame)
        except Exception as exc:
            error_payload = {
                **envelope,
                "status": "error",
                "task_id": task_id,
                "error": {
                    "stage": getattr(exc, "stage", "internal"),
                    "message": _redact_text(str(exc), secrets),
                    "status_code": getattr(exc, "status_code", None),
                    "upstream": _redact(getattr(exc, "payload", None), secrets),
                },
            }
            return LowSeedanceResult(LingliuNullVideoAdapter(_redact_text(str(exc), secrets)), task_id, self._json(error_payload, secrets), "", blank_last_frame)


class LingliuSeedanceLow(LingliuBaseNode):
    CATEGORY = "Lingliu/Seedance 低价"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL", "尾帧图")
    FUNCTION = "generate_video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "API密钥": ("STRING", {"default": "", "password": True}),
            "返回尾帧图": ("BOOLEAN", {"default": False}),
            "等待完成": ("BOOLEAN", {"default": True}),
            "最大轮询次数": ("INT", {"default": 180, "min": 1, "max": 720, "step": 1}),
            "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
            "首帧图": ("IMAGE",),
            "尾帧图": ("IMAGE",),
        }
        optional.update({f"参考图{i}": ("IMAGE",) for i in range(1, MAX_REFERENCE_IMAGES + 1)})
        optional.update({f"参考视频{i}": (IO.VIDEO, {"tooltip": "单个插槽连接一个参考视频。"}) for i in range(1, MAX_REFERENCE_VIDEOS + 1)})
        optional.update({f"参考音频{i}": (IO.AUDIO, {"tooltip": "单个插槽连接一个参考音频。"}) for i in range(1, MAX_REFERENCE_AUDIOS + 1)})
        # Appended last so saved workflows keep their widget order.
        optional["生成音频"] = ("BOOLEAN", {"default": True, "tooltip": "仅特惠系列模型生效；关闭后生成无声视频。"})
        return {
            "required": {
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "模型": (list(LOW_MODELS), {
                    "default": LOW_MODELS[0],
                    "tooltip": "各模型的时长、比例和参考素材上限不同，超出会在提交前提示；标注按次计费的 2.0 模型固定 15 秒；1080p 特惠模型暂不稳定。",
                }),
                "时长": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1}),
                "比例": (list(LOW_ALLOWED_RATIOS), {"default": "16:9", "tooltip": "首尾帧只在支持 adaptive 的模型上可用，且比例需选 adaptive。"}),
                "参考图接口数量": ("INT", {"default": 3, "min": 0, "max": MAX_REFERENCE_IMAGES, "step": 1}),
                "参考视频接口数量": ("INT", {"default": 1, "min": 0, "max": MAX_REFERENCE_VIDEOS, "step": 1}),
                "参考音频接口数量": ("INT", {"default": 1, "min": 0, "max": MAX_REFERENCE_AUDIOS, "step": 1}),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def _notify(self, unique_id):
        def notify(task_id, status):
            if task_id:
                self.publish_task_id(unique_id, task_id, append=False)
            self.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status}", append=False)

        return notify

    async def generate_video(self, **kwargs):
        raw_assets = [("首帧图", "image", "first_frame", kwargs.get("首帧图")), ("尾帧图", "image", "last_frame", kwargs.get("尾帧图"))]
        raw_assets.extend((f"参考图{i}", "image", "reference_image", kwargs.get(f"参考图{i}")) for i in range(1, MAX_REFERENCE_IMAGES + 1))
        raw_assets.extend((f"参考视频{i}", "video", "reference_video", kwargs.get(f"参考视频{i}")) for i in range(1, MAX_REFERENCE_VIDEOS + 1))
        raw_assets.extend((f"参考音频{i}", "audio", "reference_audio", kwargs.get(f"参考音频{i}")) for i in range(1, MAX_REFERENCE_AUDIOS + 1))
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)
        result = await LowSeedanceWorkflow().execute(
            prompt=kwargs.get("提示词", ""),
            model=kwargs.get("模型", LOW_MODELS[0]),
            duration=kwargs.get("时长", 5),
            ratio=kwargs.get("比例", "16:9"),
            raw_assets=raw_assets,
            api_key=kwargs.get("API密钥", ""),
            wait_for_completion=kwargs.get("等待完成", True),
            max_poll_attempts=kwargs.get("最大轮询次数", 180),
            poll_interval=kwargs.get("轮询间隔", 10),
            return_last_frame=kwargs.get("返回尾帧图", False),
            generate_audio=kwargs.get("生成音频", True),
            notify=self._notify(unique_id),
        )
        return result.video, result.task_id, result.response, result.video_url, result.last_frame
