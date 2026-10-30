"""Wan 3.0 video node.

This module deliberately owns its Wan-specific contract.  It does not alter the
Seedance routes or the shared upload helpers: media is uploaded one item at a
time and the create request contains URLs only.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import tempfile
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from comfy.comfy_types import IO

from ..lingliu_config import build_url, mask_provider_payload
from ..utils import (
    LingliuNullVideoAdapter,
    blank_image_tensor,
    image_to_png_bytes,
    pil2tensor,
    save_video_natively,
    tensor2pil,
    video_save_options,
)
from .nodes_openai_compatible import LingliuBaseNode
from .nodes_seedance25 import (
    _extract_status,
    _extract_string,
    _extract_video_url,
    _probe_video_duration,
    _read_media_input,
    _waveform_to_wav_bytes,
)


WAN3_MODELS = ("wan3.0-video", "wan3.0-video-prime")
WAN3_MODES = ("文生视频", "首帧图生视频", "首尾帧生视频", "参考生视频")
WAN3_RATIOS = ("adaptive", "16:9", "4:3", "1:1", "3:4", "9:16")
WAN3_RESOLUTIONS = ("480p", "720p", "1080p")
WAN3_CHANNELS = ("LL", "XJ")
WAN3_XJAI_MODELS = {"wan3.0-video": "xjai-wan3.0-video", "wan3.0-video-prime": "xjai-wan3.0-video-prime"}
WAN3_XJAI_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16")
ALLOWED_PROVIDERS = {"lingliu"}
NATIVE_WAN3_PROVIDERS = {"lingliu"}
MAX_REFERENCE_IMAGES = 10
MAX_REFERENCE_VIDEOS = 5
MAX_REFERENCE_AUDIOS = 5
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_AUDIO_BYTES = 15 * 1024 * 1024
MAX_MEDIA_SECONDS = 15.0


class Wan3Error(RuntimeError):
    """An error with a user-facing stage and a safe diagnostic payload."""

    def __init__(self, message, *, stage="validation", status_code=None, payload=None, details=None):
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code
        self.payload = payload
        self.details = details or {}


@dataclass(frozen=True)
class Wan3Asset:
    slot: str
    media_type: str
    role: str
    filename: str
    content_type: str
    data: bytes
    duration: float = 0.0


@dataclass(frozen=True)
class Wan3Result:
    video: object
    task_id: str
    response: str
    video_url: str
    last_frame: object


class ProxyVideoAdapter:
    """Authenticated video adapter for the platform content endpoint."""

    def __init__(self, video_url, api_key="", mime_type="video/mp4"):
        self.video_url = str(video_url or "")
        self.video_path = None
        self.api_key = str(api_key or "").strip()
        self.mime_type = mime_type

    def get_dimensions(self):
        return 1280, 720

    def _headers(self):
        headers = {"Accept": "video/mp4,application/octet-stream,*/*"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _write_chunk(output, chunk):
        if chunk:
            output.write(chunk)

    def save_to(self, output, format="auto", codec="auto", metadata=None,
                bit_depth=None, crf=None, color_space=None, preset=None, **kwargs):
        del metadata, kwargs
        options = video_save_options(output, format, codec, bit_depth, crf, color_space, preset)
        if options is not None:
            return save_video_natively(self.save_to, output, options)
        if not self.video_url:
            raise RuntimeError("视频 URL 为空，无法保存视频。")
        response = requests.get(
            self.video_url,
            headers=self._headers(),
            stream=True,
            timeout=300,
        )
        try:
            response.raise_for_status()
            if hasattr(output, "write"):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    self._write_chunk(output, chunk)
                if hasattr(output, "seek"):
                    output.seek(0)
                return True
            with open(output, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    self._write_chunk(handle, chunk)
            return True
        finally:
            response.close()


def _redact_text(value, secrets):
    text = str(value or "")
    for secret in sorted({str(item) for item in secrets if item}, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    # Keep diagnostics useful without exposing bearer keys or common API keys.
    import re

    text = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer <redacted>", text)
    text = re.sub(r"(?i)sk-[A-Za-z0-9_.:-]+(?:-\d+)?", "sk-<redacted>", text)
    # Signed object-store links can contain credentials in the query string.
    return re.sub(r"(https?://[^?\s\"']+)\?[^\s\"']+", r"\1?<redacted-query>", text)


def _redact(value, secrets):
    if isinstance(value, Mapping):
        return {
            str(key): _redact(item, secrets)
            for key, item in value.items()
            if str(key).lower() not in {"authorization", "api_key", "apikey"}
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact_text(value, secrets)
    return value


def _image_batch_size(image):
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) == 4:
        return int(shape[0])
    if isinstance(image, (list, tuple)):
        return len(image)
    return 1


def _extract_wan_task_id(payload):
    return _extract_string(payload, ("task_id", "taskId", "request_id", "requestId", "id"))


def _probe_audio_duration(data, filename):
    """Probe common audio containers without making duration optional silently."""
    if str(filename).lower().endswith(".wav"):
        try:
            with wave.open(BytesIO(data), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                if rate > 0 and frames > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass
    try:
        import av

        container = av.open(BytesIO(data))
        try:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise ValueError("不包含音频轨")
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                duration = 0.0
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("无法得到有效时长")
            return duration
        finally:
            container.close()
    except Exception as exc:
        raise Wan3Error(f"{filename} 无法解析音频时长：{exc}") from exc


def _extract_last_frame_authenticated(adapter):
    """Download through ProxyVideoAdapter, then decode the final frame locally."""
    temp_path = ""
    capture = None
    try:
        import cv2

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
            temp_path = handle.name
        adapter.save_to(temp_path)
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            return None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        return pil2tensor(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    except Exception:
        return None
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


class Wan3Workflow:
    """The isolated validation/upload/create/query workflow for Wan 3.0."""

    def __init__(self, node):
        self.node = node

    @staticmethod
    def safe_json(payload, secrets):
        safe = _redact(mask_provider_payload(payload), secrets)
        return json.dumps(safe, ensure_ascii=False, indent=2)

    def prepare_image(self, slot, role, image):
        if _image_batch_size(image) != 1:
            raise Wan3Error(f"{slot} 每个插槽只能连接 1 张图片。")
        try:
            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, (list, tuple)) and image and isinstance(image[0], Image.Image):
                pil_image = image[0]
            else:
                pil_image = tensor2pil(image)[0]
        except Exception as exc:
            raise Wan3Error(f"{slot} 不是可识别的 IMAGE：{exc}") from exc
        width, height = pil_image.size
        if min(width, height) < 1 or max(width, height) > 8000:
            raise Wan3Error(f"{slot} 图片边长必须不超过 8000 像素。")
        if min(width, height) <= 0 or max(width, height) / min(width, height) > 8:
            raise Wan3Error(f"{slot} 图片长宽比不能超过 8:1。")
        data = image_to_png_bytes(pil_image)
        if len(data) > MAX_IMAGE_BYTES:
            raise Wan3Error(f"{slot} 图片大小不能超过 20 MB。")
        return Wan3Asset(slot, "image", role, f"{slot}.png", "image/png", data)

    def prepare_video(self, slot, role, video):
        try:
            data, filename = _read_media_input(video, ".mp4")
            duration = _probe_video_duration(data, filename)
        except Wan3Error:
            raise
        except Exception as exc:
            raise Wan3Error(f"{slot} 无法读取视频：{exc}") from exc
        if len(data) > MAX_VIDEO_BYTES:
            raise Wan3Error(f"{slot} 视频大小不能超过 100 MB。")
        if not 1 <= duration <= MAX_MEDIA_SECONDS:
            raise Wan3Error(f"{slot} 视频时长必须在 1–15 秒之间。")
        filename = filename or f"{slot}.mp4"
        content_type = mimetypes.guess_type(filename)[0] or "video/mp4"
        return Wan3Asset(slot, "video", role, filename, content_type, data, duration)

    def prepare_audio(self, slot, role, audio):
        try:
            if isinstance(audio, Mapping) and audio.get("waveform") is not None:
                data, filename, content_type = _waveform_to_wav_bytes(audio)
            else:
                data, filename = _read_media_input(audio, ".wav")
                filename = filename or f"{slot}.wav"
                content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
            duration = _probe_audio_duration(data, filename)
        except Wan3Error:
            raise
        except Exception as exc:
            raise Wan3Error(f"{slot} 无法读取音频：{exc}") from exc
        if len(data) > MAX_AUDIO_BYTES:
            raise Wan3Error(f"{slot} 音频大小不能超过 15 MB。")
        if not 1 <= duration <= MAX_MEDIA_SECONDS:
            raise Wan3Error(f"{slot} 音频时长必须在 1–15 秒之间。")
        return Wan3Asset(slot, "audio", role, filename, content_type, data, duration)

    def validate_and_prepare(self, raw_assets, mode, duration, ratio, channel="LL"):
        channel = str(channel or "LL").strip().upper()
        if channel not in WAN3_CHANNELS:
            raise Wan3Error("渠道只能选择 LL 或 XJ。")
        if mode not in WAN3_MODES:
            raise Wan3Error("生成模式不受支持。")
        minimum_duration = 1 if channel == "XJ" else 2
        if not minimum_duration <= int(duration) <= 30:
            raise Wan3Error(f"当前渠道时长必须在 {minimum_duration}–30 秒之间。")
        allowed_ratios = WAN3_XJAI_RATIOS if channel == "XJ" else WAN3_RATIOS
        if ratio not in allowed_ratios:
            raise Wan3Error("比例不受 Wan 3.0 支持。")
        if channel == "XJ" and mode in {"首帧图生视频", "首尾帧生视频"}:
            raise Wan3Error("XJ 渠道当前只支持文生视频和参考生视频。")

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
                raise Wan3Error(f"不支持的素材类型：{media_type}")

        first = [item for item in prepared if item.role == "first_frame"]
        last = [item for item in prepared if item.role == "last_frame"]
        refs = [item for item in prepared if item.role.startswith("reference_")]
        ref_images = [item for item in refs if item.role == "reference_image"]
        ref_videos = [item for item in refs if item.role == "reference_video"]
        ref_audios = [item for item in refs if item.role == "reference_audio"]
        if len(first) > 1 or len(last) > 1:
            raise Wan3Error("首帧图和尾帧图各只能有一个。")

        if mode == "文生视频":
            if first or last or refs:
                raise Wan3Error("文生视频模式不能连接任何参考素材。")
        elif mode == "首帧图生视频":
            if len(first) != 1 or last or refs:
                raise Wan3Error("首帧图生视频模式必须只连接一张首帧图。")
        elif mode == "首尾帧生视频":
            if len(first) != 1 or len(last) != 1 or refs:
                raise Wan3Error("首尾帧生视频模式必须同时连接首帧和尾帧，且不能混用参考素材。")
        elif mode == "参考生视频":
            if first or last:
                raise Wan3Error("参考生视频模式不能连接首帧图或尾帧图。")
            if not refs:
                raise Wan3Error("参考生视频模式至少需要一项参考素材。")

        if len(ref_images) > MAX_REFERENCE_IMAGES:
            raise Wan3Error("参考图最多 10 张。")
        if len(ref_videos) > MAX_REFERENCE_VIDEOS:
            raise Wan3Error("参考视频最多 5 个。")
        if len(ref_audios) > MAX_REFERENCE_AUDIOS:
            raise Wan3Error("参考音频最多 5 个。")
        video_seconds = sum(item.duration for item in ref_videos)
        audio_seconds = sum(item.duration for item in ref_audios)
        if video_seconds > MAX_MEDIA_SECONDS:
            raise Wan3Error("参考视频总时长不能超过 15 秒。")
        if audio_seconds > MAX_MEDIA_SECONDS:
            raise Wan3Error("参考音频总时长不能超过 15 秒。")
        if ref_videos and video_seconds + int(duration) > 30:
            raise Wan3Error("参考视频总时长与输出时长合计不能超过 30 秒。")
        return prepared

    @staticmethod
    def build_content(prompt, prepared, uploaded_urls):
        content = [{"type": "text", "text": str(prompt).strip()}]
        for asset, url in zip(prepared, uploaded_urls):
            field = f"{asset.media_type}_url"
            content.append({"type": field, "role": asset.role, field: {"url": url}})
        return content

    @staticmethod
    def build_native_wan3_body(
        *,
        model,
        prompt,
        prepared,
        uploaded_urls,
        duration,
        ratio,
        resolution,
        generate_audio,
        prompt_extend,
        watermark,
        seed,
    ):
        role_types = {
            "first_frame": "first_frame",
            "last_frame": "last_frame",
            "reference_image": "reference_image",
            "reference_video": "reference_video",
            "reference_audio": "reference_audio",
        }
        media = []
        for asset, url in zip(prepared, uploaded_urls):
            media_type = role_types.get(asset.role)
            if not media_type:
                raise Wan3Error(f"Wan3 不支持的媒体角色：{asset.role}。")
            media.append({"type": media_type, "url": str(url)})

        body = {
            "model": str(model).strip(),
            "prompt": str(prompt).strip(),
            "parameters": {
                "resolution": str(resolution).strip().upper(),
                "ratio": str(ratio).strip(),
                "duration": int(duration),
                "prompt_extend": bool(prompt_extend),
            },
            "audio": bool(generate_audio),
            "watermark": bool(watermark),
        }
        if media:
            body["media"] = media
        if int(seed) > 0:
            body["seed"] = int(seed)
        return body

    @staticmethod
    def build_xjai_body(*, model, prompt, prepared, uploaded_urls, duration, ratio, resolution):
        media = [{"type": asset.content_type, "url": str(url)} for asset, url in zip(prepared, uploaded_urls)]
        body = {
            "model": str(model).strip(),
            "prompt": str(prompt).strip(),
            "duration": int(duration),
            "resolution": str(resolution).strip().upper(),
            "ratio": str(ratio).strip(),
        }
        if media:
            body["media"] = media
        return body

    @staticmethod
    def extract_file_url(payload):
        url = _extract_string(payload, ("url", "file_url", "fileUrl", "download_url", "downloadUrl"))
        if url.startswith(("http://", "https://")):
            return url
        return ""

    async def upload_one(self, asset, config):
        payload = await self.node.async_request_json_retrying(
            "POST",
            build_url(config["base_url"], "/v1/files"),
            config,
            max_retries=3,
            headers=self.node.headers(config),
            files={"file": (asset.filename, asset.data, asset.content_type)},
        )
        url = self.extract_file_url(payload)
        if not url:
            raise Wan3Error(f"{asset.slot} 上传响应缺少有效 URL。", stage="upload", payload=payload)
        return url, {
            "slot": asset.slot,
            "type": asset.media_type,
            "role": asset.role,
            "bytes": len(asset.data),
            "url": url,
        }

    async def upload_all(self, prepared, config):
        urls = []
        summaries = []
        # Deliberately serial: this is the OOM protection boundary for media.
        for asset in prepared:
            try:
                url, summary = await self.upload_one(asset, config)
            except Exception as exc:
                if isinstance(exc, Wan3Error):
                    raise Wan3Error(
                        str(exc),
                        stage=getattr(exc, "stage", "upload"),
                        status_code=getattr(exc, "status_code", None),
                        payload=getattr(exc, "payload", None),
                        details={"failed_slot": asset.slot, "completed_uploads": summaries},
                    ) from exc
                raise Wan3Error(
                    str(exc),
                    stage="upload",
                    details={"failed_slot": asset.slot, "completed_uploads": summaries},
                ) from exc
            urls.append(url)
            summaries.append(summary)
        return urls, summaries

    async def execute(
        self,
        *,
        prompt,
        model,
        mode,
        duration,
        ratio,
        resolution,
        generate_audio,
        prompt_extend,
        watermark,
        seed,
        raw_assets,
        channel="LL",
        api_key="",
        wait_for_completion=True,
        max_poll_attempts=180,
        poll_interval=10,
        return_last_frame=False,
        unique_id=None,
    ):
        task_id = ""
        blank_frame = blank_image_tensor(width=1, height=1, color="black")
        channel = str(channel or "LL").strip().upper()
        envelope = {"node": "lingliu_wan3", "model": model, "channel": channel}
        secrets = [api_key]
        try:
            model = str(model or "").strip()
            if model not in WAN3_MODELS:
                raise Wan3Error("模型不受 Wan 3.0 节点支持。")
            if channel not in WAN3_CHANNELS:
                raise Wan3Error("渠道只能选择 LL 或 XJ。")
            wire_model = WAN3_XJAI_MODELS[model] if channel == "XJ" else model
            if channel == "XJ" and ratio == "adaptive":
                ratio = "16:9"
            prompt = str(prompt or "").strip()
            if not prompt:
                raise Wan3Error("提示词不能为空。")
            if resolution not in WAN3_RESOLUTIONS:
                raise Wan3Error("清晰度不受 Wan 3.0 支持。")
            if not 1 <= int(max_poll_attempts) <= 720:
                raise Wan3Error("最大轮询次数必须为 1–720。")
            if not 0 <= int(poll_interval) <= 60:
                raise Wan3Error("轮询间隔必须为 0–60 秒。")
            config = self.node.load_config(api_key)
            config = dict(config or {})
            config["base_url"] = str(config.get("base_url") or "").strip().rstrip("/")
            config["api_key"] = str(config.get("api_key") or api_key or "").strip()
            provider = str(config.get("api_provider") or "").strip().lower()
            if provider not in ALLOWED_PROVIDERS:
                raise Wan3Error("Wan 3.0 节点只支持灵流线路。", stage="route")
            if not config["base_url"]:
                raise Wan3Error("API 地址为空。", stage="route")
            if not config["api_key"]:
                raise Wan3Error("API 密钥为空，请填写节点密钥或配置全局密钥。", stage="auth")
            if channel == "XJ" and provider != "lingliu":
                raise Wan3Error("XJ 渠道只能通过灵流线路访问。", stage="route")
            secrets.append(config["api_key"])
            prepared = await asyncio.to_thread(
                self.validate_and_prepare,
                raw_assets,
                mode,
                int(duration),
                ratio,
                channel,
            )
            uploaded_urls, upload_summaries = await self.upload_all(prepared, config)
            if channel == "XJ":
                body = self.build_xjai_body(
                    model=wire_model,
                    prompt=prompt,
                    prepared=prepared,
                    uploaded_urls=uploaded_urls,
                    duration=duration,
                    ratio=ratio,
                    resolution=resolution,
                )
            elif provider in NATIVE_WAN3_PROVIDERS:
                body = self.build_native_wan3_body(
                    model=model,
                    prompt=prompt,
                    prepared=prepared,
                    uploaded_urls=uploaded_urls,
                    duration=duration,
                    ratio=ratio,
                    resolution=resolution,
                    generate_audio=generate_audio,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    seed=seed,
                )
            else:
                content = self.build_content(prompt, prepared, uploaded_urls)
                body = {
                    "model": model,
                    "prompt": prompt,
                    "content": content,
                    "duration": int(duration),
                    "resolution": resolution,
                    "ratio": ratio,
                    "generateAudio": bool(generate_audio),
                    "promptExtend": bool(prompt_extend),
                    "watermark": bool(watermark),
                    "seed": int(seed),
                }
            envelope.update({"provider": config.get("api_provider", ""), "request": body, "uploads": upload_summaries})
            create_payload = await self.node.async_request_json(
                "POST",
                build_url(config["base_url"], "/v1/videos"),
                config,
                headers=self.node.headers(config, {"Content-Type": "application/json"}),
                allow_redirects=False,
                json=body,
            )
            envelope["create_response"] = create_payload
            task_id = _extract_wan_task_id(create_payload)
            if not task_id:
                raise Wan3Error("创建响应缺少任务 ID。", stage="create", payload=create_payload)
            envelope["task_id"] = task_id
            self.node.publish_task_id(unique_id, task_id, clear=True)
            if not wait_for_completion:
                envelope["status"] = "submitted"
                return Wan3Result(
                    LingliuNullVideoAdapter("任务已提交，未等待完成。"),
                    task_id,
                    self.safe_json(envelope, secrets),
                    "",
                    blank_frame,
                )

            final_payload = None
            for attempt in range(int(max_poll_attempts)):
                query_payload = await self.node.async_request_json_retrying(
                    "GET",
                    build_url(config["base_url"], f"/v1/videos/{task_id}"),
                    config,
                    max_retries=3,
                    headers=self.node.headers(config),
                )
                final_payload = query_payload
                envelope["query_response"] = query_payload
                status = _extract_status(query_payload)
                self.node.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status or 'unknown'}")
                if status in {"", "unknown", "pending", "queued", "submitted", "processing", "running", "in_progress"}:
                    if attempt + 1 < int(max_poll_attempts) and int(poll_interval) > 0:
                        await asyncio.sleep(int(poll_interval))
                    continue
                if status in {"failed", "error", "rejected", "cancelled", "expired"}:
                    reason = _extract_string(query_payload, ("message", "error", "fail_reason", "failReason"))
                    raise Wan3Error(reason or f"任务终态为 {status}。", stage="query", payload=query_payload)
                if status in {"completed", "complete", "succeeded", "success"}:
                    video_url = _extract_video_url(query_payload)
                    if not video_url:
                        video_url = build_url(config["base_url"], f"/v1/videos/{task_id}/content")
                    envelope["status"] = "succeeded"
                    last_frame = blank_frame
                    if return_last_frame:
                        # The content endpoint is authenticated; the fallback is
                        # intentionally best-effort and never affects video output.
                        adapter = ProxyVideoAdapter(video_url, config["api_key"])
                        try:
                            last_frame = await asyncio.to_thread(_extract_last_frame_authenticated, adapter)
                        except Exception:
                            last_frame = None
                        if last_frame is None:
                            last_frame = blank_frame
                    return Wan3Result(
                        ProxyVideoAdapter(video_url, config["api_key"]),
                        task_id,
                        self.safe_json(envelope, secrets),
                        video_url,
                        last_frame,
                    )
                raise Wan3Error(
                    f"查询返回未知状态：{status or '<empty>'}。",
                    stage="query",
                    payload=query_payload,
                )

            envelope["status"] = "timeout"
            envelope["query_response"] = final_payload
            envelope["error"] = {"stage": "query", "message": "轮询超时。"}
            return Wan3Result(
                LingliuNullVideoAdapter("轮询超时。"),
                task_id,
                self.safe_json(envelope, secrets),
                "",
                blank_frame,
            )
        except Exception as exc:
            details = dict(getattr(exc, "details", {}) or {})
            error_payload = {
                **envelope,
                "status": "error",
                "task_id": task_id,
                "error": {
                    "stage": getattr(exc, "stage", "internal"),
                    "message": _redact_text(str(exc), secrets),
                    "status_code": getattr(exc, "status_code", None),
                    "upstream": _redact(getattr(exc, "payload", None), secrets),
                    "details": _redact(details or None, secrets),
                },
            }
            if details.get("completed_uploads") is not None:
                error_payload["uploads"] = details["completed_uploads"]
            if details.get("failed_slot"):
                error_payload["failed_slot"] = details["failed_slot"]
            self.node.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: error")
            return Wan3Result(
                LingliuNullVideoAdapter(_redact_text(str(exc), secrets)),
                task_id,
                self.safe_json(error_payload, secrets),
                "",
                blank_frame,
            )


class LingliuWan3(LingliuBaseNode):
    CATEGORY = "Lingliu/Wan"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL", "尾帧图")
    FUNCTION = "generate_video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        del kwargs
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "生成音频": ("BOOLEAN", {"default": True}),
            "提示词扩展": ("BOOLEAN", {"default": True}),
            "水印": ("BOOLEAN", {"default": False}),
            "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
            "API密钥": (
                "STRING",
                {"default": "", "password": True, "tooltip": "留空使用全局密钥。"},
            ),
            "等待完成": ("BOOLEAN", {"default": True}),
            "最大轮询次数": ("INT", {"default": 180, "min": 1, "max": 720, "step": 1}),
            "轮询间隔": ("INT", {"default": 10, "min": 0, "max": 60, "step": 1}),
            "返回尾帧图": ("BOOLEAN", {"default": False}),
            "渠道": (list(WAN3_CHANNELS), {"default": "LL", "tooltip": "仅显示渠道缩写；LL 保持原有路由，XJ 使用新增 WAN3 路由。"}),
            "首帧图": ("IMAGE",),
            "尾帧图": ("IMAGE",),
            "参考图接口数量": ("INT", {"default": 3, "min": 0, "max": MAX_REFERENCE_IMAGES, "step": 1}),
            "参考视频接口数量": ("INT", {"default": 1, "min": 0, "max": MAX_REFERENCE_VIDEOS, "step": 1}),
            "参考音频接口数量": ("INT", {"default": 1, "min": 0, "max": MAX_REFERENCE_AUDIOS, "step": 1}),
        }
        optional.update({f"参考图{index}": ("IMAGE",) for index in range(1, MAX_REFERENCE_IMAGES + 1)})
        optional.update({f"参考视频{index}": (IO.VIDEO,) for index in range(1, MAX_REFERENCE_VIDEOS + 1)})
        optional.update({f"参考音频{index}": (IO.AUDIO,) for index in range(1, MAX_REFERENCE_AUDIOS + 1)})
        return {
            "required": {
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "模型": (list(WAN3_MODELS), {"default": "wan3.0-video"}),
                "生成模式": (list(WAN3_MODES), {"default": "文生视频"}),
                "时长": ("INT", {"default": 5, "min": 2, "max": 30, "step": 1}),
                "比例": (list(WAN3_RATIOS), {"default": "adaptive"}),
                "清晰度": (list(WAN3_RESOLUTIONS), {"default": "720p"}),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    async def generate_video(self, **kwargs):
        raw_assets = [
            ("首帧图", "image", "first_frame", kwargs.get("首帧图")),
            ("尾帧图", "image", "last_frame", kwargs.get("尾帧图")),
        ]
        raw_assets.extend(
            (f"参考图{index}", "image", "reference_image", kwargs.get(f"参考图{index}"))
            for index in range(1, MAX_REFERENCE_IMAGES + 1)
        )
        raw_assets.extend(
            (f"参考视频{index}", "video", "reference_video", kwargs.get(f"参考视频{index}"))
            for index in range(1, MAX_REFERENCE_VIDEOS + 1)
        )
        raw_assets.extend(
            (f"参考音频{index}", "audio", "reference_audio", kwargs.get(f"参考音频{index}"))
            for index in range(1, MAX_REFERENCE_AUDIOS + 1)
        )
        result = await Wan3Workflow(self).execute(
            prompt=kwargs.get("提示词", ""),
            model=kwargs.get("模型", "wan3.0-video"),
            mode=kwargs.get("生成模式", "文生视频"),
            duration=kwargs.get("时长", 5),
            ratio=kwargs.get("比例", "adaptive"),
            resolution=kwargs.get("清晰度", "720p"),
            generate_audio=kwargs.get("生成音频", True),
            prompt_extend=kwargs.get("提示词扩展", True),
            watermark=kwargs.get("水印", False),
            seed=kwargs.get("随机种子", 0),
            raw_assets=raw_assets,
            channel=kwargs.get("渠道", "LL"),
            api_key=kwargs.get("API密钥", ""),
            wait_for_completion=kwargs.get("等待完成", True),
            max_poll_attempts=kwargs.get("最大轮询次数", 180),
            poll_interval=kwargs.get("轮询间隔", 10),
            return_last_frame=kwargs.get("返回尾帧图", False),
            unique_id=kwargs.get("unique_id"),
        )
        return result.video, result.task_id, result.response, result.video_url, result.last_frame
