import asyncio
import json
import os
import re
import tempfile
import threading
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import quote

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
)
from ..utils import LingliuNullVideoAdapter, LingliuVideoAdapter, tensor2pil
from .nodes_openai_compatible import LingliuBaseNode


MODEL = "MiniMax-H3"
LOW_PRICE_MODEL = "MiniMax H3 低价"
H3_CHANNELS = ("LL", "MT")
FILES_CHANNEL = "6"
H3_CHANNEL = "17"
ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
ALLOWED_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
ALLOWED_RESOLUTIONS = ("2K", "1080p", "768P", "480p")
CHANNEL_RESOLUTIONS = {"LL": ("2K", "768P"), "MT": ("2K", "1080p", "768P", "480p")}

MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 15 * 1024 * 1024


def max_duration_seconds(channel, resolution):
    # MT follows the platform: 480p/768P run up to 30 s, the higher tiers 15.
    if channel == "MT" and resolution in ("480p", "768P"):
        return 30
    return 15


class H3Error(RuntimeError):
    def __init__(self, message, *, stage="validation", status_code=None, payload=None, details=None):
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code
        self.payload = payload
        self.details = details or {}


@dataclass(frozen=True)
class PreparedAsset:
    slot: str
    media_type: str
    role: str
    filename: str
    content_type: str
    data: bytes
    duration: float = 0.0


@dataclass(frozen=True)
class H3Result:
    video: object
    task_id: str
    response: str
    video_url: str


class MTH3VideoAdapter(LingliuVideoAdapter):
    """Fetch the authenticated MT result once, only when the VIDEO is used."""

    def __init__(self, video_url, config, http):
        super().__init__(video_url)
        self._config = dict(config)
        self._http = http
        self._cached = None
        self._lock = threading.Lock()

    def _local_video(self):
        with self._lock:
            if self._cached is None:
                # Reuse the existing content downloader: same-origin Bearer,
                # credential-free CDN redirects, real metadata and local cache.
                from .wan3_low_workflow import Wan3LowWorkflow

                download_config = {
                    "base_url": self._config["base_url"],
                    "api_key": self._config["h3_key"],
                    "timeout": self._config["timeout"],
                }
                try:
                    self._cached = Wan3LowWorkflow(http=self._http).download_result(self.video_url, download_config)
                except Exception as exc:
                    message = _redact_text(str(exc), [self._config["h3_key"]])
                    raise H3Error(message, stage="download") from None
                self.video_path = self._cached.video_path
            return self._cached

    def get_dimensions(self):
        return self._local_video().get_dimensions()

    def get_duration(self):
        return self._local_video().get_duration()

    def get_stream_source(self):
        return self._local_video().get_stream_source()

    def get_components(self):
        return self._local_video().get_components()

    def get_frame_rate(self):
        return self._local_video().get_frame_rate()

    def get_frame_count(self):
        return self._local_video().get_frame_count()

    def as_trimmed(self, *args, **kwargs):
        return self._local_video().as_trimmed(*args, **kwargs)

    def get_container_format(self):
        return self._local_video().get_container_format()

    def get_bit_depth(self):
        return self._local_video().get_bit_depth()

    def get_color_space(self):
        return self._local_video().get_color_space()

    def save_to(self, output, format="auto", codec="auto", metadata=None, **kwargs):
        return self._local_video().save_to(output, format=format, codec=codec, metadata=metadata, **kwargs)


class RequestsHTTPAdapter:
    def request(self, method, url, **kwargs):
        return requests.request(method, url, **kwargs)


class PyAVMediaProbeAdapter:
    @staticmethod
    def _duration(container, stream):
        if stream is not None and stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            import av

            return float(container.duration / av.time_base)
        return 0.0

    def probe_video(self, data, filename):
        try:
            import av

            container = av.open(BytesIO(data))
        except Exception as exc:
            raise H3Error(f"视频 {filename} 无法解析：{exc}") from exc
        try:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise H3Error(f"视频 {filename} 不包含视频轨。")
            audio_streams = [item for item in container.streams if item.type == "audio"]
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            return {
                "container": str(getattr(container.format, "name", "") or "").lower(),
                "video_codec": str(getattr(stream.codec_context, "name", "") or "").lower(),
                "audio_codecs": [str(getattr(item.codec_context, "name", "") or "").lower() for item in audio_streams],
                "duration": self._duration(container, stream),
                "fps": float(rate) if rate else 0.0,
            }
        finally:
            container.close()

    def probe_audio(self, data, filename):
        try:
            import av

            container = av.open(BytesIO(data))
        except Exception as exc:
            raise H3Error(f"音频 {filename} 无法解析：{exc}") from exc
        try:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise H3Error(f"音频 {filename} 不包含音频轨。")
            return {
                "container": str(getattr(container.format, "name", "") or "").lower(),
                "audio_codec": str(getattr(stream.codec_context, "name", "") or "").lower(),
                "duration": self._duration(container, stream),
            }
        finally:
            container.close()


def force_lingliu_channel_key(api_key, channel_id):
    """Derive a channel key without changing lingliu_channel_key compatibility."""
    key = str(api_key or "").strip()
    channel = str(channel_id or "").strip()
    if not key or not channel.isdigit():
        return key
    base_key = re.sub(r"-\d+$", "", key)
    return f"{base_key}-{channel}"


def _extract_url(payload):
    if not isinstance(payload, dict):
        return ""
    value = payload.get("url")
    if isinstance(value, str) and value.strip():
        return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        value = data.get("url")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_task_id(payload):
    if not isinstance(payload, dict):
        return ""
    for key in ("task_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    return _extract_task_id(data) if isinstance(data, dict) else ""


def _redact_text(text, secrets):
    result = str(text or "")
    for secret in sorted({str(item) for item in secrets if item}, key=len, reverse=True):
        result = result.replace(secret, "<redacted>")
    result = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer <redacted>", result)
    result = re.sub(r"(?i)sk-[A-Za-z0-9_.:-]+(?:-\d+)?", "sk-<redacted>", result)
    return result


def _redact(value, secrets):
    if isinstance(value, dict):
        return {str(key): _redact(item, secrets) for key, item in value.items() if str(key).lower() not in {"authorization", "api_key", "apikey"}}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact_text(value, secrets)
    return value


def _waveform_to_wav(audio_input):
    waveform = audio_input.get("waveform")
    sample_rate = int(audio_input.get("sample_rate") or 44100)
    if waveform is None or sample_rate <= 0:
        raise H3Error("Comfy AUDIO 输入缺少有效 waveform 或 sample_rate。")
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().float().numpy()
    else:
        waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.ndim != 2 or waveform.shape[0] < 1:
        raise H3Error(f"Comfy AUDIO waveform 应为 [C,T]，实际为 {tuple(waveform.shape)}。")
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    interleaved = pcm[0] if pcm.shape[0] == 1 else np.transpose(pcm, (1, 0)).reshape(-1)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(int(pcm.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes())
    return buffer.getvalue(), "audio.wav"


def _download_url(url):
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    filename = os.path.basename(str(url).split("?", 1)[0]) or "media.bin"
    return response.content, filename


def _read_media_input(media_input, default_suffix):
    if media_input is None:
        raise H3Error("素材输入为空。")
    if isinstance(media_input, Mapping):
        path = media_input.get("path") or media_input.get("file_path") or media_input.get("file") or media_input.get("filename")
        if path and os.path.isfile(str(path)):
            with open(str(path), "rb") as handle:
                return handle.read(), os.path.basename(str(path))
        url = media_input.get("url") or media_input.get("video_url") or media_input.get("audio_url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return _download_url(url)
    if isinstance(media_input, str):
        if os.path.isfile(media_input):
            with open(media_input, "rb") as handle:
                return handle.read(), os.path.basename(media_input)
        if media_input.startswith(("http://", "https://")):
            return _download_url(media_input)
        raise H3Error("素材字符串不是可读取的本地文件。")
    for attr in ("video_path", "audio_path", "path", "file_path"):
        path = getattr(media_input, attr, None)
        if isinstance(path, str) and os.path.isfile(path):
            with open(path, "rb") as handle:
                return handle.read(), os.path.basename(path)
    for attr in ("video_url", "audio_url", "url"):
        url = getattr(media_input, attr, None)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return _download_url(url)
    get_stream = getattr(media_input, "get_stream_source", None)
    if callable(get_stream):
        source = get_stream()
        if isinstance(source, str) and os.path.isfile(source):
            with open(source, "rb") as handle:
                return handle.read(), os.path.basename(source)
        if hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            data = source.read()
            if data:
                return data, f"media{default_suffix}"
    save_to = getattr(media_input, "save_to", None)
    if callable(save_to):
        buffer = BytesIO()
        try:
            save_to(buffer)
            if hasattr(buffer, "seek"):
                buffer.seek(0)
            data = buffer.read()
            if data:
                return data, f"media{default_suffix}"
        except Exception:
            pass
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=default_suffix) as handle:
                temp_path = handle.name
            save_to(temp_path)
            with open(temp_path, "rb") as handle:
                data = handle.read()
            if data:
                return data, os.path.basename(temp_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    raise H3Error("无法从 Comfy 素材对象读取文件内容。")


class MiniMaxH3Workflow:
    """Deep module: validation, upload, request construction, creation and polling."""

    def __init__(self, http=None, media_probe=None, sleep=None):
        self.http = http or RequestsHTTPAdapter()
        self.media_probe = media_probe or PyAVMediaProbeAdapter()
        self.sleep = sleep or asyncio.sleep

    @staticmethod
    def _image_batch_size(image):
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) == 4:
            return int(shape[0])
        if isinstance(image, (list, tuple)):
            return len(image)
        return 1

    def prepare_image(self, slot, role, image):
        batch_size = self._image_batch_size(image)
        if batch_size != 1:
            raise H3Error(f"{slot} 每个插槽批次只能为 1，当前为 {batch_size}。")
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, (list, tuple)) and image and isinstance(image[0], Image.Image):
            pil_image = image[0]
        else:
            try:
                pil_image = tensor2pil(image)[0]
            except Exception as exc:
                raise H3Error(f"{slot} 不是可识别的 IMAGE：{exc}") from exc
        width, height = pil_image.size
        if not (256 <= width <= 5760 and 256 <= height <= 5760):
            raise H3Error(f"{slot} 尺寸必须为 256–5760，当前为 {width}x{height}。")
        ratio = width / float(height)
        if not (0.4 <= ratio <= 2.5):
            raise H3Error(f"{slot} 宽高比必须为 0.4–2.5，当前为 {ratio:.4f}。")
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        data = buffer.getvalue()
        if len(data) > MAX_IMAGE_BYTES:
            raise H3Error(f"{slot} PNG 编码后超过 30MB。")
        return PreparedAsset(slot, "image", role, f"{slot}.png", "image/png", data)

    def prepare_video(self, slot, role, video):
        data, filename = _read_media_input(video, ".mp4")
        if len(data) > MAX_VIDEO_BYTES:
            raise H3Error(f"{slot} 超过 50MB。")
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in {".mp4", ".mov"}:
            raise H3Error(f"{slot} 只支持 MP4/MOV，当前文件为 {filename}。")
        info = self.media_probe.probe_video(data, filename)
        if not any(token in str(info.get("container", "")) for token in ("mp4", "mov")):
            raise H3Error(f"{slot} 容器不是 MP4/MOV。")
        if str(info.get("video_codec", "")).lower() not in {"h264", "hevc", "h265"}:
            raise H3Error(f"{slot} 视频编码必须为 H264/H265。")
        invalid_audio = [
            item
            for item in info.get("audio_codecs", [])
            if str(item).strip().lower() != "aac" and not str(item).strip().lower().startswith("mp3")
        ]
        if invalid_audio:
            raise H3Error(f"{slot} 音轨编码必须为 AAC/MP3，当前为 {invalid_audio[0]}。")
        duration = float(info.get("duration") or 0)
        fps = float(info.get("fps") or 0)
        if not (2 <= duration <= 15):
            raise H3Error(f"{slot} 时长必须为 2–15 秒，当前为 {duration:.3f} 秒。")
        if not (23.976 <= fps <= 60):
            raise H3Error(f"{slot} FPS 必须为 23.976–60，当前为 {fps:.3f}。")
        mime = "video/quicktime" if suffix == ".mov" else "video/mp4"
        return PreparedAsset(slot, "video", role, filename, mime, data, duration)

    def prepare_audio(self, slot, role, audio):
        if isinstance(audio, Mapping) and audio.get("waveform") is not None:
            data, filename = _waveform_to_wav(audio)
        else:
            data, filename = _read_media_input(audio, ".wav")
        if len(data) > MAX_AUDIO_BYTES:
            raise H3Error(f"{slot} 超过 15MB。")
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in {".wav", ".mp3"}:
            raise H3Error(f"{slot} 只支持 WAV/MP3，当前文件为 {filename}。")
        info = self.media_probe.probe_audio(data, filename)
        container = str(info.get("container", "")).lower()
        if suffix == ".wav" and "wav" not in container:
            raise H3Error(f"{slot} 扩展名为 WAV 但内容不是 WAV。")
        if suffix == ".mp3" and "mp3" not in container:
            raise H3Error(f"{slot} 扩展名为 MP3 但内容不是 MP3。")
        duration = float(info.get("duration") or 0)
        if not (2 <= duration <= 15):
            raise H3Error(f"{slot} 时长必须为 2–15 秒，当前为 {duration:.3f} 秒。")
        mime = "audio/mpeg" if suffix == ".mp3" else "audio/wav"
        return PreparedAsset(slot, "audio", role, filename, mime, data, duration)

    def validate_and_prepare(self, raw_assets, mode):
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
                raise H3Error(f"不支持的素材类型：{media_type}")
        images = sum(item.media_type == "image" for item in prepared)
        videos = sum(item.media_type == "video" for item in prepared)
        audios = sum(item.media_type == "audio" for item in prepared)
        if mode == "frames":
            if not prepared:
                raise H3Error("至少连接首帧图或尾帧图之一。")
            if any(item.media_type != "image" for item in prepared) or len(prepared) > 2:
                raise H3Error("首尾帧节点只接受首帧图和尾帧图。")
        else:
            if images > 9 or videos > 3 or audios > 3 or len(prepared) > 12:
                raise H3Error("参考素材限制为图≤9、视频≤3、音频≤3、总数≤12。")
            if images == 0 and videos == 0:
                raise H3Error("参考节点至少需要一张图片或一个视频；音频不能单独使用。")
        video_duration = sum(item.duration for item in prepared if item.media_type == "video")
        audio_duration = sum(item.duration for item in prepared if item.media_type == "audio")
        if video_duration > 15 + 1e-6:
            raise H3Error(f"参考视频合计不能超过 15 秒，当前为 {video_duration:.3f} 秒。")
        if audio_duration > 15 + 1e-6:
            raise H3Error(f"参考音频合计不能超过 15 秒，当前为 {audio_duration:.3f} 秒。")
        return prepared

    @staticmethod
    def validate_controls(prompt, duration, ratio, resolution, max_poll_attempts, poll_interval, channel="LL"):
        if not isinstance(prompt, str) or not prompt.strip():
            raise H3Error("提示词必须包含 1–7000 个 Unicode 字符。")
        if len(prompt) > 7000:
            raise H3Error("提示词超过 7000 个 Unicode 字符。")
        if ratio not in ALLOWED_RATIOS:
            raise H3Error("比例不受支持。")
        if channel == "MT":
            if resolution not in CHANNEL_RESOLUTIONS["MT"]:
                raise H3Error("MT 渠道清晰度只支持 2K、1080p、768P 或 480p。")
        elif resolution not in CHANNEL_RESOLUTIONS["LL"]:
            raise H3Error("清晰度只支持 2K 或 768P（1080p、480p 请选 MT 渠道）。")
        max_duration = max_duration_seconds(channel, resolution)
        if isinstance(duration, bool) or int(duration) != duration or not (4 <= int(duration) <= max_duration):
            if channel == "MT":
                raise H3Error(f"MT 渠道 {resolution} 时长必须是 4–{max_duration} 的整数（480p/768P 最长 30 秒，1080p/2K 最长 15 秒）。")
            raise H3Error("时长必须是 4–15 的整数。")
        if not (10 <= int(max_poll_attempts) <= 720):
            raise H3Error("最大轮询次数必须为 10–720。")
        if not (2 <= int(poll_interval) <= 60):
            raise H3Error("轮询间隔必须为 2–60 秒。")

    @staticmethod
    def resolve_config(api_key_override, channel="LL"):
        channel = str(channel or "LL").strip().upper()
        if channel not in H3_CHANNELS:
            raise H3Error("渠道只能选择 LL 或 MT。", stage="route")
        config = get_config()
        provider = normalize_provider(config.get("api_provider"))
        if provider not in ALLOWED_PROVIDERS:
            raise H3Error("MiniMax-H3 只支持灵流线路，已在上传/创建前停止。", stage="route")
        raw_key = str(api_key_override or config.get("api_key") or "").strip()
        if not raw_key:
            raise H3Error("MiniMax-H3 API 密钥为空，请连接节点密钥或设置全局密钥。", stage="auth")
        if channel == "MT":
            files_key = raw_key
            h3_key = raw_key
            public_model = LOW_PRICE_MODEL
            create_path = "/v1/videos"
            query_path = "/v1/videos/{task_id}"
            content_path = "/v1/videos/{task_id}/content"
        else:
            files_key = force_lingliu_channel_key(raw_key, FILES_CHANNEL)
            h3_key = force_lingliu_channel_key(raw_key, H3_CHANNEL)
            public_model = MODEL
            create_path = "/v2/video_generation"
            query_path = "/v2/query/video_generation/{task_id}"
            content_path = ""
        return {
            "provider": provider,
            "base_url": str(config.get("base_url") or "").rstrip("/"),
            "timeout": int(config.get("timeout") or 300),
            "raw_key": raw_key,
            "files_key": files_key,
            "h3_key": h3_key,
            "channel": channel,
            "public_model": public_model,
            "create_path": create_path,
            "query_path": query_path,
            "content_path": content_path,
        }

    @staticmethod
    def build_body(prompt, duration, ratio, resolution, prepared, uploaded_urls, *, model=MODEL, reference_video_seconds=0.0, context_ir_enabled=None):
        content = [{"type": "text", "text": prompt}]
        for asset, url in zip(prepared, uploaded_urls):
            field = f"{asset.media_type}_url"
            content.append({"type": field, "role": asset.role, field: {"url": url}})
        body = {
            "model": model,
            "content": content,
            "resolution": resolution,
            "duration": int(duration),
            "ratio": ratio,
        }
        if reference_video_seconds:
            body["reference_video_seconds"] = float(reference_video_seconds)
        if context_ir_enabled is not None:
            # Always explicit on MT so the paid option is never billed by an upstream default.
            body["context_ir_enabled"] = bool(context_ir_enabled)
        return body

    @staticmethod
    def _decode_response(response, stage):
        try:
            payload = response.json()
        except Exception as exc:
            preview = str(getattr(response, "text", "") or "")[:500]
            raise H3Error(f"{stage} 响应不是 JSON：{preview}", stage=stage, status_code=getattr(response, "status_code", None)) from exc
        if not isinstance(payload, dict):
            raise H3Error(f"{stage} 响应必须是 JSON 对象。", stage=stage, status_code=getattr(response, "status_code", None), payload=payload)
        return payload

    async def _request_json(self, method, url, config, *, stage, key, retry_network, retry_status, **kwargs):
        last_error = None
        extra_headers = dict(kwargs.pop("extra_headers", {}) or {})
        request_kwargs = dict(kwargs)
        mt_create = config.get("channel") == "MT" and stage == "create"
        attempts = 1 if mt_create else 3
        if config.get("channel") == "MT":
            request_kwargs["allow_redirects"] = False
        for attempt in range(1, attempts + 1):
            try:
                response = await asyncio.to_thread(
                    self.http.request,
                    method,
                    url,
                    headers={"Accept": "application/json", "Authorization": f"Bearer {key}", **extra_headers},
                    timeout=config["timeout"],
                    **request_kwargs,
                )
            except Exception as exc:
                last_error = exc
                if not retry_network or attempt == attempts or not isinstance(exc, (requests.exceptions.RequestException, OSError)):
                    uncertainty = "；创建提交结果不确定，已禁止自动重发以避免重复付费" if stage == "create" else ""
                    raise H3Error(f"{stage} 网络错误：{exc}{uncertainty}", stage=stage) from exc
                await self.sleep(min(2 ** (attempt - 1), 4))
                continue
            status_code = int(getattr(response, "status_code", 200) or 200)
            can_retry = not mt_create and (status_code in retry_status or status_code >= 500)
            try:
                payload = self._decode_response(response, stage)
            except H3Error as exc:
                last_error = exc
                if can_retry and attempt < attempts:
                    await self.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise
            if status_code < 400 and (config.get("channel") != "MT" or 200 <= status_code < 300):
                return payload
            task_id = _extract_task_id(payload)
            if stage == "create" and task_id and not mt_create:
                return payload
            error = H3Error(
                f"{stage} HTTP {status_code}",
                stage=stage,
                status_code=status_code,
                payload=payload,
            )
            last_error = error
            if not can_retry or attempt == attempts:
                raise error
            await self.sleep(min(2 ** (attempt - 1), 4))
        raise last_error or H3Error(f"{stage} 请求失败。", stage=stage)

    async def upload_all(self, prepared, config):
        urls = []
        summaries = []
        endpoint = build_url(config["base_url"], "/v1/files")
        for asset in prepared:
            try:
                payload = await self._request_json(
                    "POST",
                    endpoint,
                    config,
                    stage="upload",
                    key=config["files_key"],
                    retry_network=True,
                    retry_status={429},
                    files={"file": (asset.filename, asset.data, asset.content_type)},
                )
                url = _extract_url(payload)
                if not url.startswith(("http://", "https://")):
                    raise H3Error(f"{asset.slot} 上传响应缺少有效 url。", stage="upload", payload=payload)
            except H3Error as exc:
                details = dict(getattr(exc, "details", {}) or {})
                details.update({"completed_uploads": list(summaries), "failed_slot": asset.slot})
                raise H3Error(
                    str(exc),
                    stage="upload",
                    status_code=getattr(exc, "status_code", None),
                    payload=getattr(exc, "payload", None),
                    details=details,
                ) from exc
            urls.append(url)
            summaries.append({"slot": asset.slot, "type": asset.media_type, "role": asset.role, "bytes": len(asset.data), "url": url})
        return urls, summaries

    @staticmethod
    def _task_payload(payload):
        if not isinstance(payload, dict):
            return {}
        task = payload.get("task")
        return task if isinstance(task, dict) else payload

    @staticmethod
    def _status(payload):
        status = MiniMaxH3Workflow._task_payload(payload).get("status")
        return str(status or "").strip().lower()

    @staticmethod
    def _result_url(payload):
        task_payload = MiniMaxH3Workflow._task_payload(payload)
        content = task_payload.get("content")
        if isinstance(content, dict) and isinstance(content.get("url"), str):
            content_url = content["url"].strip()
            if content_url:
                return content_url
        if isinstance(payload, dict) and isinstance(payload.get("task"), dict):
            result_url = task_payload.get("result_url")
            if isinstance(result_url, str):
                return result_url.strip()
        for key in ("video_url", "videoUrl", "url"):
            value = task_payload.get(key) if isinstance(task_payload, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _json(self, payload, secrets):
        return json.dumps(mask_provider_payload(_redact(payload, secrets)), ensure_ascii=False, indent=2)

    async def execute(
        self,
        *,
        mode,
        prompt,
        duration,
        ratio,
        resolution,
        raw_assets,
        api_key="",
        channel="LL",
        wait_for_completion=True,
        max_poll_attempts=120,
        poll_interval=10,
        notify=None,
        context_ir_enabled=False,
    ):
        task_id = ""
        secrets = [api_key]
        channel = str(channel or "LL").strip().upper()
        envelope = {"model": LOW_PRICE_MODEL if channel == "MT" else MODEL, "mode": mode, "channel": channel}
        try:
            if channel not in H3_CHANNELS:
                raise H3Error("渠道只能选择 LL 或 MT。", stage="route")
            if channel == "MT" and ratio == "adaptive":
                ratio = "16:9"
            self.validate_controls(prompt, duration, ratio, resolution, max_poll_attempts, poll_interval, channel)
            if mode == "frames" and channel == "LL" and ratio != "adaptive":
                raise H3Error("首尾帧节点比例必须为 adaptive。")
            config = self.resolve_config(api_key, channel)
            secrets.extend([config["raw_key"], config["files_key"], config["h3_key"]])
            prepared = self.validate_and_prepare(raw_assets, mode)
            uploaded_urls, upload_summary = await self.upload_all(prepared, config)
            reference_video_seconds = sum(item.duration for item in prepared if item.media_type == "video")
            body = self.build_body(
                prompt,
                duration,
                ratio,
                resolution,
                prepared,
                uploaded_urls,
                model=config["public_model"],
                reference_video_seconds=reference_video_seconds if channel == "MT" else 0.0,
                context_ir_enabled=bool(context_ir_enabled) if channel == "MT" else None,
            )
            envelope.update({"provider": config["provider"], "request": body, "uploads": upload_summary})
            create_payload = await self._request_json(
                "POST",
                build_url(config["base_url"], config["create_path"]),
                config,
                stage="create",
                key=config["h3_key"],
                retry_network=False,
                retry_status={429},
                extra_headers={"Content-Type": "application/json"},
                json=body,
            )
            envelope["create_response"] = create_payload
            task_id = _extract_task_id(create_payload)
            if not task_id:
                raise H3Error("创建响应缺少 task_id。", stage="create", payload=create_payload)
            envelope["task_id"] = task_id
            if notify:
                notify(task_id, "submitted")
            if not wait_for_completion:
                envelope["status"] = "submitted"
                return H3Result(LingliuNullVideoAdapter("任务已提交，未等待完成。"), task_id, self._json(envelope, secrets), "")

            query_url = build_url(config["base_url"], config["query_path"].format(task_id=quote(task_id, safe="")))
            pending_states = {"queued", "pending", "submitted", "processing", "running", "in_progress"}
            failed_states = {"failed", "cancelled"}
            success_states = {"succeeded", "completed"}
            if channel == "MT":
                # MT can expose a transient unknown/empty status before its
                # upstream state is synchronized. Timestamps/URLs do not make
                # this a terminal result. Keep polling the same public task.
                pending_states.update({"", "unknown", "not_start"})
                failed_states.update({"failure", "error", "canceled", "rejected", "expired"})
                success_states.update({"success", "complete"})
            for attempt in range(int(max_poll_attempts)):
                query_payload = await self._request_json(
                    "GET",
                    query_url,
                    config,
                    stage="query",
                    key=config["h3_key"],
                    retry_network=True,
                    retry_status={429},
                )
                envelope["query_response"] = query_payload
                status = self._status(query_payload)
                if notify:
                    notify(task_id, status or "unknown")
                if status in pending_states:
                    if attempt + 1 < int(max_poll_attempts):
                        await self.sleep(int(poll_interval))
                    continue
                if status in failed_states:
                    envelope["status"] = status
                    message = f"MiniMax-H3 任务终态为 {status}。"
                    envelope["error"] = {"stage": "query", "message": message, "upstream": query_payload}
                    return H3Result(LingliuNullVideoAdapter(message), task_id, self._json(envelope, secrets), "")
                if status in success_states:
                    if channel == "MT":
                        # Use our known authenticated content endpoint rather
                        # than trusting a URL in a provisional task response.
                        video_url = query_url + "/content"
                        envelope["status"] = status
                        return H3Result(MTH3VideoAdapter(video_url, config, self.http), task_id, self._json(envelope, secrets), video_url)
                    video_url = self._result_url(query_payload)
                    if not video_url:
                        raise H3Error("succeeded 响应缺少 content.url。", stage="query", payload=query_payload)
                    envelope["status"] = status
                    return H3Result(LingliuVideoAdapter(video_url), task_id, self._json(envelope, secrets), video_url)
                raise H3Error(f"查询返回未知状态：{status or '<empty>'}。", stage="query", payload=query_payload)
            envelope["status"] = "timeout"
            envelope["max_poll_attempts"] = int(max_poll_attempts)
            timeout_message = "轮询超时。"
            if channel == "MT":
                timeout_message += " 原任务可能仍在生成，请使用任务 ID 查询原任务，不要重复提交生成。"
            envelope["error"] = {"stage": "query", "message": timeout_message}
            return H3Result(LingliuNullVideoAdapter(timeout_message), task_id, self._json(envelope, secrets), "")
        except Exception as exc:
            stage = getattr(exc, "stage", "internal")
            details = dict(getattr(exc, "details", {}) or {})
            error_payload = {
                **envelope,
                "status": "error",
                "task_id": task_id,
                "error": {
                    "stage": stage,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                    "upstream": getattr(exc, "payload", None),
                    "details": details or None,
                },
            }
            if "completed_uploads" in details:
                error_payload["uploads"] = details["completed_uploads"]
            if details.get("failed_slot"):
                error_payload["failed_slot"] = details["failed_slot"]
            if notify:
                notify(task_id, "error")
            message = _redact_text(str(exc), secrets)
            return H3Result(LingliuNullVideoAdapter(message), task_id, self._json(error_payload, secrets), "")


class _MiniMaxH3NodeBase(LingliuBaseNode):
    CATEGORY = "Lingliu/MiniMax-H3"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL")
    FUNCTION = "generate_video"

    def _notify(self, unique_id):
        def notify(task_id, status):
            if task_id:
                self.publish_task_id(unique_id, task_id, append=False)
            self.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status}", append=False)

        return notify

    async def _execute(self, **kwargs):
        result = await MiniMaxH3Workflow().execute(notify=self._notify(kwargs.pop("unique_id", None)), **kwargs)
        return result.video, result.task_id, result.response, result.video_url


class LingliuMiniMaxH3Frames(_MiniMaxH3NodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "时长": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1, "tooltip": "LL 为 4–15 秒；MT 渠道 480p/768P 最长 30 秒（超过 15 秒按 2 倍计费），1080p/2K 最长 15 秒。"}),
                "清晰度": (list(ALLOWED_RESOLUTIONS), {"default": "2K", "tooltip": "LL 支持 2K/768P；MT 支持 2K/1080p/768P/480p。"}),
            },
            "optional": {
                "首帧图": ("IMAGE",),
                "尾帧图": ("IMAGE",),
                "API密钥": ("STRING", {"default": ""}),
                "渠道": (list(H3_CHANNELS), {"default": "LL", "tooltip": "仅显示渠道缩写；LL 保持原有 H3 路由，MT 使用新增低价路由。"}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 10, "max": 720, "step": 1}),
                "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
                "提示词优化": ("BOOLEAN", {"default": False, "tooltip": "仅 MT 渠道生效：平台先优化提示词再生成，每次额外 +0.8 流星；LL 渠道忽略。"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    async def generate_video(self, 提示词, 时长, 清晰度, 首帧图=None, 尾帧图=None, API密钥="", 渠道="LL", 等待完成=True, 最大轮询次数=120, 轮询间隔=10, 提示词优化=False, unique_id=None):
        return await self._execute(
            mode="frames",
            prompt=提示词,
            duration=时长,
            ratio="adaptive" if 渠道 == "LL" else "16:9",
            resolution=清晰度,
            raw_assets=[
                ("首帧图", "image", "first_frame", 首帧图),
                ("尾帧图", "image", "last_frame", 尾帧图),
            ],
            api_key=API密钥,
            channel=渠道,
            wait_for_completion=等待完成,
            max_poll_attempts=最大轮询次数,
            poll_interval=轮询间隔,
            context_ir_enabled=提示词优化,
            unique_id=unique_id,
        )


class LingliuMiniMaxH3Reference(_MiniMaxH3NodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            **{f"参考图{index}": ("IMAGE",) for index in range(1, 10)},
            **{f"参考视频{index}": (IO.VIDEO,) for index in range(1, 4)},
            **{f"参考音频{index}": (IO.AUDIO,) for index in range(1, 4)},
            "API密钥": ("STRING", {"default": ""}),
            "渠道": (list(H3_CHANNELS), {"default": "LL", "tooltip": "仅显示渠道缩写；LL 保持原有 H3 路由，MT 使用新增低价路由。"}),
            "等待完成": ("BOOLEAN", {"default": True}),
            "最大轮询次数": ("INT", {"default": 120, "min": 10, "max": 720, "step": 1}),
            "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
            "提示词优化": ("BOOLEAN", {"default": False, "tooltip": "仅 MT 渠道生效：平台先优化提示词再生成，每次额外 +0.8 流星；LL 渠道忽略。"}),
        }
        return {
            "required": {
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "时长": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1, "tooltip": "LL 为 4–15 秒；MT 渠道 480p/768P 最长 30 秒（超过 15 秒按 2 倍计费），1080p/2K 最长 15 秒。"}),
                "比例": (list(ALLOWED_RATIOS), {"default": "adaptive"}),
                "清晰度": (list(ALLOWED_RESOLUTIONS), {"default": "2K", "tooltip": "LL 支持 2K/768P；MT 支持 2K/1080p/768P/480p。"}),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    async def generate_video(self, 提示词, 时长, 比例, 清晰度, **kwargs):
        raw_assets = []
        for index in range(1, 10):
            raw_assets.append((f"参考图{index}", "image", "reference_image", kwargs.get(f"参考图{index}")))
        for index in range(1, 4):
            raw_assets.append((f"参考视频{index}", "video", "reference_video", kwargs.get(f"参考视频{index}")))
        for index in range(1, 4):
            raw_assets.append((f"参考音频{index}", "audio", "reference_audio", kwargs.get(f"参考音频{index}")))
        return await self._execute(
            mode="reference",
            prompt=提示词,
            duration=时长,
            ratio=比例,
            resolution=清晰度,
            raw_assets=raw_assets,
            api_key=kwargs.get("API密钥", ""),
            channel=kwargs.get("渠道", "LL"),
            wait_for_completion=kwargs.get("等待完成", True),
            max_poll_attempts=kwargs.get("最大轮询次数", 120),
            poll_interval=kwargs.get("轮询间隔", 10),
            context_ir_enabled=kwargs.get("提示词优化", False),
            unique_id=kwargs.get("unique_id"),
        )
