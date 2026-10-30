import asyncio
import io
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

from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    get_config,
    mask_provider_payload,
    normalize_provider,
)
from ..utils import (
    LingliuNullVideoAdapter,
    LingliuVideoAdapter,
    blank_image_tensor,
    load_image_from_candidate,
    pil2tensor,
    tensor2pil,
)
from .nodes_openai_compatible import LingliuBaseNode


FILES_CHANNEL = 6
MAX_REFERENCE_IMAGES = 30
MAX_REFERENCE_VIDEOS = 10
MAX_REFERENCE_AUDIOS = 10
MAX_REFERENCE_MEDIA = 50
UPLOAD_CONCURRENCY = 4

ALLOWED_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
ALLOWED_RESOLUTIONS = ("720p", "480p")
ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}


class Seedance25Error(RuntimeError):
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
class Seedance25Result:
    video: object
    task_id: str
    response: str
    video_url: str
    last_frame: object


@dataclass(frozen=True)
class Seedance25Route:
    node_id: str
    model: str
    generation_channel: int
    create_path: str
    query_path_template: str
    include_output_options: bool = False

    def build_payload(
        self,
        *,
        prompt,
        content,
        duration,
        ratio,
        resolution,
        reference_video_seconds=0.0,
    ):
        payload = {
            "model": self.model,
            "content": content,
            "duration": int(duration),
            "ratio": ratio,
            "resolution": resolution,
        }
        if reference_video_seconds:
            payload["reference_video_seconds"] = round(float(reference_video_seconds), 3)
        if self.include_output_options:
            payload["priority"] = 0
            payload["output_format"] = "mp4"
        return payload

    def query_path(self, task_id):
        return self.query_path_template.format(task_id=task_id)


STANDARD_ROUTE = Seedance25Route(
    node_id="lingliu_seedance2.5",
    model="doubao-seedance-2.5",
    generation_channel=6,
    create_path="/seedance/v3/contents/generations/tasks",
    query_path_template="/seedance/v3/contents/generations/tasks/{task_id}",
)

OFFICIAL_ROUTE = Seedance25Route(
    node_id="lingliu_seedance2.5_official",
    model="doubao-seedance-2-5-260628",
    generation_channel=5,
    create_path="/v1/video/generations",
    query_path_template="/v1/video/generations/{task_id}",
    include_output_options=True,
)


class RequestsHTTPAdapter:
    def request(self, method, url, **kwargs):
        return requests.request(method, url, **kwargs)


def force_channel_key(api_key, channel_id):
    key = str(api_key or "").strip()
    channel = str(channel_id or "").strip()
    if not key or not channel.isdigit():
        return key
    base_key = re.sub(r"-\d+$", "", key)
    return f"{base_key}-{channel}"


def _redact_text(value, secrets):
    text = str(value or "")
    for secret in sorted({str(item) for item in secrets if item}, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    text = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer <redacted>", text)
    return re.sub(r"(?i)sk-[A-Za-z0-9_.:-]+(?:-\d+)?", "sk-<redacted>", text)


def _redact(value, secrets):
    if isinstance(value, dict):
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


def _nested_dicts(payload):
    if not isinstance(payload, dict):
        return
    yield payload
    for key in ("data", "result", "content", "metadata", "output", "error"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            yield from _nested_dicts(nested)


def _extract_string(payload, keys):
    wanted = tuple(keys)
    for container in _nested_dicts(payload):
        for key in wanted:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_task_id(payload):
    return _extract_string(payload, ("task_id", "id", "taskId"))


def _extract_status(payload):
    raw = _extract_string(payload, ("status", "state", "task_status"))
    status = raw.lower()
    aliases = {
        "success": "succeeded",
        "completed": "succeeded",
        "complete": "succeeded",
        "failure": "failed",
        "fail": "failed",
        "canceled": "cancelled",
        "in_progress": "processing",
        "not_start": "pending",
    }
    return aliases.get(status, status)


def _extract_video_url(payload):
    url = _extract_string(payload, ("video_url", "videoUrl", "result_url", "url", "video"))
    if url:
        return url
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("url") or "").strip()
                output_type = str(item.get("outputType") or item.get("type") or "").lower()
                if candidate and (output_type in {"mp4", "mov", "video"} or candidate.lower().endswith((".mp4", ".mov"))):
                    return candidate
    return ""


def _extract_last_frame_url(payload):
    return _extract_string(
        payload,
        (
            "last_frame_url",
            "lastFrameUrl",
            "last_frame_image_url",
            "lastframe_url",
            "keyframe_url",
            "keyframeUrl",
        ),
    )


def _waveform_to_wav_bytes(audio_input):
    waveform = audio_input.get("waveform")
    sample_rate = int(audio_input.get("sample_rate") or 44100)
    if waveform is None or sample_rate <= 0:
        raise Seedance25Error("音频输入缺少有效的 waveform 或 sample_rate。")
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().float().numpy()
    else:
        waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.ndim != 2 or waveform.shape[0] < 1:
        raise Seedance25Error(f"音频 waveform 应为 [C,T]，实际为 {tuple(waveform.shape)}。")
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
    interleaved = pcm[0] if pcm.shape[0] == 1 else np.transpose(pcm, (1, 0)).reshape(-1)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(int(pcm.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes())
    return buffer.getvalue(), "reference_audio.wav", "audio/wav"


def _download_media(url):
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    filename = os.path.basename(str(url).split("?", 1)[0]) or "reference_media.bin"
    return response.content, filename


def _read_media_input(media_input, default_suffix):
    if media_input is None:
        raise Seedance25Error("素材输入为空。")
    if isinstance(media_input, Mapping):
        path = (
            media_input.get("path")
            or media_input.get("file_path")
            or media_input.get("file")
            or media_input.get("filename")
        )
        if path and os.path.isfile(str(path)):
            with open(str(path), "rb") as handle:
                return handle.read(), os.path.basename(str(path))
        url = media_input.get("url") or media_input.get("video_url") or media_input.get("audio_url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return _download_media(url)
    if isinstance(media_input, str):
        if os.path.isfile(media_input):
            with open(media_input, "rb") as handle:
                return handle.read(), os.path.basename(media_input)
        if media_input.startswith(("http://", "https://")):
            return _download_media(media_input)
    for attr in ("video_path", "audio_path", "path", "file_path"):
        path = getattr(media_input, attr, None)
        if isinstance(path, str) and os.path.isfile(path):
            with open(path, "rb") as handle:
                return handle.read(), os.path.basename(path)
    for attr in ("video_url", "audio_url", "url"):
        url = getattr(media_input, attr, None)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return _download_media(url)
    # save_to materializes slices/trims; reading the original stream first would
    # silently upload the unedited source instead of the VIDEO wired by the user.
    save_to = getattr(media_input, "save_to", None)
    if callable(save_to):
        buffer = BytesIO()
        try:
            save_to(buffer)
            buffer.seek(0)
            data = buffer.read()
            if data:
                return data, f"reference_media{default_suffix}"
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
                return data, f"reference_media{default_suffix}"
    raise Seedance25Error("无法从该 ComfyUI 素材对象读取文件内容。")


def _probe_video_duration(data, filename):
    try:
        import av

        container = av.open(BytesIO(data))
    except Exception as exc:
        raise Seedance25Error(f"{filename} 无法解析视频时长：{exc}") from exc
    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise Seedance25Error(f"{filename} 不包含视频轨。")
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            duration = 0.0
        if not np.isfinite(duration) or duration <= 0:
            raise Seedance25Error(f"{filename} 无法得到有效视频时长。")
        return duration
    finally:
        container.close()


def _extract_last_frame_from_video(video_url):
    if not video_url:
        return None
    temp_path = ""
    capture = None
    try:
        import cv2

        response = requests.get(video_url, timeout=300, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
            temp_path = handle.name
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            return None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame = None
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
            ok, frame = capture.read()
            if not ok:
                frame = None
        if frame is None:
            while True:
                ok, current = capture.read()
                if not ok or current is None:
                    break
                frame = current
        if frame is None:
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


class Seedance25Workflow:
    """Deep module hiding validation, upload, routing, creation and polling."""

    def __init__(self, route, http=None, sleep=None):
        self.route = route
        self.http = http or RequestsHTTPAdapter()
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
            raise Seedance25Error(f"{slot} 每个插槽只能连接 1 张图片，当前批次为 {batch_size}。")
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, (list, tuple)) and image and isinstance(image[0], Image.Image):
            pil_image = image[0]
        else:
            try:
                pil_image = tensor2pil(image)[0]
            except Exception as exc:
                raise Seedance25Error(f"{slot} 不是可识别的 IMAGE：{exc}") from exc
        buffer = BytesIO()
        pil_image.convert("RGB").save(buffer, format="PNG")
        return PreparedAsset(slot, "image", role, f"{slot}.png", "image/png", buffer.getvalue())

    def prepare_video(self, slot, role, video):
        data, filename = _read_media_input(video, ".mp4")
        filename = filename or f"{slot}.mp4"
        duration = _probe_video_duration(data, filename)
        content_type = mimetypes.guess_type(filename)[0] or "video/mp4"
        return PreparedAsset(slot, "video", role, filename, content_type, data, duration)

    def prepare_audio(self, slot, role, audio):
        if isinstance(audio, Mapping) and audio.get("waveform") is not None:
            data, filename, content_type = _waveform_to_wav_bytes(audio)
        else:
            data, filename = _read_media_input(audio, ".wav")
            filename = filename or f"{slot}.wav"
            content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
        return PreparedAsset(slot, "audio", role, filename, content_type, data)

    def validate_and_prepare(self, raw_assets, ratio):
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
                raise Seedance25Error(f"不支持的素材类型：{media_type}")

        first_frames = [item for item in prepared if item.role == "first_frame"]
        last_frames = [item for item in prepared if item.role == "last_frame"]
        references = [item for item in prepared if item.role.startswith("reference_")]
        image_count = sum(item.role == "reference_image" for item in prepared)
        video_count = sum(item.role == "reference_video" for item in prepared)
        audio_count = sum(item.role == "reference_audio" for item in prepared)

        if len(first_frames) > 1 or len(last_frames) > 1:
            raise Seedance25Error("首帧图和尾帧图各只能有一个。")
        if last_frames and not first_frames:
            raise Seedance25Error("连接尾帧图时必须同时连接首帧图。")
        if (first_frames or last_frames) and references:
            raise Seedance25Error("首尾帧模式不能与参考图、参考视频或参考音频混用。")
        if (first_frames or last_frames) and ratio != "adaptive":
            raise Seedance25Error("首尾帧模式的比例必须为 adaptive。")
        if image_count > MAX_REFERENCE_IMAGES:
            raise Seedance25Error(f"参考图最多 {MAX_REFERENCE_IMAGES} 张。")
        if video_count > MAX_REFERENCE_VIDEOS:
            raise Seedance25Error(f"参考视频最多 {MAX_REFERENCE_VIDEOS} 个。")
        if audio_count > MAX_REFERENCE_AUDIOS:
            raise Seedance25Error(f"参考音频最多 {MAX_REFERENCE_AUDIOS} 个。")
        if len(references) > MAX_REFERENCE_MEDIA:
            raise Seedance25Error(f"参考媒体总数最多 {MAX_REFERENCE_MEDIA} 个。")

        reference_video_seconds = sum(
            item.duration for item in prepared if item.role == "reference_video"
        )
        if video_count:
            minimum = 2 * video_count
            if reference_video_seconds + 1e-6 < minimum:
                raise Seedance25Error(
                    f"参考视频总时长至少为 {minimum} 秒（每个至少按 2 秒计），当前为 {reference_video_seconds:.3f} 秒。"
                )
            if reference_video_seconds > 30 + 1e-6:
                raise Seedance25Error(
                    f"参考视频总时长不能超过 30 秒，当前为 {reference_video_seconds:.3f} 秒。"
                )
        return prepared, reference_video_seconds

    @staticmethod
    def validate_controls(
        prompt,
        duration,
        ratio,
        resolution,
        image_port_count,
        video_port_count,
        audio_port_count,
        max_poll_attempts,
        poll_interval,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise Seedance25Error("提示词不能为空。")
        if isinstance(duration, bool) or int(duration) != duration or not (4 <= int(duration) <= 30):
            raise Seedance25Error("时长必须是 4–30 的整数。")
        if ratio not in ALLOWED_RATIOS:
            raise Seedance25Error("比例不受支持。")
        if resolution not in ALLOWED_RESOLUTIONS:
            raise Seedance25Error("清晰度只支持 720p 或 480p。")
        limits = (
            ("参考图接口数量", image_port_count, MAX_REFERENCE_IMAGES),
            ("参考视频接口数量", video_port_count, MAX_REFERENCE_VIDEOS),
            ("参考音频接口数量", audio_port_count, MAX_REFERENCE_AUDIOS),
        )
        for label, value, maximum in limits:
            if isinstance(value, bool) or int(value) != value or not (0 <= int(value) <= maximum):
                raise Seedance25Error(f"{label} 必须是 0–{maximum} 的整数。")
        if not (1 <= int(max_poll_attempts) <= 720):
            raise Seedance25Error("最大轮询次数必须为 1–720。")
        if not (2 <= int(poll_interval) <= 60):
            raise Seedance25Error("轮询间隔必须为 2–60 秒。")

    def resolve_config(self, api_key_override):
        runtime = get_config()
        provider = normalize_provider(runtime.get("api_provider"))
        if provider not in ALLOWED_PROVIDERS:
            raise Seedance25Error(
                "Seedance 2.5 只支持灵流线路。",
                stage="route",
            )
        raw_key = str(api_key_override or runtime.get("api_key") or "").strip()
        if not raw_key:
            raise Seedance25Error("API 密钥为空，请填写节点密钥或配置全局密钥。", stage="auth")
        base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise Seedance25Error("API 地址为空。", stage="route")
        return {
            "provider": provider,
            "base_url": base_url,
            "timeout": int(runtime.get("timeout") or 300),
            "raw_key": raw_key,
            "files_key": force_channel_key(raw_key, FILES_CHANNEL),
            "generation_key": force_channel_key(raw_key, self.route.generation_channel),
        }

    @staticmethod
    def build_content(prompt, prepared, uploaded_urls):
        content = [{"type": "text", "text": prompt.strip()}]
        for asset, url in zip(prepared, uploaded_urls):
            field = f"{asset.media_type}_url"
            content.append({"type": field, "role": asset.role, field: {"url": url}})
        return content

    @staticmethod
    def _decode_response(response, stage):
        try:
            payload = response.json()
        except Exception as exc:
            preview = str(getattr(response, "text", "") or "")[:500]
            raise Seedance25Error(
                f"{stage} 响应不是 JSON：{preview}",
                stage=stage,
                status_code=getattr(response, "status_code", None),
            ) from exc
        if not isinstance(payload, dict):
            raise Seedance25Error(
                f"{stage} 响应必须是 JSON 对象。",
                stage=stage,
                status_code=getattr(response, "status_code", None),
                payload=payload,
            )
        return payload

    async def _request_json(
        self,
        method,
        url,
        config,
        *,
        stage,
        key,
        retry_network=False,
        retry_status=frozenset(),
        **kwargs,
    ):
        attempts = 3 if retry_network or retry_status else 1
        last_error = None
        extra_headers = dict(kwargs.pop("extra_headers", {}) or {})
        for attempt in range(1, attempts + 1):
            try:
                response = await asyncio.to_thread(
                    self.http.request,
                    method,
                    url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {key}",
                        **extra_headers,
                    },
                    timeout=config["timeout"],
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc
                if not retry_network or attempt == attempts:
                    suffix = "；创建结果不确定，已禁止自动重发以避免重复付费。" if stage == "create" else ""
                    raise Seedance25Error(f"{stage} 网络错误：{exc}{suffix}", stage=stage) from exc
                await self.sleep(min(2 ** (attempt - 1), 4))
                continue

            status_code = int(getattr(response, "status_code", 200) or 200)
            try:
                payload = self._decode_response(response, stage)
            except Seedance25Error as exc:
                last_error = exc
                if status_code in retry_status and attempt < attempts:
                    await self.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise
            if status_code < 400:
                return payload
            if stage == "create" and _extract_task_id(payload):
                return payload
            last_error = Seedance25Error(
                f"{stage} HTTP {status_code}",
                stage=stage,
                status_code=status_code,
                payload=payload,
            )
            if status_code not in retry_status or attempt == attempts:
                raise last_error
            await self.sleep(min(2 ** (attempt - 1), 4))
        raise last_error or Seedance25Error(f"{stage} 请求失败。", stage=stage)

    async def _upload_one(self, asset, config):
        payload = await self._request_json(
            "POST",
            build_url(config["base_url"], "/v1/files"),
            config,
            stage="upload",
            key=config["files_key"],
            retry_network=True,
            retry_status=frozenset({429, 500, 502, 503, 504}),
            files={"file": (asset.filename, asset.data, asset.content_type)},
        )
        url = _extract_string(payload, ("url", "file_url", "fileUrl"))
        if not url.startswith(("http://", "https://")):
            raise Seedance25Error(
                f"{asset.slot} 上传响应缺少有效 URL。",
                stage="upload",
                payload=payload,
            )
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
        for start in range(0, len(prepared), UPLOAD_CONCURRENCY):
            batch = prepared[start : start + UPLOAD_CONCURRENCY]
            results = await asyncio.gather(
                *(self._upload_one(asset, config) for asset in batch),
                return_exceptions=True,
            )
            for asset, result in zip(batch, results):
                if isinstance(result, Exception):
                    raise Seedance25Error(
                        str(result),
                        stage=getattr(result, "stage", "upload"),
                        status_code=getattr(result, "status_code", None),
                        payload=getattr(result, "payload", None),
                        details={
                            "failed_slot": asset.slot,
                            "completed_uploads": list(summaries),
                        },
                    ) from result
                url, summary = result
                urls.append(url)
                summaries.append(summary)
        return urls, summaries

    def _json(self, payload, secrets):
        safe = mask_provider_payload(_redact(payload, secrets))
        return json.dumps(safe, ensure_ascii=False, indent=2)

    async def _last_frame(self, payload, video_url):
        last_frame_url = _extract_last_frame_url(payload)
        if last_frame_url:
            try:
                image = await asyncio.to_thread(load_image_from_candidate, last_frame_url, 120)
                if image is not None:
                    return pil2tensor(image), last_frame_url, "last_frame_url"
            except Exception:
                pass
        image = await asyncio.to_thread(_extract_last_frame_from_video, video_url)
        if image is not None:
            return image, last_frame_url, "video_last_frame_extracted"
        return blank_image_tensor(width=1, height=1, color="black"), last_frame_url, "unavailable"

    async def execute(
        self,
        *,
        prompt,
        duration,
        ratio,
        resolution,
        image_port_count,
        video_port_count,
        audio_port_count,
        raw_assets,
        api_key="",
        wait_for_completion=True,
        max_poll_attempts=180,
        poll_interval=10,
        return_last_frame=False,
        notify=None,
    ):
        blank_last_frame = blank_image_tensor(width=1, height=1, color="black")
        task_id = ""
        secrets = [api_key]
        envelope = {"node": self.route.node_id, "model": self.route.model}
        try:
            self.validate_controls(
                prompt,
                duration,
                ratio,
                resolution,
                image_port_count,
                video_port_count,
                audio_port_count,
                max_poll_attempts,
                poll_interval,
            )
            config = self.resolve_config(api_key)
            secrets.extend([config["raw_key"], config["files_key"], config["generation_key"]])
            prepared, reference_video_seconds = await asyncio.to_thread(
                self.validate_and_prepare,
                raw_assets,
                ratio,
            )
            uploaded_urls, upload_summaries = await self.upload_all(prepared, config)
            content = self.build_content(prompt, prepared, uploaded_urls)
            body = self.route.build_payload(
                prompt=prompt,
                content=content,
                duration=duration,
                ratio=ratio,
                resolution=resolution,
                reference_video_seconds=reference_video_seconds,
            )
            envelope.update(
                {
                    "provider": config["provider"],
                    "request": body,
                    "uploads": upload_summaries,
                }
            )

            create_payload = await self._request_json(
                "POST",
                build_url(config["base_url"], self.route.create_path),
                config,
                stage="create",
                key=config["generation_key"],
                extra_headers={"Content-Type": "application/json"},
                json=body,
            )
            envelope["create_response"] = create_payload
            task_id = _extract_task_id(create_payload)
            if not task_id:
                raise Seedance25Error(
                    "创建响应缺少任务 ID。",
                    stage="create",
                    payload=create_payload,
                )
            envelope["task_id"] = task_id
            if notify:
                notify(task_id, "submitted")
            if not wait_for_completion:
                envelope["status"] = "submitted"
                return Seedance25Result(
                    LingliuNullVideoAdapter("任务已提交，未等待完成。"),
                    task_id,
                    self._json(envelope, secrets),
                    "",
                    blank_last_frame,
                )

            final_payload = None
            for attempt in range(int(max_poll_attempts)):
                query_payload = await self._request_json(
                    "GET",
                    build_url(config["base_url"], self.route.query_path(task_id)),
                    config,
                    stage="query",
                    key=config["generation_key"],
                    retry_network=True,
                    retry_status=frozenset({429, 500, 502, 503, 504}),
                )
                final_payload = query_payload
                envelope["query_response"] = query_payload
                status = _extract_status(query_payload)
                if notify:
                    notify(task_id, status or "unknown")
                if status in {"submitted", "queued", "pending", "running", "processing"}:
                    if attempt + 1 < int(max_poll_attempts):
                        await self.sleep(int(poll_interval))
                    continue
                if status in {"failed", "cancelled", "expired"}:
                    reason = _extract_string(query_payload, ("message", "error", "fail_reason", "failReason"))
                    raise Seedance25Error(
                        reason or f"任务终态为 {status}。",
                        stage="query",
                        payload=query_payload,
                    )
                if status == "succeeded":
                    video_url = _extract_video_url(query_payload)
                    if not video_url:
                        raise Seedance25Error(
                            "任务成功，但响应中没有视频 URL。",
                            stage="query",
                            payload=query_payload,
                        )
                    envelope["status"] = "succeeded"
                    last_frame = blank_last_frame
                    if return_last_frame:
                        last_frame, last_frame_url, source = await self._last_frame(query_payload, video_url)
                        envelope["last_frame"] = {"url": last_frame_url, "source": source}
                    return Seedance25Result(
                        LingliuVideoAdapter(video_url, mime_type="video/mp4"),
                        task_id,
                        self._json(envelope, secrets),
                        video_url,
                        last_frame,
                    )
                raise Seedance25Error(
                    f"查询返回未知状态：{status or '<empty>'}。",
                    stage="query",
                    payload=query_payload,
                )

            envelope["status"] = "timeout"
            envelope["query_response"] = final_payload
            envelope["error"] = {"stage": "query", "message": "轮询超时。"}
            return Seedance25Result(
                LingliuNullVideoAdapter("轮询超时。"),
                task_id,
                self._json(envelope, secrets),
                "",
                blank_last_frame,
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
            if notify:
                notify(task_id, "error")
            message = _redact_text(str(exc), secrets)
            return Seedance25Result(
                LingliuNullVideoAdapter(message),
                task_id,
                self._json(error_payload, secrets),
                "",
                blank_last_frame,
            )


class _LingliuSeedance25NodeBase(LingliuBaseNode):
    CATEGORY = "Lingliu/Seedance 2.5"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL", "尾帧图")
    FUNCTION = "generate_video"
    ROUTE = None

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A queued API generation is an intentional new side effect even when
        # all visible inputs match the previous run.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "API密钥": (
                "STRING",
                {
                    "default": "",
                    "password": True,
                    "tooltip": "留空使用全局密钥；节点内密钥会随工作流保存，不建议用于共享工作流。",
                },
            ),
            "返回尾帧图": ("BOOLEAN", {"default": False}),
            "等待完成": ("BOOLEAN", {"default": True}),
            "最大轮询次数": ("INT", {"default": 180, "min": 1, "max": 720, "step": 1}),
            "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
            "首帧图": ("IMAGE",),
            "尾帧图": ("IMAGE",),
        }
        optional.update({f"参考图{index}": ("IMAGE",) for index in range(1, MAX_REFERENCE_IMAGES + 1)})
        optional.update(
            {
                f"参考视频{index}": (IO.VIDEO, {"tooltip": "单个插槽连接一个参考视频。"})
                for index in range(1, MAX_REFERENCE_VIDEOS + 1)
            }
        )
        optional.update(
            {
                f"参考音频{index}": (IO.AUDIO, {"tooltip": "单个插槽连接一个参考音频。"})
                for index in range(1, MAX_REFERENCE_AUDIOS + 1)
            }
        )
        return {
            "required": {
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "时长": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1}),
                "比例": (list(ALLOWED_RATIOS), {"default": "adaptive"}),
                "清晰度": (list(ALLOWED_RESOLUTIONS), {"default": "720p"}),
                "参考图接口数量": (
                    "INT",
                    {"default": 3, "min": 0, "max": MAX_REFERENCE_IMAGES, "step": 1},
                ),
                "参考视频接口数量": (
                    "INT",
                    {"default": 1, "min": 0, "max": MAX_REFERENCE_VIDEOS, "step": 1},
                ),
                "参考音频接口数量": (
                    "INT",
                    {"default": 1, "min": 0, "max": MAX_REFERENCE_AUDIOS, "step": 1},
                ),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def _notify(self, unique_id):
        def notify(task_id, status):
            if task_id:
                self.publish_task_id(unique_id, task_id, append=False)
            self.publish_live_message(
                unique_id,
                f"Task ID: {task_id}\n状态: {status}",
                append=False,
            )

        return notify

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
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)
        result = await Seedance25Workflow(self.ROUTE).execute(
            prompt=kwargs.get("提示词", ""),
            duration=kwargs.get("时长", 5),
            ratio=kwargs.get("比例", "adaptive"),
            resolution=kwargs.get("清晰度", "720p"),
            image_port_count=kwargs.get("参考图接口数量", 3),
            video_port_count=kwargs.get("参考视频接口数量", 1),
            audio_port_count=kwargs.get("参考音频接口数量", 1),
            raw_assets=raw_assets,
            api_key=kwargs.get("API密钥", ""),
            wait_for_completion=kwargs.get("等待完成", True),
            max_poll_attempts=kwargs.get("最大轮询次数", 180),
            poll_interval=kwargs.get("轮询间隔", 10),
            return_last_frame=kwargs.get("返回尾帧图", False),
            notify=self._notify(unique_id),
        )
        return result.video, result.task_id, result.response, result.video_url, result.last_frame


class LingliuSeedance25(_LingliuSeedance25NodeBase):
    ROUTE = STANDARD_ROUTE


class LingliuSeedance25Official(_LingliuSeedance25NodeBase):
    ROUTE = OFFICIAL_ROUTE
