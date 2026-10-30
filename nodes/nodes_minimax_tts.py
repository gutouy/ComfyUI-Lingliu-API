import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote, urlparse

import requests
import torch
from comfy.comfy_types import IO

from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    get_config,
    normalize_provider,
)
from .nodes_openai_compatible import LingliuBaseNode

try:
    from .minimax_tts_voices import resolve_voice_id
except ImportError:  # The fallback keeps the backend usable without a catalog snapshot.
    def resolve_voice_id(value):
        return str(value or "").strip()


NODE_ID = "lingliu_minimax_tts"
OPTIONS_NODE_ID = "lingliu_minimax_tts_options"
OPTIONS_TYPE = "LINGLIU_MINIMAX_TTS_OPTIONS"

MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
)
MODELS_WITH_VOICE_MODIFY = frozenset({"speech-2.8-hd", "speech-2.8-turbo"})

LANGUAGE_UNSPECIFIED = "不指定"
LANGUAGE_BOOST_VALUES = (
    "auto",
    "Chinese",
    "Chinese,Yue",
    "English",
    "Arabic",
    "Russian",
    "Spanish",
    "French",
    "Portuguese",
    "German",
    "Turkish",
    "Dutch",
    "Ukrainian",
    "Vietnamese",
    "Indonesian",
    "Japanese",
    "Italian",
    "Korean",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
    "Bulgarian",
    "Danish",
    "Hebrew",
    "Malay",
    "Slovak",
    "Swedish",
    "Croatian",
    "Hungarian",
    "Norwegian",
    "Slovenian",
    "Catalan",
    "Nynorsk",
    "Afrikaans",
)
LANGUAGE_CHOICES = (LANGUAGE_UNSPECIFIED,) + LANGUAGE_BOOST_VALUES

EMOTION_UNSPECIFIED = "不指定"
EMOTIONS = (
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "neutral",
)
EMOTION_CHOICES = (EMOTION_UNSPECIFIED,) + EMOTIONS

SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100)
BITRATES = (32000, 64000, 128000, 256000)
AUDIO_FORMATS = ("mp3", "flac")
CHANNELS = (1, 2)

NORMALIZATION_DEFAULT = "使用上游默认"
NORMALIZATION_ENABLED = "开启"
NORMALIZATION_DISABLED = "关闭"
NORMALIZATION_MODES = (
    NORMALIZATION_DEFAULT,
    NORMALIZATION_ENABLED,
    NORMALIZATION_DISABLED,
)
VOICE_MODIFY_OMIT = "不发送"
VOICE_MODIFY_SEND = "发送"
VOICE_MODIFY_MODES = (VOICE_MODIFY_OMIT, VOICE_MODIFY_SEND)

ALLOWED_PROVIDERS = frozenset({LINGLIU_PROVIDER})
RETRYABLE_GET_STATUS = frozenset({429, 500, 502, 503, 504})
PENDING_STATUSES = frozenset({"IN_QUEUE", "IN_PROGRESS"})
TERMINAL_FAILURE_STATUSES = frozenset({"FAILED", "CANCELLED", "CANCELED"})
MAX_PROMPT_CHARACTERS = 10_000
MAX_AUDIO_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_POLL_ATTEMPTS = 600
DEFAULT_POLL_INTERVAL = 1.0


class MiniMaxTTSError(RuntimeError):
    """A public-safe error with a structured, redacted execution envelope."""

    def __init__(
        self,
        message,
        *,
        stage="validation",
        status_code=None,
        payload=None,
        details=None,
        task_id="",
        response_json="",
    ):
        self.message = str(message or "MiniMax TTS 执行失败。")
        self.stage = str(stage or "internal")
        self.status_code = status_code
        self.payload = payload
        self.details = details or {}
        self.task_id = str(task_id or "")
        self.response_json = str(response_json or "")
        super().__init__(self.message)

    def __str__(self):
        if self.response_json:
            return f"{self.message}\n{self.response_json}"
        return self.message


@dataclass(frozen=True)
class MiniMaxTTSOptionsSpec:
    english_normalization: bool = False
    sample_rate: int = 32000
    bitrate: int = 128000
    audio_format: str = "mp3"
    channel: int = 1
    pronunciation_rules: tuple = ()
    normalization_mode: str = NORMALIZATION_DEFAULT
    target_loudness: float = -18.0
    target_range: float = 8.0
    target_peak: float = -0.5
    voice_modify_mode: str = VOICE_MODIFY_OMIT
    voice_modify_pitch: int = 0
    voice_modify_intensity: int = 0
    voice_modify_timbre: int = 0


@dataclass(frozen=True)
class MiniMaxTTSConfig:
    provider: str
    base_url: str
    api_key: str
    timeout: int


@dataclass(frozen=True)
class MiniMaxTTSResult:
    audio: dict
    task_id: str
    response: str
    audio_url: str
    duration_ms: int


class RequestsHTTPAdapter:
    """Production adapter for the remote-owned Lingliu Fal queue seam."""

    def request(self, method, url, **kwargs):
        return requests.request(method, url, **kwargs)


class PyAVAudioDecoder:
    """Decode a downloaded audio file into ComfyUI's native AUDIO contract."""

    @staticmethod
    def _float32_pcm(waveform):
        if waveform.dtype.is_floating_point:
            return waveform.to(torch.float32)
        if waveform.dtype == torch.int16:
            return waveform.to(torch.float32) / float(2**15)
        if waveform.dtype == torch.int32:
            return waveform.to(torch.float32) / float(2**31)
        if waveform.dtype == torch.int8:
            return waveform.to(torch.float32) / float(2**7)
        if waveform.dtype == torch.uint8:
            return (waveform.to(torch.float32) - 128.0) / 128.0
        raise MiniMaxTTSError(
            f"不支持的音频采样格式：{waveform.dtype}",
            stage="decode",
        )

    def decode_file(self, path):
        try:
            import av
        except ImportError as exc:
            raise MiniMaxTTSError(
                "当前 ComfyUI 环境缺少 PyAV，无法解码生成的音频。",
                stage="decode",
            ) from exc

        try:
            with av.open(path) as container:
                if not container.streams.audio:
                    raise MiniMaxTTSError("结果文件不包含音频流。", stage="decode")
                stream = container.streams.audio[0]
                channels = int(getattr(stream, "channels", 0) or 0)
                sample_rate = int(
                    getattr(getattr(stream, "codec_context", None), "sample_rate", 0) or 0
                )
                frames = []
                for frame in container.decode(streams=stream.index):
                    array = frame.to_ndarray()
                    tensor = torch.from_numpy(array)
                    frame_channels = int(getattr(frame.layout, "nb_channels", 0) or channels or 1)
                    if tensor.ndim == 1:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.shape[0] != frame_channels:
                        tensor = tensor.reshape(-1, frame_channels).transpose(0, 1)
                    frames.append(self._float32_pcm(tensor))
                    sample_rate = int(getattr(frame, "sample_rate", 0) or sample_rate)
                if not frames:
                    raise MiniMaxTTSError("结果文件没有可解码的音频帧。", stage="decode")
                waveform = torch.cat(frames, dim=1).contiguous().clamp_(-1.0, 1.0)
                if sample_rate <= 0:
                    raise MiniMaxTTSError("结果音频缺少有效采样率。", stage="decode")
                return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except MiniMaxTTSError:
            raise
        except Exception as exc:
            raise MiniMaxTTSError(f"音频解码失败：{exc}", stage="decode") from exc


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_number(value, name, minimum, maximum, *, integer=False):
    if not _is_number(value):
        raise MiniMaxTTSError(f"{name} 必须是数字。")
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        raise MiniMaxTTSError(f"{name} 必须在 {minimum}～{maximum} 之间。")
    if integer and int(numeric) != numeric:
        raise MiniMaxTTSError(f"{name} 必须是整数。")
    return int(numeric) if integer else numeric


def _pronunciation_rules(value):
    if value is None:
        return ()
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, (list, tuple)):
        lines = value
    else:
        raise MiniMaxTTSError("发音词典必须是多行文本或字符串列表。")
    return tuple(str(line).strip() for line in lines if str(line).strip())


def _coerce_options(options):
    if options is None:
        return None
    if isinstance(options, MiniMaxTTSOptionsSpec):
        return options
    if isinstance(options, dict):
        allowed = set(MiniMaxTTSOptionsSpec.__dataclass_fields__)
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise MiniMaxTTSError(f"高级设置包含未知字段：{', '.join(unknown)}")
        values = dict(options)
        if "pronunciation_rules" in values:
            values["pronunciation_rules"] = _pronunciation_rules(values["pronunciation_rules"])
        try:
            return MiniMaxTTSOptionsSpec(**values)
        except TypeError as exc:
            raise MiniMaxTTSError(f"高级设置无效：{exc}") from exc
    raise MiniMaxTTSError("高级设置类型无效。")


def validate_options(options):
    options = _coerce_options(options)
    if options is None:
        return None
    if not isinstance(options.english_normalization, bool):
        raise MiniMaxTTSError("英文标准化必须是布尔值。")
    if options.sample_rate not in SAMPLE_RATES:
        raise MiniMaxTTSError("采样率不受支持。")
    if options.bitrate not in BITRATES:
        raise MiniMaxTTSError("比特率不受支持。")
    if options.audio_format not in AUDIO_FORMATS:
        raise MiniMaxTTSError("音频格式不受支持。")
    if options.channel not in CHANNELS:
        raise MiniMaxTTSError("声道数只能是 1 或 2。")
    if options.normalization_mode not in NORMALIZATION_MODES:
        raise MiniMaxTTSError("响度标准化模式无效。")
    _validate_number(options.target_loudness, "目标响度", -70, -10)
    _validate_number(options.target_range, "目标响度范围", 0, 20)
    _validate_number(options.target_peak, "目标峰值", -3, 0)
    if options.voice_modify_mode not in VOICE_MODIFY_MODES:
        raise MiniMaxTTSError("二次音色修改模式无效。")
    _validate_number(options.voice_modify_pitch, "二次音调", -100, 100, integer=True)
    _validate_number(options.voice_modify_intensity, "声音强度", -100, 100, integer=True)
    _validate_number(options.voice_modify_timbre, "音色调整", -100, 100, integer=True)
    return options


def build_speech_payload(
    *,
    text,
    model,
    language_boost,
    voice_id,
    speed,
    volume,
    pitch,
    emotion,
    options=None,
):
    """Validate node inputs and map them to the Fal MiniMax request body."""

    if not isinstance(text, str) or not text.strip():
        raise MiniMaxTTSError("文本不能为空。")
    if len(text) > MAX_PROMPT_CHARACTERS:
        raise MiniMaxTTSError(f"文本不能超过 {MAX_PROMPT_CHARACTERS} 个字符。")
    if model not in MODELS:
        raise MiniMaxTTSError("模型不受支持。")
    if language_boost not in LANGUAGE_CHOICES:
        raise MiniMaxTTSError("语种增强值不受支持。")

    resolved_voice_id = str(resolve_voice_id(voice_id) or "").strip()
    if not resolved_voice_id:
        raise MiniMaxTTSError("音色ID不能为空。")
    speed = _validate_number(speed, "语速", 0.5, 2.0)
    volume = _validate_number(volume, "音量", 0.01, 10.0)
    pitch = _validate_number(pitch, "基础音调", -12, 12, integer=True)
    if emotion not in EMOTION_CHOICES:
        raise MiniMaxTTSError("情绪值不受支持。")

    voice_setting = {
        "voice_id": resolved_voice_id,
        "speed": speed,
        "vol": volume,
        "pitch": pitch,
    }
    if emotion != EMOTION_UNSPECIFIED:
        voice_setting["emotion"] = emotion

    payload = {
        "prompt": text,
        "voice_setting": voice_setting,
        "output_format": "url",
    }
    if language_boost != LANGUAGE_UNSPECIFIED:
        payload["language_boost"] = language_boost

    options = validate_options(options)
    if options is None:
        return payload

    voice_setting["english_normalization"] = options.english_normalization
    audio_setting = {
        "sample_rate": int(options.sample_rate),
        "format": options.audio_format,
        "channel": int(options.channel),
    }
    if options.audio_format == "mp3":
        audio_setting["bitrate"] = int(options.bitrate)
    payload["audio_setting"] = audio_setting

    if options.pronunciation_rules:
        payload["pronunciation_dict"] = {"tone_list": list(options.pronunciation_rules)}

    if options.normalization_mode != NORMALIZATION_DEFAULT:
        normalization = {"enabled": options.normalization_mode == NORMALIZATION_ENABLED}
        if normalization["enabled"]:
            normalization.update(
                {
                    "target_loudness": float(options.target_loudness),
                    "target_range": float(options.target_range),
                    "target_peak": float(options.target_peak),
                }
            )
        payload["normalization_setting"] = normalization

    if options.voice_modify_mode == VOICE_MODIFY_SEND:
        if model not in MODELS_WITH_VOICE_MODIFY:
            raise MiniMaxTTSError(
                "speech-2.6 模型不支持二次音色修改；请改用 speech-2.8 或设为不发送。"
            )
        payload["voice_modify"] = {
            "pitch": int(options.voice_modify_pitch),
            "intensity": int(options.voice_modify_intensity),
            "timbre": int(options.voice_modify_timbre),
        }
    return payload


def _redact_text(value, secrets, *, signed_urls=False):
    text = str(value or "")
    for secret in sorted({str(item) for item in secrets if item}, key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    text = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer <redacted>", text)
    text = re.sub(r"(?i)sk-[A-Za-z0-9_.:-]+(?:-\d+)?", "sk-<redacted>", text)
    if signed_urls:
        text = re.sub(
            r"(https?://[^\s\"'<>?]+)\?[^\s\"'<>]+",
            r"\1?<redacted>",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _redact(value, secrets, *, signed_urls=False):
    if isinstance(value, dict):
        return {
            str(key): _redact(item, secrets, signed_urls=signed_urls)
            for key, item in value.items()
            if str(key).lower() not in {"authorization", "api_key", "apikey", "key"}
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets, signed_urls=signed_urls) for item in value]
    if isinstance(value, str):
        return _redact_text(value, secrets, signed_urls=signed_urls)
    return value


def _extract_task_id(payload):
    if not isinstance(payload, dict):
        return ""
    for container in (payload, payload.get("data")):
        if not isinstance(container, dict):
            continue
        for key in ("request_id", "task_id", "id"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_status(payload):
    if not isinstance(payload, dict):
        return ""
    for container in (payload, payload.get("data")):
        if not isinstance(container, dict):
            continue
        for key in ("status", "state"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper().replace("-", "_").replace(" ", "_")
    return ""


def _extract_failure_reason(payload):
    if not isinstance(payload, dict):
        return ""
    for container in (payload, payload.get("error"), payload.get("data")):
        if isinstance(container, str) and container.strip():
            return container.strip()
        if not isinstance(container, dict):
            continue
        for key in ("message", "error", "detail", "fail_reason", "reason"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _extract_audio_result(payload):
    if not isinstance(payload, dict):
        raise MiniMaxTTSError("结果响应必须是 JSON 对象。", stage="result", payload=payload)
    audio = payload.get("audio")
    if not isinstance(audio, dict):
        raise MiniMaxTTSError("结果响应缺少 audio 对象。", stage="result", payload=payload)
    url = audio.get("url")
    if not isinstance(url, str) or not url.strip():
        raise MiniMaxTTSError("结果响应缺少 audio.url。", stage="result", payload=payload)
    duration = payload.get("duration_ms")
    duration_ms = int(duration) if _is_number(duration) and float(duration) >= 0 else 0
    return url.strip(), duration_ms, audio


def _validate_audio_url(url):
    try:
        parsed = urlparse(str(url or ""))
    except Exception as exc:
        raise MiniMaxTTSError("结果音频 URL 无效。", stage="result") from exc
    hostname = str(parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (hostname != "fal.media" and not hostname.endswith(".fal.media"))
    ):
        raise MiniMaxTTSError(
            "结果音频 URL 未通过 Fal 媒体域名校验。",
            stage="result",
            payload={"audio_url": url},
        )
    return parsed


def _audio_suffix(url, audio_metadata, content_type=""):
    file_name = str((audio_metadata or {}).get("file_name") or "").strip()
    candidates = (file_name, PurePosixPath(urlparse(url).path).name)
    allowed = {".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a", ".aac"}
    for candidate in candidates:
        suffix = os.path.splitext(candidate)[1].lower()
        if suffix in allowed:
            return suffix
    mime_suffixes = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/ogg": ".ogg",
    }
    return mime_suffixes.get(str(content_type or "").split(";", 1)[0].strip().lower(), ".audio")


class MiniMaxTTSWorkflow:
    """Deep module hiding routing, one-shot creation, polling, download and decode."""

    def __init__(
        self,
        *,
        http=None,
        decoder=None,
        sleep=None,
        max_poll_attempts=DEFAULT_MAX_POLL_ATTEMPTS,
        poll_interval=DEFAULT_POLL_INTERVAL,
    ):
        self.http = http or RequestsHTTPAdapter()
        self.decoder = decoder or PyAVAudioDecoder()
        self.sleep = sleep or asyncio.sleep
        self.max_poll_attempts = int(max_poll_attempts)
        self.poll_interval = float(poll_interval)
        if self.max_poll_attempts < 1:
            raise ValueError("max_poll_attempts must be positive")
        if self.poll_interval < 0:
            raise ValueError("poll_interval must not be negative")

    @staticmethod
    def resolve_config():
        runtime = get_config()
        provider = normalize_provider(runtime.get("api_provider"))
        if provider not in ALLOWED_PROVIDERS:
            raise MiniMaxTTSError(
                "MiniMax TTS 仅支持灵流线路。",
                stage="route",
            )
        api_key = str(runtime.get("api_key") or "").strip()
        if not api_key:
            raise MiniMaxTTSError(
                "全局 API 密钥为空，请先在 lingliu_api_set 中配置。",
                stage="auth",
            )
        base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise MiniMaxTTSError("全局 API 地址为空。", stage="route")
        timeout = int(runtime.get("timeout") or 300)
        if timeout < 1:
            raise MiniMaxTTSError("全局请求超时必须大于 0 秒。", stage="route")
        return MiniMaxTTSConfig(provider, base_url, api_key, timeout)

    @staticmethod
    def create_url(config, model):
        return build_url(config.base_url, f"/fal/fal-ai/minimax/{model}")

    @staticmethod
    def status_url(create_url, task_id):
        return f"{create_url.rstrip('/')}/requests/{quote(task_id, safe='')}/status"

    @staticmethod
    def result_url(create_url, task_id):
        return f"{create_url.rstrip('/')}/requests/{quote(task_id, safe='')}"

    @staticmethod
    def _decode_json(response, stage):
        try:
            payload = response.json()
        except Exception as exc:
            preview = str(getattr(response, "text", "") or "")[:500]
            raise MiniMaxTTSError(
                f"{stage} 响应不是 JSON：{preview}",
                stage=stage,
                status_code=getattr(response, "status_code", None),
            ) from exc
        if not isinstance(payload, dict):
            raise MiniMaxTTSError(
                f"{stage} 响应必须是 JSON 对象。",
                stage=stage,
                status_code=getattr(response, "status_code", None),
                payload=payload,
            )
        return payload

    async def _request_json(self, method, url, config, *, stage, retry_get=False, **kwargs):
        if method.upper() != "GET" and retry_get:
            raise ValueError("Only GET requests may be retried")
        attempts = 3 if retry_get else 1
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                response = await asyncio.to_thread(
                    self.http.request,
                    method,
                    url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {config.api_key}",
                        **({"Content-Type": "application/json"} if method.upper() == "POST" else {}),
                    },
                    timeout=config.timeout,
                    **kwargs,
                )
            except Exception as exc:
                last_error = MiniMaxTTSError(
                    (
                        f"{stage} 网络错误：{exc}；创建结果不确定，已禁止自动重发以避免重复付费。"
                        if method.upper() == "POST"
                        else f"{stage} 网络错误：{exc}"
                    ),
                    stage=stage,
                )
                if attempt == attempts:
                    raise last_error from exc
                await self.sleep(min(2 ** (attempt - 1), 4))
                continue

            status_code = int(getattr(response, "status_code", 200) or 200)
            try:
                payload = self._decode_json(response, stage)
            except MiniMaxTTSError as exc:
                last_error = exc
                if retry_get and status_code in RETRYABLE_GET_STATUS and attempt < attempts:
                    await self.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise
            if 200 <= status_code < 300:
                return payload
            last_error = MiniMaxTTSError(
                f"{stage} HTTP {status_code}",
                stage=stage,
                status_code=status_code,
                payload=payload,
            )
            if not retry_get or status_code not in RETRYABLE_GET_STATUS or attempt == attempts:
                raise last_error
            await self.sleep(min(2 ** (attempt - 1), 4))
        raise last_error or MiniMaxTTSError(f"{stage} 请求失败。", stage=stage)

    def _download_once(self, url, config, audio_metadata):
        _validate_audio_url(url)
        response = None
        temp_path = ""
        try:
            response = self.http.request(
                "GET",
                url,
                headers={"Accept": "audio/*, application/octet-stream"},
                timeout=config.timeout,
                stream=True,
            )
            status_code = int(getattr(response, "status_code", 200) or 200)
            if not 200 <= status_code < 300:
                payload = None
                try:
                    payload = response.json()
                except Exception:
                    payload = str(getattr(response, "text", "") or "")[:500]
                raise MiniMaxTTSError(
                    f"download HTTP {status_code}",
                    stage="download",
                    status_code=status_code,
                    payload=payload,
                )

            headers = getattr(response, "headers", {}) or {}
            content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type and not (
                media_type.startswith("audio/")
                or media_type in {"application/octet-stream", "binary/octet-stream"}
            ):
                raise MiniMaxTTSError(
                    f"下载响应不是音频类型：{content_type}",
                    stage="download",
                )
            content_length = str(headers.get("Content-Length") or headers.get("content-length") or "").strip()
            if content_length.isdigit() and int(content_length) > MAX_AUDIO_BYTES:
                raise MiniMaxTTSError(
                    f"结果音频超过 {MAX_AUDIO_BYTES // (1024 * 1024)} MiB 下载上限。",
                    stage="download",
                )

            suffix = _audio_suffix(url, audio_metadata, content_type)
            with tempfile.NamedTemporaryFile(prefix="lingliu_minimax_tts_", suffix=suffix, delete=False) as handle:
                temp_path = handle.name
                total = 0
                iterator = getattr(response, "iter_content", None)
                chunks = iterator(chunk_size=1024 * 1024) if callable(iterator) else (getattr(response, "content", b""),)
                for chunk in chunks:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise MiniMaxTTSError(
                            f"结果音频超过 {MAX_AUDIO_BYTES // (1024 * 1024)} MiB 下载上限。",
                            stage="download",
                        )
                    handle.write(chunk)
            if total == 0:
                raise MiniMaxTTSError("下载到的音频文件为空。", stage="download")
            return temp_path
        except Exception:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    async def _download_and_decode(self, url, config, audio_metadata):
        last_error = None
        temp_path = ""
        for attempt in range(1, 4):
            try:
                temp_path = await asyncio.to_thread(
                    self._download_once,
                    url,
                    config,
                    audio_metadata,
                )
                break
            except MiniMaxTTSError as exc:
                last_error = exc
                if exc.status_code not in RETRYABLE_GET_STATUS or attempt == 3:
                    raise
            except (requests.exceptions.RequestException, OSError) as exc:
                last_error = MiniMaxTTSError(f"download 网络错误：{exc}", stage="download")
                if attempt == 3:
                    raise last_error from exc
            await self.sleep(min(2 ** (attempt - 1), 4))
        if not temp_path:
            raise last_error or MiniMaxTTSError("音频下载失败。", stage="download")
        try:
            audio = await asyncio.to_thread(self.decoder.decode_file, temp_path)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        self._validate_audio_contract(audio)
        return audio

    @staticmethod
    def _validate_audio_contract(audio):
        if not isinstance(audio, dict):
            raise MiniMaxTTSError("音频解码器没有返回 AUDIO 对象。", stage="decode")
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise MiniMaxTTSError("AUDIO waveform 必须是 [B,C,T] 三维张量。", stage="decode")
        if waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or waveform.shape[2] < 1:
            raise MiniMaxTTSError("AUDIO waveform 的批次、声道或长度无效。", stage="decode")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise MiniMaxTTSError("AUDIO sample_rate 无效。", stage="decode")

    @staticmethod
    def _duration_from_audio(audio):
        return int(round(audio["waveform"].shape[-1] * 1000.0 / audio["sample_rate"]))

    @staticmethod
    def _json(payload, secrets, *, error=False):
        safe = _redact(payload, secrets, signed_urls=error)
        return json.dumps(safe, ensure_ascii=False, indent=2)

    @staticmethod
    def _notify(notify, task_id, status):
        if not notify:
            return
        try:
            notify(task_id, status)
        except Exception:
            pass

    async def execute(
        self,
        *,
        text,
        model,
        language_boost,
        voice_id,
        speed,
        volume,
        pitch,
        emotion,
        options=None,
        notify=None,
    ):
        task_id = ""
        secrets = []
        envelope = {"node": NODE_ID, "model": model}
        try:
            body = build_speech_payload(
                text=text,
                model=model,
                language_boost=language_boost,
                voice_id=voice_id,
                speed=speed,
                volume=volume,
                pitch=pitch,
                emotion=emotion,
                options=options,
            )
            config = self.resolve_config()
            secrets.append(config.api_key)
            create_url = self.create_url(config, model)
            envelope.update(
                {
                    "provider": config.provider,
                    "request": body,
                }
            )
            create_payload = await self._request_json(
                "POST",
                create_url,
                config,
                stage="create",
                json=body,
            )
            envelope["create_response"] = create_payload
            task_id = _extract_task_id(create_payload)
            if not task_id:
                raise MiniMaxTTSError(
                    "创建响应缺少 request_id。",
                    stage="create",
                    payload=create_payload,
                )
            envelope["task_id"] = task_id
            initial_status = _extract_status(create_payload) or "SUBMITTED"
            self._notify(notify, task_id, initial_status)

            status_endpoint = self.status_url(create_url, task_id)
            final_status_payload = None
            for attempt in range(self.max_poll_attempts):
                status_payload = await self._request_json(
                    "GET",
                    status_endpoint,
                    config,
                    stage="status",
                    retry_get=True,
                )
                final_status_payload = status_payload
                envelope["status_response"] = status_payload
                status = _extract_status(status_payload)
                self._notify(notify, task_id, status or "UNKNOWN")
                if status in PENDING_STATUSES:
                    if attempt + 1 < self.max_poll_attempts:
                        await self.sleep(self.poll_interval)
                    continue
                if status in TERMINAL_FAILURE_STATUSES:
                    reason = _extract_failure_reason(status_payload)
                    raise MiniMaxTTSError(
                        reason or f"任务终态为 {status}。",
                        stage="status",
                        payload=status_payload,
                        task_id=task_id,
                    )
                if status == "COMPLETED":
                    break
                raise MiniMaxTTSError(
                    f"状态接口返回未知状态：{status or '<empty>'}。",
                    stage="status",
                    payload=status_payload,
                    task_id=task_id,
                )
            else:
                raise MiniMaxTTSError(
                    "等待 MiniMax TTS 任务完成超时。",
                    stage="status",
                    payload=final_status_payload,
                    task_id=task_id,
                )

            result_payload = await self._request_json(
                "GET",
                self.result_url(create_url, task_id),
                config,
                stage="result",
                retry_get=True,
            )
            envelope["result_response"] = result_payload
            audio_url, upstream_duration_ms, audio_metadata = _extract_audio_result(result_payload)
            _validate_audio_url(audio_url)
            self._notify(notify, task_id, "DOWNLOADING")
            audio = await self._download_and_decode(audio_url, config, audio_metadata)
            duration_ms = upstream_duration_ms or self._duration_from_audio(audio)
            envelope.update(
                {
                    "status": "succeeded",
                    "audio_url": audio_url,
                    "duration_ms": duration_ms,
                }
            )
            self._notify(notify, task_id, "SUCCEEDED")
            return MiniMaxTTSResult(
                audio=audio,
                task_id=task_id,
                response=self._json(envelope, secrets),
                audio_url=audio_url,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            if isinstance(exc, MiniMaxTTSError):
                message = exc.message
                stage = exc.stage
                status_code = exc.status_code
                upstream = exc.payload
                details = exc.details
                task_id = task_id or exc.task_id
            else:
                message = str(exc) or exc.__class__.__name__
                stage = "internal"
                status_code = None
                upstream = None
                details = {"type": exc.__class__.__name__}
            safe_message = _redact_text(message, secrets, signed_urls=True)
            error_envelope = {
                **envelope,
                "status": "error",
                "task_id": task_id,
                "error": {
                    "stage": stage,
                    "message": safe_message,
                    "status_code": status_code,
                    "upstream": upstream,
                    "details": details or None,
                },
            }
            response_json = self._json(error_envelope, secrets, error=True)
            self._notify(notify, task_id, "ERROR")
            raise MiniMaxTTSError(
                safe_message,
                stage=stage,
                status_code=status_code,
                task_id=task_id,
                response_json=response_json,
            ) from exc


class LingliuMiniMaxTTSOptions:
    CATEGORY = "Lingliu/MiniMax TTS"
    RETURN_TYPES = (OPTIONS_TYPE,)
    RETURN_NAMES = ("高级设置",)
    FUNCTION = "build_options"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "英文标准化": ("BOOLEAN", {"default": False}),
                "采样率": (list(SAMPLE_RATES), {"default": 32000}),
                "比特率": (list(BITRATES), {"default": 128000}),
                "音频格式": (list(AUDIO_FORMATS), {"default": "mp3"}),
                "声道": (list(CHANNELS), {"default": 1}),
                "发音词典": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "每行一条 Fal tone_list 发音规则；留空不发送。",
                    },
                ),
                "响度标准化": (list(NORMALIZATION_MODES), {"default": NORMALIZATION_DEFAULT}),
                "目标响度": (
                    "FLOAT",
                    {"default": -18.0, "min": -70.0, "max": -10.0, "step": 0.1},
                ),
                "目标响度范围": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "目标峰值": (
                    "FLOAT",
                    {"default": -0.5, "min": -3.0, "max": 0.0, "step": 0.1},
                ),
                "二次音色修改": (list(VOICE_MODIFY_MODES), {"default": VOICE_MODIFY_OMIT}),
                "二次音调": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "声音强度": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "音色调整": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    def build_options(self, **kwargs):
        spec = MiniMaxTTSOptionsSpec(
            english_normalization=kwargs.get("英文标准化", False),
            sample_rate=kwargs.get("采样率", 32000),
            bitrate=kwargs.get("比特率", 128000),
            audio_format=kwargs.get("音频格式", "mp3"),
            channel=kwargs.get("声道", 1),
            pronunciation_rules=_pronunciation_rules(kwargs.get("发音词典", "")),
            normalization_mode=kwargs.get("响度标准化", NORMALIZATION_DEFAULT),
            target_loudness=kwargs.get("目标响度", -18.0),
            target_range=kwargs.get("目标响度范围", 8.0),
            target_peak=kwargs.get("目标峰值", -0.5),
            voice_modify_mode=kwargs.get("二次音色修改", VOICE_MODIFY_OMIT),
            voice_modify_pitch=kwargs.get("二次音调", 0),
            voice_modify_intensity=kwargs.get("声音强度", 0),
            voice_modify_timbre=kwargs.get("音色调整", 0),
        )
        return (validate_options(spec),)


class LingliuMiniMaxTTS(LingliuBaseNode):
    CATEGORY = "Lingliu/MiniMax TTS"
    RETURN_TYPES = (IO.AUDIO, "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("音频", "任务ID", "响应", "音频URL", "时长毫秒")
    FUNCTION = "generate_speech"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Every queue execution intentionally creates one new paid task.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {"default": "", "multiline": True}),
                "模型": (list(MODELS), {"default": "speech-2.8-hd"}),
                "语种增强": (list(LANGUAGE_CHOICES), {"default": LANGUAGE_UNSPECIFIED}),
                "音色ID": (
                    "STRING",
                    {
                        "default": "male-qn-qingse",
                        "tooltip": "由可搜索音色选择器同步；也可保存自定义或克隆音色 ID。",
                    },
                ),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "音量": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05}),
                "基础音调": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
                "情绪": (list(EMOTION_CHOICES), {"default": EMOTION_UNSPECIFIED}),
            },
            "optional": {"高级设置": (OPTIONS_TYPE,)},
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

    async def generate_speech(self, **kwargs):
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)
        workflow = MiniMaxTTSWorkflow()
        try:
            result = await workflow.execute(
                text=kwargs.get("文本", ""),
                model=kwargs.get("模型", "speech-2.8-hd"),
                language_boost=kwargs.get("语种增强", LANGUAGE_UNSPECIFIED),
                voice_id=kwargs.get("音色ID", "male-qn-qingse"),
                speed=kwargs.get("语速", 1.0),
                volume=kwargs.get("音量", 1.0),
                pitch=kwargs.get("基础音调", 0),
                emotion=kwargs.get("情绪", EMOTION_UNSPECIFIED),
                options=kwargs.get("高级设置"),
                notify=self._notify(unique_id),
            )
        except MiniMaxTTSError:
            raise
        return (
            result.audio,
            result.task_id,
            result.response,
            result.audio_url,
            result.duration_ms,
        )


__all__ = [
    "LingliuMiniMaxTTS",
    "LingliuMiniMaxTTSOptions",
    "MiniMaxTTSError",
    "MiniMaxTTSOptionsSpec",
    "MiniMaxTTSResult",
    "MiniMaxTTSWorkflow",
    "PyAVAudioDecoder",
    "build_speech_payload",
]
