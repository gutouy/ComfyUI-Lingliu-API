import asyncio
import base64
import io
import json
import math
import os
import re
import tempfile
import wave
from dataclasses import dataclass
from urllib.parse import quote, urljoin, urlparse

import requests
import torch
from comfy.comfy_types import IO

from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    get_config,
    normalize_provider,
)
from .nodes_minimax_tts import MiniMaxTTSError, PyAVAudioDecoder
from .nodes_openai_compatible import LingliuBaseNode


NODE_ID = "lingliu_minimax_voice_clone"
PREVIEW_MODELS = (
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
)
ALLOWED_PROVIDERS = frozenset({LINGLIU_PROVIDER})
ENDPOINT_SLUG = "fal-ai/minimax/voice-clone"
PENDING_STATUSES = frozenset(
    {
        "NOT_START",
        "SUBMITTED",
        "QUEUED",
        "PENDING",
        "IN_QUEUE",
        "IN_PROGRESS",
        "RUNNING",
        "PROCESSING",
    }
)
SUCCESS_STATUSES = frozenset({"COMPLETED", "SUCCESS", "SUCCEEDED"})
FAILURE_STATUSES = frozenset(
    {"FAILED", "ERROR", "CANCELLED", "CANCELED", "FAILURE"}
)
MAX_REFERENCE_WAV_BYTES = 20 * 1024 * 1024
MAX_PREVIEW_AUDIO_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_POLL_ATTEMPTS = 600
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_INITIAL_POLL_DELAY = 2.0


class MiniMaxVoiceCloneError(RuntimeError):
    def __init__(
        self,
        message,
        *,
        stage="validation",
        status_code=None,
        payload=None,
        task_id="",
        voice_id="",
        response_json="",
    ):
        self.message = str(message or "MiniMax 语音克隆执行失败。")
        self.stage = str(stage or "internal")
        self.status_code = status_code
        self.payload = payload
        self.task_id = str(task_id or "")
        self.voice_id = str(voice_id or "")
        self.response_json = str(response_json or "")
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}\n{self.response_json}" if self.response_json else self.message


@dataclass(frozen=True)
class MiniMaxVoiceCloneConfig:
    provider: str
    base_url: str
    api_key: str
    timeout: int


@dataclass(frozen=True)
class EncodedReferenceAudio:
    data_uri: str
    duration_seconds: float
    sample_rate: int
    channels: int
    wav_bytes: int


@dataclass(frozen=True)
class MiniMaxVoiceCloneResult:
    voice_id: str
    audio: object
    task_id: str
    response: str
    audio_url: str


class RequestsHTTPAdapter:
    def request(self, method, url, **kwargs):
        return requests.request(method, url, **kwargs)


def _encode_reference_audio(audio):
    if not isinstance(audio, dict):
        raise MiniMaxVoiceCloneError("参考音频必须是 ComfyUI AUDIO 对象。")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise MiniMaxVoiceCloneError("参考音频 waveform 必须是 [B,C,T] 三维张量。")
    if not waveform.dtype.is_floating_point:
        raise MiniMaxVoiceCloneError("参考音频 waveform 必须是浮点张量。")
    if waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or waveform.shape[2] < 1:
        raise MiniMaxVoiceCloneError("参考音频只支持单批次、单声道或双声道。")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise MiniMaxVoiceCloneError("参考音频 sample_rate 必须是正整数。")
    if not bool(torch.isfinite(waveform).all().item()):
        raise MiniMaxVoiceCloneError("参考音频包含 NaN 或 Inf。")
    channels = int(waveform.shape[1])
    frames = int(waveform.shape[2])
    duration_seconds = frames / float(sample_rate)
    if duration_seconds < 10.0 or duration_seconds > 300.0:
        raise MiniMaxVoiceCloneError("参考音频时长必须在 10 秒至 300 秒之间。")
    expected_size = 44 + frames * channels * 2
    if expected_size > MAX_REFERENCE_WAV_BYTES:
        raise MiniMaxVoiceCloneError("参考音频编码后超过 20 MiB 上限。")

    samples = waveform.detach().to(device="cpu", dtype=torch.float32)[0]
    samples = torch.clamp(samples, -1.0, 1.0)
    pcm = torch.where(samples < 0, samples * 32768.0, samples * 32767.0)
    pcm = torch.round(pcm).to(torch.int16).transpose(0, 1).contiguous()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.numpy().tobytes())
    wav_data = output.getvalue()
    if len(wav_data) > MAX_REFERENCE_WAV_BYTES:
        raise MiniMaxVoiceCloneError("参考音频编码后超过 20 MiB 上限。")
    return EncodedReferenceAudio(
        data_uri="data:audio/wav;base64," + base64.b64encode(wav_data).decode("ascii"),
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        wav_bytes=len(wav_data),
    )


def _validate_public_parameters(
    *, preview_text, preview_model, noise_reduction, volume_normalization, accuracy
):
    if not isinstance(preview_text, str) or not preview_text.strip():
        raise MiniMaxVoiceCloneError("试听文本不能为空。")
    if len(preview_text) > 1000:
        raise MiniMaxVoiceCloneError("试听文本不能超过 1000 个字符。")
    if preview_model not in PREVIEW_MODELS:
        raise MiniMaxVoiceCloneError(
            f"不支持试听模型 {preview_model!r}，可用模型：{', '.join(PREVIEW_MODELS)}。"
        )
    if not isinstance(noise_reduction, bool):
        raise MiniMaxVoiceCloneError("降噪必须是布尔值。")
    if not isinstance(volume_normalization, bool):
        raise MiniMaxVoiceCloneError("音量标准化必须是布尔值。")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or not 0.0 <= float(accuracy) <= 1.0
    ):
        raise MiniMaxVoiceCloneError("准确度必须是 0 到 1 之间的有限数值。")
    return float(accuracy)


def _validate_preview_audio(audio):
    if not isinstance(audio, dict):
        raise MiniMaxVoiceCloneError("试听解码器没有返回 AUDIO 对象。", stage="decode")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise MiniMaxVoiceCloneError("试听 AUDIO waveform 无效。", stage="decode")
    if waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2) or waveform.shape[2] < 1:
        raise MiniMaxVoiceCloneError("试听 AUDIO 的批次、声道或长度无效。", stage="decode")
    if waveform.dtype != torch.float32 or not bool(torch.isfinite(waveform).all().item()):
        raise MiniMaxVoiceCloneError(
            "试听 AUDIO waveform 必须是有限的 float32 张量。", stage="decode"
        )
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise MiniMaxVoiceCloneError("试听 AUDIO sample_rate 无效。", stage="decode")
    return audio


def _validate_preview_url(url):
    try:
        parsed = urlparse(str(url or ""))
        port = parsed.port
    except Exception as exc:
        raise MiniMaxVoiceCloneError("试听音频 URL 无效。", stage="preview") from exc
    hostname = str(parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port not in (None, 443)
        or (hostname != "fal.media" and not hostname.endswith(".fal.media"))
    ):
        raise MiniMaxVoiceCloneError("试听音频 URL 未通过 Fal 域名校验。", stage="preview")
    return str(url).strip()


def _redact_signed_url(url):
    value = str(url or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.query:
        return value
    return parsed._replace(query="<redacted>").geturl()


def _unwrap_data(payload):
    current = payload
    while isinstance(current, dict) and isinstance(current.get("data"), dict):
        current = current["data"]
    return current if isinstance(current, dict) else {}


def _extract_result_data(payload):
    for source in _dict_layers(payload):
        if "custom_voice_id" in source:
            return source
    return _unwrap_data(payload)


def _dict_layers(payload):
    current = payload
    while isinstance(current, dict):
        yield current
        nested = current.get("data")
        if not isinstance(nested, dict):
            break
        current = nested


def _extract_task_id(payload):
    for source in _dict_layers(payload):
        for key in ("request_id", "task_id", "id"):
            value = source.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
                return str(value).strip()
    return ""


def _extract_status(payload):
    for source in _dict_layers(payload):
        for key in ("status", "state"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
    return ""


def _extract_failure_reason(payload):
    for source in _dict_layers(payload):
        for key in ("fail_reason", "failure_reason", "message", "detail"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        error = source.get("error")
        if isinstance(error, dict):
            value = error.get("message") or error.get("detail")
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()
    return ""


def _redact_text(value, secrets=()):
    text = str(value or "")
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "<redacted>")
    text = re.sub(
        r"(?i)data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+",
        "<redacted-data-uri>",
        text,
    )
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer <redacted>", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "sk-<redacted>", text)
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<redacted>", text)
    return text


def _redact_value(value, secrets=()):
    if isinstance(value, dict):
        return {
            _redact_text(key, secrets): _redact_value(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(value, secrets)


class MiniMaxVoiceCloneWorkflow:
    def __init__(
        self,
        *,
        http=None,
        decoder=None,
        sleep=None,
        max_poll_attempts=DEFAULT_MAX_POLL_ATTEMPTS,
        poll_interval=DEFAULT_POLL_INTERVAL,
        initial_poll_delay=DEFAULT_INITIAL_POLL_DELAY,
    ):
        self.http = http or RequestsHTTPAdapter()
        self.decoder = decoder or PyAVAudioDecoder()
        self.sleep = sleep or asyncio.sleep
        self.max_poll_attempts = int(max_poll_attempts)
        self.poll_interval = float(poll_interval)
        self.initial_poll_delay = float(initial_poll_delay)

    @staticmethod
    def resolve_config():
        runtime = get_config() or {}
        provider = normalize_provider(runtime.get("api_provider"))
        if provider not in ALLOWED_PROVIDERS:
            raise MiniMaxVoiceCloneError(
                "MiniMax 语音克隆仅支持灵流线路。",
                stage="route",
            )
        api_key = str(runtime.get("api_key") or "").strip()
        if not api_key:
            raise MiniMaxVoiceCloneError("全局 API 密钥为空。", stage="auth")
        base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise MiniMaxVoiceCloneError("全局 API 地址为空。", stage="route")
        timeout_value = runtime.get("timeout", 300)
        if isinstance(timeout_value, bool):
            raise MiniMaxVoiceCloneError("全局请求超时必须是正整数。", stage="route")
        try:
            timeout = int(timeout_value)
        except (TypeError, ValueError) as exc:
            raise MiniMaxVoiceCloneError("全局请求超时必须是正整数。", stage="route") from exc
        if timeout < 1:
            raise MiniMaxVoiceCloneError("全局请求超时必须大于 0 秒。", stage="route")
        return MiniMaxVoiceCloneConfig(provider, base_url, api_key, timeout)

    @staticmethod
    def create_url(config):
        return build_url(config.base_url, f"/fal/{ENDPOINT_SLUG}")

    @staticmethod
    def status_url(create_url, task_id):
        return f"{create_url.rstrip('/')}/requests/{quote(task_id, safe='')}/status"

    @staticmethod
    def result_url(create_url, task_id):
        return f"{create_url.rstrip('/')}/requests/{quote(task_id, safe='')}"

    @staticmethod
    def _notify(notify, task_id, status):
        if not notify:
            return
        try:
            notify(task_id, status)
        except Exception:
            pass

    @staticmethod
    def _decode_json(response, stage):
        try:
            payload = response.json()
        except Exception as exc:
            raise MiniMaxVoiceCloneError(
                f"{stage} 响应不是 JSON。",
                stage=stage,
                status_code=getattr(response, "status_code", None),
            ) from exc
        if not isinstance(payload, dict):
            raise MiniMaxVoiceCloneError(
                f"{stage} 响应必须是 JSON 对象。",
                stage=stage,
                status_code=getattr(response, "status_code", None),
            )
        return payload

    async def _post_create_once(self, url, config, body):
        try:
            response = await asyncio.to_thread(
                self.http.request,
                "POST",
                url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=config.timeout,
                json=body,
            )
        except Exception as exc:
            raise MiniMaxVoiceCloneError(
                "create 网络错误；创建结果不确定，已禁止自动重发以避免重复付费。",
                stage="create",
            ) from exc
        status_code = int(getattr(response, "status_code", 200) or 200)
        payload = self._decode_json(response, "create")
        if 200 <= status_code < 300 or _extract_task_id(payload):
            return payload
        raise MiniMaxVoiceCloneError(
            f"create HTTP {status_code}；已禁止自动重发以避免重复付费。",
            stage="create",
            status_code=status_code,
        )

    @staticmethod
    def _retry_delay(response, attempt):
        headers = getattr(response, "headers", {}) or {}
        raw = str(headers.get("Retry-After") or headers.get("retry-after") or "").strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                value = -1
            if value >= 0:
                return value
        return float(min(2 ** (attempt - 1), 4))

    async def _request_json(
        self,
        method,
        url,
        config,
        *,
        stage,
        retry_get=True,
        allow_retry_exhausted=False,
        task_id="",
    ):
        if method.upper() != "GET":
            raise ValueError("Only idempotent GET requests use the retry path")
        attempts = 3 if retry_get else 1
        last_error = None
        for attempt in range(1, attempts + 1):
            response = None
            try:
                response = await asyncio.to_thread(
                    self.http.request,
                    "GET",
                    url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {config.api_key}",
                    },
                    timeout=config.timeout,
                )
            except Exception as exc:
                last_error = MiniMaxVoiceCloneError(
                    f"{stage} 网络错误：{exc}", stage=stage, task_id=task_id
                )
                if attempt < attempts:
                    await self.sleep(float(min(2 ** (attempt - 1), 4)))
                    continue
                if allow_retry_exhausted:
                    return None
                raise last_error from exc

            status_code = int(getattr(response, "status_code", 200) or 200)
            try:
                payload = self._decode_json(response, stage)
            except MiniMaxVoiceCloneError as exc:
                if task_id and not exc.task_id:
                    exc.task_id = task_id
                last_error = exc
                if status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                    await self.sleep(self._retry_delay(response, attempt))
                    continue
                if allow_retry_exhausted and status_code in {429, 500, 502, 503, 504}:
                    return None
                raise
            if 200 <= status_code < 300:
                return payload
            last_error = MiniMaxVoiceCloneError(
                f"{stage} HTTP {status_code}",
                stage=stage,
                status_code=status_code,
                payload=payload,
                task_id=task_id,
            )
            if status_code in {429, 500, 502, 503, 504} and attempt < attempts:
                await self.sleep(self._retry_delay(response, attempt))
                continue
            if allow_retry_exhausted and status_code in {429, 500, 502, 503, 504}:
                return None
            raise last_error
        if allow_retry_exhausted:
            return None
        raise last_error or MiniMaxVoiceCloneError(f"{stage} 请求失败。", stage=stage)

    def _download_once(self, url, config):
        current_url = _validate_preview_url(url)
        response = None
        for redirect_index in range(4):
            response = self.http.request(
                "GET",
                current_url,
                headers={"Accept": "audio/*, application/octet-stream"},
                timeout=config.timeout,
                stream=True,
                allow_redirects=False,
            )
            status_code = int(getattr(response, "status_code", 200) or 200)
            if not 300 <= status_code < 400:
                break
            headers = getattr(response, "headers", {}) or {}
            location = str(headers.get("Location") or headers.get("location") or "").strip()
            close = getattr(response, "close", None)
            if callable(close):
                close()
            response = None
            if not location or redirect_index == 3:
                raise MiniMaxVoiceCloneError(
                    "试听音频重定向无效或次数过多。", stage="download"
                )
            current_url = _validate_preview_url(urljoin(current_url, location))
        if response is None:
            raise MiniMaxVoiceCloneError("试听音频下载没有返回响应。", stage="download")
        temp_path = ""
        try:
            status_code = int(getattr(response, "status_code", 200) or 200)
            if not 200 <= status_code < 300:
                raise MiniMaxVoiceCloneError(
                    f"download HTTP {status_code}", stage="download", status_code=status_code
                )
            headers = getattr(response, "headers", {}) or {}
            content_type = str(headers.get("Content-Type") or headers.get("content-type") or "")
            media_type = content_type.split(";", 1)[0].strip().lower()
            if not (
                media_type.startswith("audio/")
                or media_type in {"application/octet-stream", "binary/octet-stream"}
            ):
                raise MiniMaxVoiceCloneError("下载响应不是音频类型。", stage="download")
            content_length = str(
                headers.get("Content-Length") or headers.get("content-length") or ""
            ).strip()
            if content_length.isdigit() and int(content_length) > MAX_PREVIEW_AUDIO_BYTES:
                raise MiniMaxVoiceCloneError(
                    "试听音频超过 256 MiB 下载上限。", stage="download"
                )
            with tempfile.NamedTemporaryFile(
                prefix="lingliu_minimax_voice_clone_", suffix=".audio", delete=False
            ) as handle:
                temp_path = handle.name
                total = 0
                iterator = getattr(response, "iter_content", None)
                chunks = iterator(chunk_size=1024 * 1024) if callable(iterator) else (
                    getattr(response, "content", b""),
                )
                for chunk in chunks:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_PREVIEW_AUDIO_BYTES:
                        raise MiniMaxVoiceCloneError("试听音频超过 256 MiB 下载上限。", stage="download")
                    handle.write(chunk)
            if total == 0:
                raise MiniMaxVoiceCloneError("下载到的试听音频为空。", stage="download")
            return temp_path
        except Exception:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    async def _download_and_decode(self, url, config):
        temp_path = ""
        last_error = None
        for attempt in range(1, 4):
            try:
                temp_path = await asyncio.to_thread(self._download_once, url, config)
                break
            except MiniMaxVoiceCloneError as exc:
                last_error = exc
                if exc.status_code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise
            except (requests.exceptions.RequestException, OSError) as exc:
                last_error = MiniMaxVoiceCloneError(
                    f"download 网络错误：{exc}", stage="download"
                )
                if attempt == 3:
                    raise last_error from exc
            await self.sleep(float(min(2 ** (attempt - 1), 4)))
        if not temp_path:
            raise last_error or MiniMaxVoiceCloneError("试听音频下载失败。", stage="download")
        try:
            try:
                audio = await asyncio.to_thread(self.decoder.decode_file, temp_path)
            except MiniMaxVoiceCloneError:
                raise
            except MiniMaxTTSError as exc:
                raise MiniMaxVoiceCloneError(
                    exc.message,
                    stage=exc.stage or "decode",
                    status_code=exc.status_code,
                    payload=exc.payload,
                ) from exc
            except Exception as exc:
                raise MiniMaxVoiceCloneError(
                    "试听音频解码失败。", stage="decode"
                ) from exc
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return _validate_preview_audio(audio)

    async def _execute_impl(
        self,
        *,
        reference_audio,
        preview_text,
        preview_model,
        noise_reduction,
        volume_normalization,
        accuracy,
        notify=None,
    ):
        accuracy = _validate_public_parameters(
            preview_text=preview_text,
            preview_model=preview_model,
            noise_reduction=noise_reduction,
            volume_normalization=volume_normalization,
            accuracy=accuracy,
        )
        config = self.resolve_config()
        encoded = await asyncio.to_thread(_encode_reference_audio, reference_audio)
        create_url = self.create_url(config)
        audio_summary = {
            "mime": "audio/wav",
            "duration_seconds": encoded.duration_seconds,
            "sample_rate": encoded.sample_rate,
            "channels": encoded.channels,
            "wav_bytes": encoded.wav_bytes,
        }
        body = {
            "audio_url": encoded.data_uri,
            "noise_reduction": noise_reduction,
            "need_volume_normalization": volume_normalization,
            "accuracy": accuracy,
            "text": preview_text,
            "model": preview_model,
        }
        create_payload = await self._post_create_once(create_url, config, body)
        body = None
        encoded = None
        task_id = _extract_task_id(create_payload)
        if not task_id:
            raise MiniMaxVoiceCloneError("创建响应缺少任务 ID。", stage="create")
        self._notify(notify, task_id, _extract_status(create_payload) or "SUBMITTED")
        if self.initial_poll_delay:
            await self.sleep(self.initial_poll_delay)
        status_endpoint = self.status_url(create_url, task_id)
        for attempt in range(self.max_poll_attempts):
            status_payload = await self._request_json(
                "GET",
                status_endpoint,
                config,
                stage="status",
                allow_retry_exhausted=True,
                task_id=task_id,
            )
            if status_payload is None:
                if attempt + 1 < self.max_poll_attempts and self.poll_interval:
                    await self.sleep(self.poll_interval)
                continue
            status = _extract_status(status_payload)
            self._notify(notify, task_id, status or "UNKNOWN")
            if status in SUCCESS_STATUSES:
                break
            if status in PENDING_STATUSES:
                if attempt + 1 < self.max_poll_attempts and self.poll_interval:
                    await self.sleep(self.poll_interval)
                continue
            if status in FAILURE_STATUSES:
                reason = _extract_failure_reason(status_payload)
                raise MiniMaxVoiceCloneError(
                    reason or f"语音克隆任务失败：{status}。",
                    stage="status",
                    payload=status_payload,
                    task_id=task_id,
                )
            raise MiniMaxVoiceCloneError(
                f"状态接口返回未知状态：{status or '<empty>'}。",
                stage="status",
                task_id=task_id,
            )
        else:
            self._notify(notify, task_id, "TIMEOUT")
            raise MiniMaxVoiceCloneError("等待语音克隆任务完成超时。", stage="status", task_id=task_id)

        result_payload = await self._request_json(
            "GET",
            self.result_url(create_url, task_id),
            config,
            stage="result",
            task_id=task_id,
        )
        result_data = _extract_result_data(result_payload)
        raw_voice_id = result_data.get("custom_voice_id")
        if not isinstance(raw_voice_id, str) or not raw_voice_id.strip():
            raise MiniMaxVoiceCloneError("结果响应缺少 custom_voice_id。", stage="result", task_id=task_id)
        voice_id = raw_voice_id.strip()
        audio_info = result_data.get("audio")
        raw_audio_url = str(
            audio_info.get("url") if isinstance(audio_info, dict) else ""
        ).strip()
        envelope = {
            "node": NODE_ID,
            "provider": config.provider,
            "model": preview_model,
            "request": {
                "audio": audio_summary,
                "text_characters": len(preview_text),
                "noise_reduction": noise_reduction,
                "need_volume_normalization": volume_normalization,
                "accuracy": accuracy,
            },
            "task_id": task_id,
            "custom_voice_id": voice_id,
        }
        audio = None
        audio_url = ""
        warning = None
        try:
            if not raw_audio_url:
                raise MiniMaxVoiceCloneError(
                    "克隆结果没有返回试听音频。", stage="preview"
                )
            audio_url = _validate_preview_url(raw_audio_url)
            self._notify(notify, task_id, "DOWNLOADING")
            audio = await self._download_and_decode(audio_url, config)
        except Exception as exc:
            stage = exc.stage if isinstance(exc, MiniMaxVoiceCloneError) else "preview"
            warning = {
                "stage": stage,
                "message": _redact_text(
                    (
                        exc.message
                        if isinstance(exc, MiniMaxVoiceCloneError)
                        else "克隆已成功，但试听音频处理失败。"
                    ),
                    (config.api_key,),
                ),
            }
            if not audio_url:
                audio_url = ""
        envelope["status"] = "succeeded_with_warning" if warning else "succeeded"
        if warning:
            envelope["warning"] = warning
        if audio_url:
            envelope["audio_url"] = _redact_signed_url(audio_url)
        self._notify(
            notify,
            task_id,
            "SUCCEEDED_WITH_WARNING" if warning else "SUCCEEDED",
        )
        return MiniMaxVoiceCloneResult(
            voice_id=voice_id,
            audio=audio,
            task_id=task_id,
            response=json.dumps(envelope, ensure_ascii=False, indent=2),
            audio_url=audio_url,
        )

    async def execute(
        self,
        *,
        reference_audio,
        preview_text,
        preview_model,
        noise_reduction,
        volume_normalization,
        accuracy,
        notify=None,
    ):
        try:
            return await self._execute_impl(
                reference_audio=reference_audio,
                preview_text=preview_text,
                preview_model=preview_model,
                noise_reduction=noise_reduction,
                volume_normalization=volume_normalization,
                accuracy=accuracy,
                notify=notify,
            )
        except Exception as exc:
            if isinstance(exc, MiniMaxVoiceCloneError):
                stage = exc.stage
                status_code = exc.status_code
                task_id = exc.task_id
                voice_id = exc.voice_id
                message = exc.message
                payload = exc.payload
            else:
                stage = "internal"
                status_code = None
                task_id = ""
                voice_id = ""
                message = str(exc) or exc.__class__.__name__
                payload = None
            secrets = []
            try:
                runtime = get_config() or {}
                secret = str(runtime.get("api_key") or "").strip()
                if secret:
                    secrets.append(secret)
            except Exception:
                pass
            safe_message = _redact_text(message, secrets)
            safe_upstream = _redact_value(payload, secrets) if payload is not None else None
            error = {
                "stage": stage,
                "message": safe_message,
                "status_code": status_code,
            }
            if safe_upstream is not None:
                error["upstream"] = safe_upstream
            error_response = json.dumps(
                {
                    "node": NODE_ID,
                    "model": _redact_text(preview_model, secrets),
                    "status": "error",
                    "task_id": task_id,
                    "custom_voice_id": voice_id,
                    "error": error,
                },
                ensure_ascii=False,
                indent=2,
            )
            raise MiniMaxVoiceCloneError(
                safe_message,
                stage=stage,
                status_code=status_code,
                task_id=task_id,
                voice_id=voice_id,
                response_json=error_response,
            ) from exc


class LingliuMiniMaxVoiceClone(LingliuBaseNode):
    CATEGORY = "Lingliu/MiniMax TTS"
    FUNCTION = "clone_voice"
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", IO.AUDIO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("音色ID", "试听音频", "任务ID", "响应", "试听音频URL")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "参考音频": (
                    IO.AUDIO,
                    {
                        "tooltip": (
                            "仅上传已获得合法授权的声音；音频会发送到当前配置的服务商。"
                            "支持 10 秒至 300 秒的单声道或双声道音频。"
                        )
                    },
                ),
                "试听文本": (
                    "STRING",
                    {
                        "default": "你好，这是克隆音色试听。",
                        "multiline": True,
                        "tooltip": (
                            "每次排队都会生成试听并可能按字符计费，最多 1000 个字符。"
                            "克隆接口的试听不会激活音色；请在 7 天内通过 TTS 合成以长期保留。"
                        ),
                    },
                ),
                "试听模型": (PREVIEW_MODELS, {"default": "speech-02-hd"}),
                "降噪": ("BOOLEAN", {"default": False}),
                "音量标准化": ("BOOLEAN", {"default": False}),
                "准确度": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

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

    async def clone_voice(self, **kwargs):
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)
        workflow = MiniMaxVoiceCloneWorkflow()
        result = await workflow.execute(
            reference_audio=kwargs.get("参考音频"),
            preview_text=kwargs.get("试听文本", ""),
            preview_model=kwargs.get("试听模型", "speech-02-hd"),
            noise_reduction=kwargs.get("降噪", False),
            volume_normalization=kwargs.get("音量标准化", False),
            accuracy=kwargs.get("准确度", 0.7),
            notify=self._notify(unique_id),
        )
        status = json.loads(result.response).get("status", "succeeded")
        if status == "succeeded_with_warning":
            message = (
                f"音色ID: {result.voice_id}\n"
                "克隆已成功，但试听处理失败；请先复制 ID，勿重复排队以免再次付费。"
            )
        else:
            message = f"音色ID: {result.voice_id}\n状态: succeeded"
        self.publish_live_message(unique_id, message, append=False)
        return (
            result.voice_id,
            result.audio,
            result.task_id,
            result.response,
            result.audio_url,
        )


__all__ = ["LingliuMiniMaxVoiceClone"]
