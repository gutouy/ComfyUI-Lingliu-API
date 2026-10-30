"""WAN low-price contracts and a file-backed, one-create execution pipeline."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import tempfile
import uuid
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import numpy as np
import requests
from PIL import Image

from ..lingliu_config import build_url, get_config, mask_provider_payload, normalize_provider, provider_defaults
from ..utils import LingliuNullVideoAdapter, tensor2pil, video_save_options
from .nodes_wan3 import _redact, _redact_text


WAN3_LOW_MODELS = (
    "wan3.0-低价-1080p", "wan3.0-按次计费",
)
# Removed by the platform; old workflows that saved them get a clear message.
WAN3_LOW_RETIRED_MODELS = frozenset({"wan3.0-低价-480p", "wan3.0-低价-720p"})
WAN3_LOW_RETIRED_MESSAGE = "该模型已下线，请改选 wan3.0-低价-1080p 或 wan3.0-按次计费。"
WAN3_LOW_RESOLUTIONS = dict(zip(WAN3_LOW_MODELS, ("1080p", "720p")))
WAN3_LOW_DURATION_RANGES = dict(zip(WAN3_LOW_MODELS, ((2, 30), (5, 30))))
WAN3_LOW_REFERENCE_REQUIRED = frozenset({"wan3.0-按次计费"})
MEDIA_LIMITS = {"image": 10, "video": 5, "audio": 5}
# Local upload resource bounds, following the documented Wan media sizes.
FILE_LIMITS = {"image": 20 * 1024**2, "video": 100 * 1024**2, "audio": 15 * 1024**2}


class Wan3LowError(RuntimeError):
    def __init__(self, message, *, stage="validation", status_code=None, payload=None):
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code
        self.payload = payload


@dataclass(frozen=True)
class WanMediaAsset:
    slot: str
    media_type: str
    role: str
    path: str
    content_type: str
    size: int
    duration: float = 0.0


@dataclass(frozen=True)
class Wan3LowResult:
    video: object
    task_id: str
    response: str
    video_url: str


def _extract(payload, keys):
    if not isinstance(payload, Mapping):
        return ""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("data", "result", "output", "metadata", "error"):
        value = _extract(payload.get(key), keys)
        if value:
            return value
    return ""


def probe_media(path, media_type):
    """Read metadata and decode one frame from the materialized file."""
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == media_type), None)
        if stream is None:
            raise Wan3LowError(f"素材不包含 {media_type} 轨道。")
        duration = float(stream.duration * stream.time_base) if stream.duration is not None and stream.time_base else 0
        if not duration and container.duration is not None:
            duration = float(container.duration / av.time_base)
        if not math.isfinite(duration) or duration <= 0:
            raise Wan3LowError("无法读取有效素材时长。")
        if next(container.decode(stream), None) is None:
            raise Wan3LowError("素材轨道无法解码。")
        result = {"duration": duration}
        if media_type == "video":
            result.update(width=int(stream.width), height=int(stream.height))
            if result["width"] < 1 or result["height"] < 1:
                raise Wan3LowError("视频尺寸无效。")
        else:
            result.update(sample_rate=int(stream.codec_context.sample_rate))
        return result


class MultipartFile:
    """A bounded-read multipart body. requests sends it without buffering it."""
    def __init__(self, asset):
        boundary = "lingliu-" + uuid.uuid4().hex
        filename = Path(asset.path).name
        self.content_type = f"multipart/form-data; boundary={boundary}"
        self.header = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {asset.content_type}\r\n\r\n"
        ).encode("utf-8")
        self.footer = f"\r\n--{boundary}--\r\n".encode("ascii")
        self.file = open(asset.path, "rb")
        self.parts = [self.header, self.file, self.footer]
        self.part_index = 0
        self.offset = 0
        self.length = len(self.header) + asset.size + len(self.footer)

    def __len__(self):
        return self.length

    def read(self, size=-1):
        remaining = self.length if size is None or size < 0 else size
        chunks = []
        while remaining and self.part_index < len(self.parts):
            part = self.parts[self.part_index]
            if isinstance(part, bytes):
                chunk = part[self.offset:self.offset + remaining]
                self.offset += len(chunk)
                complete = self.offset >= len(part)
            else:
                chunk = part.read(remaining)
                complete = not chunk
            chunks.append(chunk)
            remaining -= len(chunk)
            if complete:
                self.part_index += 1
                self.offset = 0
        return b"".join(chunks)

    def close(self):
        self.file.close()


class CachedWanVideo:
    """A real local VIDEO result: dimensions, trims and saves reuse one file."""
    def __init__(self, path, metadata):
        self.video_path = str(path)
        self.metadata = metadata
        self._native_video = None

    def get_dimensions(self):
        return self.metadata["width"], self.metadata["height"]

    def get_duration(self):
        return self.metadata["duration"]

    def get_stream_source(self):
        return self.video_path

    def _native(self):
        if self._native_video is None:
            from comfy_api.latest._input_impl.video_types import VideoFromFile
            self._native_video = VideoFromFile(self.video_path)
        return self._native_video

    def get_components(self):
        return self._native().get_components()

    def get_frame_rate(self):
        return self._native().get_frame_rate()

    def get_frame_count(self):
        return self._native().get_frame_count()

    def as_trimmed(self, *args, **kwargs):
        return self._native().as_trimmed(*args, **kwargs)

    def get_container_format(self):
        return self._native().get_container_format()

    def get_bit_depth(self):
        # ComfyUI before v0.41 has no bit-depth API; its videos are 8-bit.
        native = getattr(self._native(), "get_bit_depth", None)
        return native() if callable(native) else 8

    def get_color_space(self):
        native = getattr(self._native(), "get_color_space", None)
        return native() if callable(native) else "auto"

    def save_to(self, output, format="auto", codec="auto", metadata=None,
                bit_depth=None, crf=None, color_space=None, preset=None, **kwargs):
        # Match the previous remote adapter: workflow metadata may contain
        # node API credentials and must not be embedded in a shareable video.
        del metadata, kwargs
        options = video_save_options(output, format, codec, bit_depth, crf, color_space, preset)
        if options is not None:
            return self._native().save_to(output, metadata=None, **options)
        if hasattr(output, "write"):
            with open(self.video_path, "rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if hasattr(output, "seek"):
                output.seek(0)
        elif os.path.abspath(str(output)) != os.path.abspath(self.video_path):
            shutil.copyfile(self.video_path, output)
        return True


class Wan3LowWorkflow:
    def __init__(self, http=None, sleep=None, config_resolver=None):
        self.http = http or requests
        self.sleep = sleep or asyncio.sleep
        self.config_resolver = config_resolver or get_config

    @staticmethod
    def check_cancelled():
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except ImportError:
            return
        throw_exception_if_processing_interrupted()

    @staticmethod
    def duration_range(model):
        if model in WAN3_LOW_RETIRED_MODELS:
            raise Wan3LowError(WAN3_LOW_RETIRED_MESSAGE)
        if model not in WAN3_LOW_DURATION_RANGES:
            raise Wan3LowError("模型不在当前 WAN3 低价模型列表中。")
        return WAN3_LOW_DURATION_RANGES[model]

    @classmethod
    def effective_duration(cls, model, duration):
        minimum, maximum = cls.duration_range(model)
        try:
            valid = not isinstance(duration, bool) and math.isfinite(float(duration)) and int(duration) == duration
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid or not minimum <= int(duration) <= maximum:
            raise Wan3LowError(f"{model} 时长必须为 {minimum}–{maximum} 的整数秒。")
        value = int(duration)
        return value, value, "call" if model in WAN3_LOW_REFERENCE_REQUIRED else "video_seconds"

    def resolve_config(self, override=""):
        config = dict(self.config_resolver() or {})
        provider = normalize_provider(config.get("api_provider"))
        if provider != "lingliu":
            raise Wan3LowError("WAN 低价节点只支持灵流线路。", stage="route")
        key = str(override or config.get("api_key") or "").strip()
        base = str(config.get("base_url") or provider_defaults(provider).get("base_url") or "").rstrip("/")
        if not key:
            raise Wan3LowError("API 密钥为空。", stage="auth")
        if urlsplit(base).scheme not in {"http", "https"} or not urlsplit(base).netloc:
            raise Wan3LowError("API 地址无效。", stage="route")
        return {"provider": provider, "api_key": key, "base_url": base, "timeout": int(config.get("timeout") or 300)}

    @staticmethod
    def expand_assets(raw_assets):
        expanded = []
        counts = dict.fromkeys(MEDIA_LIMITS, 0)
        for slot, kind, role, value in raw_assets:
            if value is None:
                continue
            if kind not in MEDIA_LIMITS or role != f"reference_{kind}":
                raise Wan3LowError(f"{slot} 的素材类型或角色不受支持。")
            size = 1
            if kind == "image":
                shape = getattr(value, "shape", None)
                if shape is not None and len(shape) == 4:
                    size = int(shape[0])
                elif isinstance(value, (list, tuple)):
                    size = len(value)
            if size < 1:
                raise Wan3LowError(f"{slot} 的图片批次为空。")
            counts[kind] += size
            if counts[kind] > MEDIA_LIMITS[kind]:
                label = {"image": "参考图", "video": "参考视频", "audio": "参考音频"}[kind]
                raise Wan3LowError(f"{label} 最多 {MEDIA_LIMITS[kind]} 个；计入 {slot} 后共 {counts[kind]} 个。")
            if size > 1:
                expanded.extend((f"{slot}[{i + 1}]", kind, role, value[i:i + 1]) for i in range(size))
            else:
                expanded.append((slot, kind, role, value))
        return expanded, counts

    @staticmethod
    def _waveform_to_file(value, path):
        rate = value.get("sample_rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0 or int(rate) != rate:
            raise Wan3LowError("音频 sample_rate 必须是有效的正整数。")
        waveform = value["waveform"]
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform[0]
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        if waveform.ndim != 2 or min(waveform.shape) < 1:
            raise Wan3LowError("一个音频接口只接受一段音频，waveform 应为 [1,C,T] 或 [C,T]。")
        if waveform.size * 2 + 44 > FILE_LIMITS["audio"]:
            raise Wan3LowError("编码后的参考音频超过节点 15 MB 上传上限。")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(waveform.shape[0])
            output.setsampwidth(2)
            output.setframerate(int(rate))
            for start in range(0, waveform.shape[1], 65536):
                chunk = waveform[:, start:start + 65536]
                if not np.isfinite(chunk).all():
                    raise Wan3LowError("音频包含 NaN 或无穷值。")
                pcm = (np.clip(chunk, -1, 1).T * 32767).astype("<i2")
                output.writeframesraw(pcm.tobytes())

    def _materialize(self, value, path, kind):
        if kind == "audio" and isinstance(value, Mapping) and value.get("waveform") is not None:
            self._waveform_to_file(value, path)
            return
        # VIDEO.save_to materializes trims/edits. Do not fall back to an
        # original path if saving the connected edited VIDEO fails.
        saver = getattr(value, "save_to", None)
        if callable(saver):
            saver(str(path))
            return
        source = value if isinstance(value, (str, os.PathLike)) else None
        if source is None:
            for key in ("path", "file_path", "file", "filename", "video_path", "audio_path", "url", "video_url", "audio_url"):
                candidate = value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
                if isinstance(candidate, (str, os.PathLike)) and str(candidate):
                    source = candidate
                    break
        if source is not None and os.path.isfile(str(source)):
            if os.path.getsize(source) > FILE_LIMITS[kind]:
                raise Wan3LowError("素材超过节点上传文件大小上限。")
            shutil.copyfile(source, path)
            return
        if isinstance(source, str) and source.startswith(("https://", "http://")):
            response = requests.get(source, stream=True, timeout=300)
            try:
                response.raise_for_status()
                size = 0
                with open(path, "wb") as output:
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > FILE_LIMITS[kind]:
                            raise Wan3LowError("素材超过节点上传文件大小上限。")
                        output.write(chunk)
            finally:
                response.close()
            return
        raise Wan3LowError("无法读取该媒体对象；请连接可保存的 VIDEO 或标准 AUDIO。")

    def prepare_assets(self, expanded, directory):
        prepared = []
        for index, (slot, kind, role, value) in enumerate(expanded, 1):
            self.check_cancelled()
            path = Path(directory) / f"reference-{index:02d}.{dict(image='png', video='mp4', audio='wav')[kind]}"
            try:
                if kind == "image":
                    if isinstance(value, Image.Image):
                        image = value
                    elif isinstance(value, (list, tuple)) and isinstance(value[0], Image.Image):
                        image = value[0]
                    else:
                        image = tensor2pil(value)[0]
                    if min(image.size) < 1:
                        raise Wan3LowError("图片尺寸无效。")
                    image.convert("RGB").save(path, "PNG")
                    duration = 0.0
                    mime = "image/png"
                else:
                    self._materialize(value, path, kind)
                    duration = probe_media(path, kind)["duration"]
                    # AV detects the container; normalize the suffix/MIME of
                    # copied MP3/MOV files instead of labeling them WAV/MP4.
                    import av
                    with av.open(str(path)) as container:
                        is_mp3 = kind == "audio" and "mp3" in container.format.name
                        is_wav = kind == "audio" and "wav" in container.format.name
                        is_mp4 = kind == "video" and any(n in container.format.name.split(",") for n in ("mp4", "mov"))
                    if kind == "audio" and not (is_mp3 or is_wav):
                        raise Wan3LowError("参考音频必须为 WAV/MP3。")
                    if kind == "video" and not is_mp4:
                        raise Wan3LowError("参考视频必须为 MP4/MOV。")
                    mime = "audio/mpeg" if is_mp3 else ("audio/wav" if kind == "audio" else "video/mp4")
                    if is_mp3:
                        renamed = path.with_suffix(".mp3")
                        path.rename(renamed)
                        path = renamed
                size = path.stat().st_size
                if size <= 0 or size > FILE_LIMITS[kind]:
                    raise Wan3LowError(f"文件大小必须在 1 字节至 {FILE_LIMITS[kind] // 1024**2} MB 之间。")
                prepared.append(WanMediaAsset(slot, kind, role, str(path), mime, size, duration))
            except Exception as exc:
                raise Wan3LowError(f"{slot}：{exc}") from exc
        return prepared

    @staticmethod
    def build_payload(model, prompt, content, duration, ratio=None, reference_video_seconds=0.0):
        payload = {
            "model": model, "prompt": prompt, "content": content,
            "duration": int(duration), "resolution": WAN3_LOW_RESOLUTIONS[model],
        }
        if reference_video_seconds:
            payload["reference_video_seconds"] = float(reference_video_seconds)
        return payload

    async def request_json(self, method, url, config, *, stage, body=None, asset=None, retry=False):
        for attempt in range(3 if retry else 1):
            encoder = None
            response = None
            try:
                headers = {"Accept": "application/json", "Authorization": f"Bearer {config['api_key']}"}
                kwargs = {"headers": headers, "timeout": config["timeout"], "allow_redirects": False}
                if asset is not None:
                    encoder = MultipartFile(asset)
                    headers.update({"Content-Type": encoder.content_type, "Content-Length": str(len(encoder))})
                    kwargs["data"] = encoder
                elif body is not None:
                    kwargs["json"] = body
                response = await asyncio.to_thread(self.http.request, method, url, **kwargs)
                status = int(response.status_code)
                try:
                    payload = response.json()
                except Exception:
                    payload = {"message": "接口没有返回合法 JSON。"}
                accepted = {200, 202} if stage == "create" else {200, 201, 202}
                if status in accepted and isinstance(payload, Mapping):
                    return payload
                reason = _extract(payload, ("message", "error", "detail", "reason"))
                error = Wan3LowError(f"{stage} HTTP {status}：{reason}", stage=stage, status_code=status, payload=payload)
                if not retry or status not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise error
            except (requests.RequestException, TimeoutError, OSError) as exc:
                error = Wan3LowError(f"{stage} 网络错误：{exc}", stage=stage)
                if not retry or attempt == 2:
                    raise error from exc
            finally:
                if response is not None:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                if encoder is not None:
                    encoder.close()
            await self.sleep(2 ** attempt)
        raise error

    def download_result(self, url, config):
        import folder_paths
        temp_dir = getattr(folder_paths, "get_temp_directory", tempfile.gettempdir)()
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        target = Path(temp_dir) / f"lingliu-wan3-{uuid.uuid4().hex}.mp4"
        original_origin = urlsplit(config["base_url"])[:2]
        current = url
        try:
            for _ in range(4):
                headers = {"Accept": "video/mp4,application/octet-stream"}
                if urlsplit(current)[:2] == original_origin:
                    headers["Authorization"] = f"Bearer {config['api_key']}"
                response = self.http.request("GET", current, headers=headers, stream=True, timeout=config["timeout"], allow_redirects=False)
                try:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location:
                            raise Wan3LowError("视频下载重定向缺少地址。", stage="download")
                        current = urljoin(current, location)
                        parsed = urlsplit(current)
                        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
                            raise Wan3LowError("视频下载重定向地址无效。", stage="download")
                        continue
                    if response.status_code != 200:
                        raise Wan3LowError(f"视频下载 HTTP {response.status_code}。", stage="download")
                    with open(target, "wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                finally:
                    response.close()
                metadata = probe_media(target, "video")
                metadata["bytes"] = target.stat().st_size
                return CachedWanVideo(target, metadata)
            raise Wan3LowError("视频下载重定向次数过多。", stage="download")
        except Exception:
            target.unlink(missing_ok=True)
            raise

    async def execute(self, *, prompt, model, duration, raw_assets, mode="自动", ratio="模型默认",
                      api_key="", wait_for_completion=True, max_poll_attempts=180, poll_interval=10,
                      notify=None):
        task_id = ""
        create_started = False
        envelope = {"node": "lingliu_wan3_low", "model": model, "uploads": []}
        secrets = [api_key]
        stage = "validation"
        try:
            model = str(model).strip()
            _, duration, billing_mode = self.effective_duration(model, duration)
            prompt = str(prompt or "").strip()
            if not prompt:
                raise Wan3LowError("提示词不能为空。")
            if mode not in {"自动", "文生视频", "参考生视频"}:
                raise Wan3LowError("生成模式不受支持。")
            expanded, counts = self.expand_assets(raw_assets)
            if mode == "文生视频" and expanded:
                raise Wan3LowError("文生视频模式不能连接参考素材。")
            if mode == "参考生视频" and not expanded:
                raise Wan3LowError("参考生视频模式至少需要一项参考素材。")
            if model in WAN3_LOW_REFERENCE_REQUIRED and not counts["image"]:
                raise Wan3LowError("按次计费模型必须连接至少一张参考图像。")
            if isinstance(max_poll_attempts, bool) or int(max_poll_attempts) != max_poll_attempts or not 1 <= max_poll_attempts <= 720:
                raise Wan3LowError("最大轮询次数必须是 1–720 的整数。")
            if isinstance(poll_interval, bool) or int(poll_interval) != poll_interval or not 2 <= poll_interval <= 60:
                raise Wan3LowError("轮询间隔必须是 2–60 的整数秒。")
            config = self.resolve_config(api_key)
            secrets.append(config["api_key"])
            warnings = []
            if ratio not in {"模型默认", "adaptive", "", None}:
                warnings.append(f"旧比例值 {ratio} 没有应用；当前使用模型默认比例。")
            with tempfile.TemporaryDirectory(prefix="lingliu-wan3-inputs-") as directory:
                prepared = await asyncio.to_thread(self.prepare_assets, expanded, directory)
                video_seconds = sum(a.duration for a in prepared if a.media_type == "video")
                audio_seconds = sum(a.duration for a in prepared if a.media_type == "audio")
                # This is the current platform's validated incoming limit.
                if video_seconds > 30 + 1e-6:
                    raise Wan3LowError("当前平台参考视频总时长不能超过 30 秒。")
                # Official Wan limits are advisory until each reseller variant
                # confirms them; do not inherit another node's hard limits.
                if video_seconds > 15 + 1e-6 or audio_seconds > 15 + 1e-6 or (video_seconds and video_seconds + duration > 30 + 1e-6):
                    warnings.append("素材超出 Wan 官方参考建议（视频/音频各合计15秒，视频输入+输出30秒）；平台/上游将按当前变体合同校验。")
                envelope.update(provider=config["provider"], media_counts=counts,
                                reference_video_seconds=video_seconds, reference_audio_seconds=audio_seconds,
                                billing_mode=billing_mode, warnings=warnings)
                content = [{"type": "text", "text": prompt}]
                ordinals = dict.fromkeys(MEDIA_LIMITS, 0)
                stage = "upload"
                for index, asset in enumerate(prepared, 1):
                    self.check_cancelled()
                    envelope["active_slot"] = asset.slot
                    if notify:
                        notify("", f"正在上传 {index}/{len(prepared)}：{asset.slot}")
                    payload = await self.request_json("POST", build_url(config["base_url"], "/v1/files"),
                                                      config, stage="upload", asset=asset, retry=True)
                    url = _extract(payload, ("url", "file_url", "fileUrl", "download_url"))
                    parsed = urlsplit(url)
                    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                        raise Wan3LowError(f"{asset.slot} 上传响应缺少有效 URL。", stage="upload")
                    field = f"{asset.media_type}_url"
                    content.append({"type": field, "role": asset.role, field: {"url": url}})
                    ordinals[asset.media_type] += 1
                    envelope["uploads"].append({"slot": asset.slot, "type": asset.media_type,
                        "ordinal": ordinals[asset.media_type], "bytes": asset.size, "duration": asset.duration})
                    Path(asset.path).unlink(missing_ok=True)
                envelope.pop("active_slot", None)
                body = self.build_payload(model, prompt, content, duration, reference_video_seconds=video_seconds)
                # Keep the exact HTTP body separate from the user-visible
                # summary so uploaded URLs are not exposed in diagnostics.
                envelope["request"] = {"model": model, "duration": duration, "resolution": body["resolution"],
                                       "reference_video_seconds": video_seconds, "media_counts": counts}
                stage = "create"
                self.check_cancelled()
                create_started = True
                created = await self.request_json("POST", build_url(config["base_url"], "/v1/videos"),
                                                   config, stage=stage, body=body)
            task_id = _extract(created, ("task_id", "taskId", "id", "video_id"))
            if not task_id:
                raise Wan3LowError("创建响应缺少任务 ID，请核查平台记录后再决定是否重新生成。", stage="create")
            envelope.update(task_id=task_id, status="submitted")
            if notify:
                notify(task_id, "submitted")
            if not wait_for_completion:
                return self._result(envelope, secrets, task_id)
            query_url = build_url(config["base_url"], f"/v1/videos/{quote(task_id, safe='')}")
            stage = "query"
            for attempt in range(int(max_poll_attempts)):
                self.check_cancelled()
                queried = await self.request_json("GET", query_url, config, stage=stage, retry=True)
                status = _extract(queried, ("status", "state", "task_status")).lower()
                if notify:
                    notify(task_id, status or "unknown")
                if status in {"completed", "complete", "success", "succeeded"}:
                    stage = "download"
                    url = query_url + "/content"
                    video = await asyncio.to_thread(self.download_result, url, config)
                    envelope.update(status="succeeded", video=video.metadata)
                    return self._result(envelope, secrets, task_id, video, url)
                if status in {"failed", "failure", "error", "rejected", "cancelled", "canceled", "expired"}:
                    raise Wan3LowError(_extract(queried, ("message", "reason", "fail_reason", "failReason")) or f"任务失败：{status}", stage="query", payload=queried)
                if status not in {"", "unknown", "queued", "pending", "submitted", "running", "processing", "in_progress"}:
                    raise Wan3LowError(f"查询返回未知状态：{status}", stage="query", payload=queried)
                if attempt + 1 < max_poll_attempts:
                    await self.sleep(poll_interval)
            envelope.update(status="timeout", error={"stage": "query", "message": "等待超时；请使用任务 ID 查询原任务，不要重新创建。"})
        except Exception as exc:
            if create_started and not task_id and getattr(exc, "status_code", None) in {None, 408, 500, 502, 503, 504}:
                envelope["submission_uncertain"] = True
            envelope.update(status="error", task_id=task_id, error={
                "stage": stage if stage == "download" else getattr(exc, "stage", stage), "message": str(exc),
                "status_code": getattr(exc, "status_code", None),
                "upstream": getattr(exc, "payload", None),
            })
            if envelope.get("submission_uncertain"):
                envelope["error"]["message"] += " 创建结果未知，请先核查平台任务记录，勿直接重复生成。"
        return self._result(envelope, secrets, task_id)

    @staticmethod
    def _result(envelope, secrets, task_id, video=None, url=""):
        safe = json.dumps(mask_provider_payload(_redact(envelope, secrets)), ensure_ascii=False, indent=2)
        if video is None:
            message = envelope.get("error", {}).get("message") or "任务已提交，尚无可保存的视频。"
            video = LingliuNullVideoAdapter(_redact_text(message, secrets))
        return Wan3LowResult(video, task_id, safe, url)
