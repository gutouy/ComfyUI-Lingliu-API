import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import tempfile
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from urllib.parse import quote, urlencode, urlparse

import av
import requests
from comfy.comfy_types import IO
from PIL import Image

from ..lingliu_config import (
    LINGLIU_BASE_URL,
    LINGLIU_PROVIDER,
    build_url,
    lingliu_channel_key,
    provider_defaults,
)
from ..utils import (
    LingliuNullVideoAdapter,
    LingliuVideoAdapter,
    blank_image_tensor,
    image_to_png_bytes,
    lingliu_temp_dir,
    load_image_from_candidate,
    pil2tensor,
)
from .nodes_doubao import (
    _media_input_to_bytes,
    _official_seedance_resolution,
    _waveform_to_wav_bytes,
)
from .nodes_openai_compatible import LingliuBaseNode


def _split_values(value):
    if not value or not str(value).strip():
        return []
    raw = str(value).replace("\r", "\n")
    items = []
    for part in raw.replace(",", "\n").split("\n"):
        candidate = str(part).strip()
        if candidate:
            items.append(candidate)
    return items


def _kwarg_alias(kwargs, *names, default=None, empty_string_is_missing=False):
    for name in names:
        if name not in kwargs:
            continue
        value = kwargs.get(name)
        if value is None:
            continue
        if empty_string_is_missing and isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _lingliu_force_channel_key(api_key, channel_id):
    key = str(api_key or "").strip()
    if key.lower().startswith("sk-"):
        parts = key[3:].split("-")
        if len(parts) > 1 and parts[-1].isdigit():
            key = "sk-" + "-".join(parts[:-1])
    return lingliu_channel_key(key, channel_id)


def _normalize_reference_url(value):
    raw = str(value or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered.startswith(("http://", "https://", "asset://", "file-")):
        return raw
    if lowered.startswith("asset-"):
        return f"asset://{raw}"
    return raw


def _parse_bundle(bundle_json):
    if not bundle_json or not str(bundle_json).strip():
        return {}
    try:
        payload = json.loads(str(bundle_json).strip())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _probe_video_duration_seconds(file_bytes):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=lingliu_temp_dir()) as handle:
            temp_path = handle.name
            handle.write(file_bytes)
        container = av.open(temp_path)
        try:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None or stream.duration is None or stream.time_base is None:
                raise RuntimeError("无法识别参考视频时长。")
            duration = float(stream.duration * stream.time_base)
        finally:
            container.close()
        return max(0.0, duration)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


class LingliuSeedanceOfficialBase(LingliuBaseNode):
    CATEGORY = "Lingliu/SeedanceOfficial"
    DEFAULT_PROVIDER = LINGLIU_PROVIDER
    TASK_POLL_INTERVAL = 10
    TASK_MAX_WAIT = 900

    def load_lingliu_official_config(self, api_key_override=""):
        runtime = self.load_config(api_key_override=api_key_override)
        config = provider_defaults(LINGLIU_PROVIDER)
        config["api_provider"] = LINGLIU_PROVIDER
        config["base_url"] = LINGLIU_BASE_URL
        config["api_key"] = lingliu_channel_key(str(api_key_override or runtime.get("api_key") or "").strip(), 5)
        config["timeout"] = int(runtime.get("timeout") or config.get("timeout") or 300)
        config["videos_generate_path"] = "/v1/video/generations"
        config["videos_status_path_template"] = "/v1/video/generations/{task_id}"
        return config

    def file_upload_config(self, config):
        upload_config = dict(config or {})
        upload_config["api_key"] = _lingliu_force_channel_key(upload_config.get("api_key", ""), 6)
        return upload_config

    def build_official_video_payload(
        self,
        *,
        model,
        prompt,
        duration,
        ratio,
        resolution,
        watermark,
        image_urls=None,
        reference_video_seconds=0,
        seed=-1,
        generate_audio=None,
        reference_video_urls=None,
        reference_audio_urls=None,
    ):
        duration_int = int(duration)
        resolution_value = str(resolution or "720p").strip() or "720p"
        resolution_value = _official_seedance_resolution(resolution_value)
        metadata = {
            "resolution": resolution_value,
            "ratio": str(ratio or "16:9").strip() or "16:9",
            "watermark": bool(watermark),
            "duration": duration_int,
        }
        if generate_audio is not None:
            metadata["generate_audio"] = bool(generate_audio)
        image_urls = [_normalize_reference_url(item) for item in (image_urls or []) if _normalize_reference_url(item)]
        reference_video_urls = [
            _normalize_reference_url(item)
            for item in (reference_video_urls or [])
            if _normalize_reference_url(item)
        ]
        reference_audio_urls = [
            _normalize_reference_url(item)
            for item in (reference_audio_urls or [])
            if _normalize_reference_url(item)
        ]
        if reference_video_urls:
            metadata["reference_videos"] = reference_video_urls
        if reference_audio_urls:
            metadata["reference_audios"] = reference_audio_urls
        if image_urls:
            metadata["urls"] = image_urls
        content = []
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
        for url in reference_video_urls:
            content.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
        for url in reference_audio_urls:
            content.append({"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"})
        if content:
            metadata["content"] = content

        payload = {
            "model": str(model or "").strip(),
            "prompt": str(prompt or "").strip(),
            "seconds": str(duration_int),
            "duration": duration_int,
            "resolution": metadata["resolution"],
            "metadata": metadata,
        }
        if image_urls:
            payload["images"] = image_urls
        if reference_video_seconds:
            value = round(float(reference_video_seconds), 3)
            metadata["reference_video_seconds"] = value
            payload["reference_video_seconds"] = value
        if int(seed) >= 0:
            payload["seed"] = int(seed)
        return payload

    def load_r2_config(self):
        config = provider_defaults(self.DEFAULT_PROVIDER)
        runtime = self.load_config("")
        for key in (
            "r2_account_id",
            "r2_access_key_id",
            "r2_secret_access_key",
            "r2_bucket_name",
            "r2_public_base_url",
            "r2_object_prefix",
        ):
            config[key] = str(runtime.get(key) or "").strip()
        return config

    def task_status_url(self, config, task_id):
        template = config.get("videos_status_path_template", "/contents/generations/tasks/{task_id}")
        return build_url(config.get("base_url"), template.format(task_id=task_id))

    def bridge_upload_file_request(self, bridge_config, file_content, filename, content_type, timeout=None):
        url = build_url(bridge_config.get("base_url"), "/v1/files")
        api_key = str(bridge_config.get("api_key") or "").strip()
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, file_content, content_type)},
            timeout=timeout or self.timeout(bridge_config),
        )
        self._raise_for_status_with_details(response, url)
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"隐藏桥接上传未返回 JSON 对象: {payload}")
        public_url = str(payload.get("url") or "").strip()
        if not public_url:
            raise RuntimeError(f"隐藏桥接上传未返回公网 URL: {payload}")
        return public_url, payload

    async def async_bridge_upload_file(self, bridge_config, file_content, filename, content_type):
        last_error = None
        base_timeout = self.timeout(bridge_config)
        for attempt in range(1, 6):
            current_timeout = min(int(base_timeout * (1.5 ** (attempt - 1))), 300)
            try:
                return await asyncio.to_thread(
                    self.bridge_upload_file_request,
                    bridge_config,
                    file_content,
                    filename,
                    content_type,
                    current_timeout,
                )
            except Exception as exc:
                last_error = exc
                if not self.is_transient_request_error(exc) or attempt == 5:
                    raise
            await asyncio.sleep(min(2 ** (attempt - 1), 20))
        raise last_error or RuntimeError("隐藏桥接上传失败，但没有返回具体错误。")

    def r2_endpoint_host(self, account_id):
        return f"{str(account_id or '').strip()}.r2.cloudflarestorage.com"

    def r2_endpoint_url(self, account_id, bucket_name, object_key):
        bucket_name = str(bucket_name or "").strip()
        object_key = str(object_key or "").strip().lstrip("/")
        encoded_key = quote(object_key, safe="/-_.~")
        return f"https://{self.r2_endpoint_host(account_id)}/{bucket_name}/{encoded_key}"

    def _sha256_hex(self, value):
        if isinstance(value, bytes):
            payload = value
        else:
            payload = str(value or "").encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _sign_hmac_hex(self, key_bytes, message):
        if isinstance(message, bytes):
            payload = message
        else:
            payload = str(message).encode("utf-8")
        return hmac.new(key_bytes, payload, hashlib.sha256).hexdigest()

    def _sign_hmac_bytes(self, key_bytes, message):
        if isinstance(message, bytes):
            payload = message
        else:
            payload = str(message).encode("utf-8")
        return hmac.new(key_bytes, payload, hashlib.sha256).digest()

    def _build_r2_put_headers(self, account_id, access_key_id, secret_access_key, bucket_name, object_key, payload_bytes, content_type):
        account_id = str(account_id or "").strip()
        access_key_id = str(access_key_id or "").strip()
        secret_access_key = str(secret_access_key or "").strip()
        bucket_name = str(bucket_name or "").strip()
        object_key = str(object_key or "").strip().lstrip("/")
        payload_hash = self._sha256_hex(payload_bytes)
        now = datetime.now(timezone.utc)
        x_amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        short_date = now.strftime("%Y%m%d")
        region = "auto"
        service = "s3"
        host = self.r2_endpoint_host(account_id)
        canonical_uri = f"/{quote(bucket_name, safe='-_.~')}/{quote(object_key, safe='/-_.~')}"
        canonical_headers = (
            f"content-type:{content_type}\n"
            f"host:{host}\n"
            f"x-amz-content-sha256:{payload_hash}\n"
            f"x-amz-date:{x_amz_date}\n"
        )
        signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
        canonical_request = "\n".join(
            [
                "PUT",
                canonical_uri,
                "",
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        credential_scope = f"{short_date}/{region}/{service}/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                x_amz_date,
                credential_scope,
                self._sha256_hex(canonical_request),
            ]
        )
        signing_key = self._sign_hmac_bytes(
            self._sign_hmac_bytes(
                self._sign_hmac_bytes(
                    self._sign_hmac_bytes(("AWS4" + secret_access_key).encode("utf-8"), short_date),
                    region,
                ),
                service,
            ),
            "aws4_request",
        )
        signature = self._sign_hmac_hex(signing_key, string_to_sign)
        authorization = (
            f"AWS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        headers = {
            "Authorization": authorization,
            "Content-Type": content_type,
            "Host": host,
            "X-Amz-Content-Sha256": payload_hash,
            "X-Amz-Date": x_amz_date,
        }
        return headers

    def _r2_public_url(self, public_base_url, object_key):
        base = str(public_base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError("R2 public_base_url is required.")
        return f"{base}/{quote(str(object_key or '').strip().lstrip('/'), safe='/-_.~')}"

    def _prepare_local_media_for_r2(self, media_type, media_input):
        if media_type == "image":
            file_bytes = image_to_png_bytes(media_input)
            filename = f"seedance_{uuid.uuid4().hex}.png"
            content_type = "image/png"
            return file_bytes, filename, content_type
        if media_type == "video":
            file_bytes, filename = _media_input_to_bytes(media_input, ".mp4", "video")
            if not file_bytes:
                raise RuntimeError("Could not read local video bytes for R2 upload.")
            filename = filename or f"seedance_{uuid.uuid4().hex}.mp4"
            content_type = mimetypes.guess_type(filename)[0] or "video/mp4"
            return file_bytes, filename, content_type
        file_bytes, filename = _media_input_to_bytes(media_input, ".wav", "audio")
        if not file_bytes and isinstance(media_input, Mapping) and media_input.get("waveform") is not None:
            waveform = media_input["waveform"]
            sample_rate = int(media_input.get("sample_rate", 44100))
            file_bytes = _waveform_to_wav_bytes(waveform, sample_rate)
            filename = "seedance_audio.wav"
        if not file_bytes:
            raise RuntimeError("Could not read local audio bytes for R2 upload.")
        filename = filename or f"seedance_{uuid.uuid4().hex}.wav"
        content_type = mimetypes.guess_type(filename)[0] or "audio/wav"
        return file_bytes, filename, content_type

    def r2_upload_bytes_request(self, r2_config, object_key, payload_bytes, content_type):
        account_id = str(r2_config.get("r2_account_id") or "").strip()
        bucket_name = str(r2_config.get("r2_bucket_name") or "").strip()
        access_key_id = str(r2_config.get("r2_access_key_id") or "").strip()
        secret_access_key = str(r2_config.get("r2_secret_access_key") or "").strip()
        if not all([account_id, bucket_name, access_key_id, secret_access_key]):
            raise RuntimeError("R2 配置不完整，请先运行 lingliu_r2_storage_set。")
        url = self.r2_endpoint_url(account_id, bucket_name, object_key)
        headers = self._build_r2_put_headers(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
            object_key=object_key,
            payload_bytes=payload_bytes,
            content_type=content_type,
        )
        response = requests.put(url, headers=headers, data=payload_bytes, timeout=300)
        self._raise_for_status_with_details(response, url)
        public_url = self._r2_public_url(r2_config.get("r2_public_base_url", ""), object_key)
        return {
            "bucket": bucket_name,
            "object_key": object_key,
            "content_type": content_type,
            "r2_url": url,
            "public_url": public_url,
        }

    async def r2_upload_local_media(self, media_type, media_input, name=""):
        r2_config = self.load_r2_config()
        file_bytes, filename, content_type = await asyncio.to_thread(self._prepare_local_media_for_r2, media_type, media_input)
        prefix = str(r2_config.get("r2_object_prefix") or "").strip().strip("/")
        safe_name = str(name or "").strip().replace("\\", "/").split("/")[-1].strip()
        ext = os.path.splitext(filename)[1] or ""
        stem = safe_name or os.path.splitext(filename)[0] or f"{media_type}_{uuid.uuid4().hex}"
        object_key = f"{prefix}/{stem}{ext}" if prefix else f"{stem}{ext}"
        upload_info = await asyncio.to_thread(
            self.r2_upload_bytes_request,
            r2_config,
            object_key,
            file_bytes,
            content_type,
        )
        return upload_info["public_url"], upload_info

    async def gateway_upload_local_media(self, config, media_type, media_input, name=""):
        upload_config = self.file_upload_config(config)
        file_bytes, filename, content_type = await asyncio.to_thread(
            self._prepare_local_media_for_r2,
            media_type,
            media_input,
        )
        safe_name = str(name or "").strip().replace("\\", "/").split("/")[-1].strip()
        stem = safe_name or os.path.splitext(filename)[0] or f"{media_type}_{uuid.uuid4().hex}"
        ext = os.path.splitext(filename)[1] or ""
        upload_filename = f"{stem}{ext}"
        public_url, raw_response = await self.async_bridge_upload_file(
            upload_config,
            file_bytes,
            upload_filename,
            content_type,
        )
        upload_info = {
            "provider": LINGLIU_PROVIDER,
            "upload_filename": upload_filename,
            "content_type": content_type,
            "public_url": public_url,
            "raw_response": raw_response,
        }
        if media_type == "video":
            try:
                upload_info["duration_seconds"] = _probe_video_duration_seconds(file_bytes)
            except Exception as exc:
                upload_info["duration_probe_error"] = str(exc)
        return public_url, upload_info

    def extract_video_url(self, payload):
        if not isinstance(payload, dict):
            return ""
        for key in ("result_url", "video_url", "videoUrl", "url", "video"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("url", "result_url", "video_url", "videoUrl"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        content = payload.get("content")
        if isinstance(content, dict):
            for key in ("video_url", "videoUrl", "url", "video"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        data = payload.get("data")
        if isinstance(data, dict):
            content = data.get("content")
            if isinstance(content, dict):
                for key in ("video_url", "videoUrl", "url", "video"):
                    value = content.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            nested_data = data.get("data")
            if isinstance(nested_data, dict):
                nested_url = self.extract_video_url(nested_data)
                if nested_url:
                    return nested_url
            for key in ("result_url", "video_url", "videoUrl", "url", "video"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        outputs = payload.get("outputs")
        if isinstance(outputs, list):
            for item in outputs:
                if not isinstance(item, dict):
                    continue
                for key in ("video_url", "videoUrl", "url"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return ""

    def extract_last_frame_url(self, payload):
        if not isinstance(payload, dict):
            return ""
        for container in (payload, payload.get("content"), payload.get("data"), payload.get("result")):
            if not isinstance(container, dict):
                continue
            for key in ("last_frame_url", "lastFrameUrl", "lastframe_url", "keyframe_url", "keyframeUrl"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def normalize_status(self, payload):
        raw = ""
        if isinstance(payload, dict):
            raw = payload.get("status", "") or payload.get("state", "") or ""
            if not raw and isinstance(payload.get("data"), dict):
                raw = payload["data"].get("status", "") or payload["data"].get("state", "") or ""
            if not raw and isinstance(payload.get("result"), dict):
                raw = payload["result"].get("status", "") or payload["result"].get("state", "") or ""
        normalized = str(raw or "").strip().lower()
        if normalized == "success":
            return "succeeded"
        if normalized in ("fail", "failure"):
            return "failed"
        return normalized

    async def wait_for_task(self, config, task_id, max_poll_attempts=120, poll_interval=10, unique_id=""):
        final_payload = None
        for _ in range(max(1, int(max_poll_attempts))):
            await asyncio.sleep(max(2, int(poll_interval)))
            try:
                payload = await self.async_request_json(
                    "GET",
                    self.task_status_url(config, task_id),
                    config,
                    headers=self.headers(config),
                )
            except Exception as exc:
                if self.is_transient_request_error(exc):
                    continue
                raise
            final_payload = payload
            progress = self.extract_progress(payload)
            status = self.normalize_status(payload)
            if unique_id:
                status_label = status or "processing"
                if self.is_success_status(status):
                    progress_text = "100%"
                elif progress:
                    progress_text = progress if str(progress).strip().endswith("%") else f"{progress}%"
                else:
                    progress_text = "处理中"
                self.publish_live_message(
                    unique_id,
                    f"Task ID: {task_id}\n进度: {progress_text}\n状态: {status_label}",
                )
            if status == "succeeded":
                return payload
            if status in ("failed", "cancelled", "canceled"):
                reason = ""
                if isinstance(payload, dict):
                    reason = payload.get("error", "") or payload.get("message", "") or ""
                    if not reason and isinstance(payload.get("data"), dict):
                        reason = payload["data"].get("error", "") or payload["data"].get("message", "") or ""
                raise RuntimeError(reason or f"Seedance official task ended with status: {status}")
        raise TimeoutError(f"Seedance official task polling timed out. task_id={task_id}")

    async def download_last_frame(self, url):
        if not url:
            return blank_image_tensor(width=1, height=1, color="black")
        image = await asyncio.to_thread(load_image_from_candidate, url, 120)
        if image is None:
            return blank_image_tensor(width=1, height=1, color="black")
        return pil2tensor(image)

    def extract_last_frame_from_video_url_sync(self, video_url):
        if not video_url:
            return blank_image_tensor(width=1, height=1, color="black")

        temp_path = None
        capture = None
        try:
            import cv2

            response = requests.get(video_url, timeout=300, stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=lingliu_temp_dir()) as handle:
                temp_path = handle.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

            capture = cv2.VideoCapture(temp_path)
            if not capture.isOpened():
                return blank_image_tensor(width=1, height=1, color="black")

            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame = None

            if frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
                success, frame = capture.read()
                if not success or frame is None:
                    frame = None

            if frame is None:
                while True:
                    success, current = capture.read()
                    if not success or current is None:
                        break
                    frame = current

            if frame is None:
                return blank_image_tensor(width=1, height=1, color="black")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            return pil2tensor(image)
        except Exception:
            return blank_image_tensor(width=1, height=1, color="black")
        finally:
            try:
                if capture is not None:
                    capture.release()
            except Exception:
                pass
            if temp_path:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

class LingliuR2StorageSet(LingliuSeedanceOfficialBase):
    @classmethod
    def INPUT_TYPES(cls):
        defaults = cls().load_r2_config()
        return {
            "required": {
                "R2账号ID": ("STRING", {"default": defaults.get("r2_account_id", "")}),
                "R2访问密钥ID": ("STRING", {"default": defaults.get("r2_access_key_id", "")}),
                # 密钥不预填，避免随工作流 JSON 导出；留空时沿用已保存的密钥。
                "R2访问密钥": ("STRING", {"default": "", "tooltip": "留空则沿用已保存的密钥"}),
                "存储桶名称": ("STRING", {"default": defaults.get("r2_bucket_name", "")}),
                "公网基础URL": ("STRING", {"default": defaults.get("r2_public_base_url", "")}),
            },
            "optional": {
                "对象前缀": ("STRING", {"default": defaults.get("r2_object_prefix", "seedance-assets")}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("状态",)
    FUNCTION = "save_r2_settings"
    CATEGORY = "Lingliu/Storage"

    def save_r2_settings(self, **kwargs):
        from ..lingliu_config import save_r2_config, set_runtime_config

        account_id = kwargs.get("R2账号ID", "")
        access_key_id = kwargs.get("R2访问密钥ID", "")
        secret_access_key = str(kwargs.get("R2访问密钥", "") or "").strip()
        if not secret_access_key:
            secret_access_key = self.load_r2_config().get("r2_secret_access_key", "")
        bucket_name = kwargs.get("存储桶名称", "")
        public_base_url = kwargs.get("公网基础URL", "")
        object_prefix = kwargs.get("对象前缀", "seedance-assets")
        set_runtime_config(
            r2_account_id=account_id,
            r2_access_key_id=access_key_id,
            r2_secret_access_key=secret_access_key,
            r2_bucket_name=bucket_name,
            r2_public_base_url=public_base_url,
            r2_object_prefix=object_prefix,
        )
        save_r2_config(
            {
                "r2_account_id": account_id,
                "r2_access_key_id": access_key_id,
                "r2_secret_access_key": secret_access_key,
                "r2_bucket_name": bucket_name,
                "r2_public_base_url": public_base_url,
                "r2_object_prefix": object_prefix,
            }
        )
        return ("已设置R2全局配置",)


class LingliuSeedance2Official(LingliuSeedanceOfficialBase):
    MODEL_CHOICES = [
        "doubao-seedance-2-0-260128",
        "doubao-seedance-2-0-fast-260128",
        "doubao-seedance-2-0-mini-260615",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "模型": (cls.MODEL_CHOICES, {"default": "doubao-seedance-2-0-260128"}),
                "时长": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1}),
                "比例": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "adaptive"], {"default": "16:9"}),
                "清晰度": (["1080p", "720p", "480p", "4K"], {"default": "720p"}),
            },
            "optional": {
                "API密钥": ("STRING", {"default": ""}),
                "生成音频": ("BOOLEAN", {"default": True}),
                "返回尾帧图": ("BOOLEAN", {"default": False}),
                "水印": ("BOOLEAN", {"default": True}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 2147483647, "control_after_generate": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 10, "max": 720}),
                "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60}),
                "首帧图": ("IMAGE",),
                "尾帧图": ("IMAGE",),
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "参考图4": ("IMAGE",),
                "参考图5": ("IMAGE",),
                "参考图6": ("IMAGE",),
                "参考图7": ("IMAGE",),
                "参考图8": ("IMAGE",),
                "参考图9": ("IMAGE",),
                "参考视频1": (IO.VIDEO, {"tooltip": "可直接连接本地参考视频；会自动上传为公网 URL 后生成。"}),
                "参考视频2": (IO.VIDEO, {"tooltip": "可直接连接本地参考视频；会自动上传为公网 URL 后生成。"}),
                "参考视频3": (IO.VIDEO, {"tooltip": "可直接连接本地参考视频；会自动上传为公网 URL 后生成。"}),
                "参考音频1": (IO.AUDIO, {"tooltip": "可直接连接本地参考音频；会自动上传为公网 URL 后生成。"}),
                "参考音频2": (IO.AUDIO, {"tooltip": "可直接连接本地参考音频；会自动上传为公网 URL 后生成。"}),
                "参考音频3": (IO.AUDIO, {"tooltip": "可直接连接本地参考音频；会自动上传为公网 URL 后生成。"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL", "尾帧图")
    FUNCTION = "generate_video"
    CATEGORY = "Lingliu/SeedanceOfficial"

    async def generate_video(self, **kwargs):
        blank_last_frame = blank_image_tensor(width=1, height=1, color="black")
        task_id = ""
        unique_id = kwargs.get("unique_id")
        try:
            prompt = _kwarg_alias(kwargs, "提示词", "prompt", default="", empty_string_is_missing=True)
            model = _kwarg_alias(
                kwargs,
                "模型",
                "model",
                default="doubao-seedance-2-0-260128",
                empty_string_is_missing=True,
            )
            duration = _kwarg_alias(kwargs, "时长", "duration", default=5)
            ratio = _kwarg_alias(kwargs, "比例", "ratio", default="16:9", empty_string_is_missing=True)
            resolution = _kwarg_alias(kwargs, "清晰度", "resolution", default="720p", empty_string_is_missing=True)
            local_media_mode = "public_url"
            generate_audio = kwargs.get("生成音频", kwargs.get("generate_audio", True))
            return_last_frame = kwargs.get("返回尾帧图", kwargs.get("return_last_frame", False))
            watermark = kwargs.get("水印", kwargs.get("watermark", True))
            seed = kwargs.get("随机种子", kwargs.get("seed", -1))
            wait_for_completion = kwargs.get("等待完成", kwargs.get("wait_for_completion", True))
            max_poll_attempts = kwargs.get("最大轮询次数", kwargs.get("max_poll_attempts", 120))
            poll_interval = kwargs.get("轮询间隔", kwargs.get("poll_interval", 10))
            first_frame = _kwarg_alias(kwargs, "首帧图", "first_frame")
            last_frame = _kwarg_alias(kwargs, "尾帧图", "last_frame")
            ref_image1 = _kwarg_alias(kwargs, "参考图1", "ref_image1")
            ref_image2 = _kwarg_alias(kwargs, "参考图2", "ref_image2")
            ref_image3 = _kwarg_alias(kwargs, "参考图3", "ref_image3")
            ref_image4 = _kwarg_alias(kwargs, "参考图4", "ref_image4")
            ref_image5 = _kwarg_alias(kwargs, "参考图5", "ref_image5")
            ref_image6 = _kwarg_alias(kwargs, "参考图6", "ref_image6")
            ref_image7 = _kwarg_alias(kwargs, "参考图7", "ref_image7")
            ref_image8 = _kwarg_alias(kwargs, "参考图8", "ref_image8")
            ref_image9 = _kwarg_alias(kwargs, "参考图9", "ref_image9")
            video1 = _kwarg_alias(kwargs, "参考视频1", "video1")
            video2 = _kwarg_alias(kwargs, "参考视频2", "video2")
            video3 = _kwarg_alias(kwargs, "参考视频3", "video3")
            audio1 = _kwarg_alias(kwargs, "参考音频1", "audio1")
            audio2 = _kwarg_alias(kwargs, "参考音频2", "audio2")
            audio3 = _kwarg_alias(kwargs, "参考音频3", "audio3")
            api_key = _kwarg_alias(kwargs, "API密钥", "api_key", "api密钥", default="", empty_string_is_missing=True)
            self.publish_task_id(unique_id, clear=True)
            config = self.load_lingliu_official_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "请先在 lingliu_api_set 中填写灵流 API key，或在本节点 API密钥 中填写。"
                return LingliuNullVideoAdapter(error), "", self.error_json(error), "", blank_last_frame

            prompt = str(prompt or "").strip()
            if not prompt:
                error = "Prompt is required for lingliu_seedance2_official."
                return LingliuNullVideoAdapter(error), "", self.error_json(error), "", blank_last_frame

            merged_first_frame = ""
            merged_last_frame = ""
            merged_ref_images = []
            merged_ref_videos = []
            merged_ref_audios = []
            reference_video_durations = []

            local_media_info = []
            if not merged_first_frame and first_frame is not None:
                merged_first_frame, info = await self.gateway_upload_local_media(config, "image", first_frame, "first_frame")
                info["slot"] = "first_frame"
                local_media_info.append(info)

            if not merged_last_frame and last_frame is not None:
                merged_last_frame, info = await self.gateway_upload_local_media(config, "image", last_frame, "last_frame")
                info["slot"] = "last_frame"
                local_media_info.append(info)

            for index, ref_image in enumerate([ref_image1, ref_image2, ref_image3, ref_image4, ref_image5, ref_image6, ref_image7, ref_image8, ref_image9], start=1):
                if ref_image is None:
                    continue
                value, info = await self.gateway_upload_local_media(config, "image", ref_image, f"ref_image{index}")
                merged_ref_images.append(value)
                info["slot"] = f"ref_image{index}"
                local_media_info.append(info)

            for index, ref_video in enumerate([video1, video2, video3], start=1):
                if ref_video is None:
                    continue
                value, info = await self.gateway_upload_local_media(config, "video", ref_video, f"video{index}")
                merged_ref_videos.append(value)
                duration_seconds = float(((info.get("bridge_upload") or {}).get("duration_seconds") or 0) or 0)
                if not duration_seconds:
                    duration_seconds = float((info.get("duration_seconds") or 0) or 0)
                if duration_seconds > 0:
                    reference_video_durations.append(duration_seconds)
                info["slot"] = f"video{index}"
                local_media_info.append(info)

            for index, ref_audio in enumerate([audio1, audio2, audio3], start=1):
                if ref_audio is None:
                    continue
                value, info = await self.gateway_upload_local_media(config, "audio", ref_audio, f"audio{index}")
                merged_ref_audios.append(value)
                info["slot"] = f"audio{index}"
                local_media_info.append(info)

            prompt_lines = [prompt]
            if merged_first_frame and merged_last_frame:
                prompt_lines.append("Use the first reference image as the opening frame and the second reference image as the ending frame.")
            elif merged_first_frame:
                prompt_lines.append("Use the first reference image as the opening frame.")
            elif merged_last_frame:
                prompt_lines.append("Use the first reference image as the ending frame.")

            if (merged_first_frame or merged_last_frame) and (merged_ref_images or merged_ref_videos or merged_ref_audios):
                error = (
                    "官方 Seedance 不允许首帧/尾帧和参考媒体混用。"
                    "请二选一：使用首帧图/尾帧图时不要连接参考图、参考视频或参考音频；"
                    "使用参考媒体时不要连接首帧图或尾帧图。"
                )
                return LingliuNullVideoAdapter(error), "", self.error_json(error), "", blank_last_frame

            for duration_seconds in reference_video_durations:
                if duration_seconds > 15.2:
                    error = (
                        f"参考视频时长超过官方限制：{duration_seconds:.3f}s，当前上限为 15.2s。"
                        "请将视频切到 14.8-14.9 秒，并开启严格时长。"
                    )
                    return LingliuNullVideoAdapter(error), "", self.error_json(error), "", blank_last_frame

            official_prompt = "\n".join([line for line in prompt_lines if str(line).strip()])
            image_urls = [item for item in [merged_first_frame, merged_last_frame, *merged_ref_images] if item]
            payload = self.build_official_video_payload(
                model=model,
                prompt=official_prompt,
                duration=duration,
                ratio=ratio,
                resolution=resolution,
                watermark=watermark,
                image_urls=image_urls,
                reference_video_seconds=round(sum(reference_video_durations), 3) if reference_video_durations else 0,
                seed=seed,
                generate_audio=generate_audio,
                reference_video_urls=merged_ref_videos,
                reference_audio_urls=merged_ref_audios,
            )

            submit_payload = await self.async_request_json_retrying(
                "POST",
                self.endpoint(config, "videos_generate_path"),
                config,
                # 付费创建只发一次：超时或 5xx 时任务可能已在上游生成，重发会重复扣费。
                max_retries=1,
                initial_timeout=self.timeout(config),
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            submit_data = submit_payload.get("data") if isinstance(submit_payload, dict) else None
            task_id = str(
                (submit_payload.get("task_id") if isinstance(submit_payload, dict) else "")
                or (submit_payload.get("id") if isinstance(submit_payload, dict) else "")
                or (submit_data.get("task_id") if isinstance(submit_data, dict) else "")
                or self.extract_task_id(submit_payload)
                or ""
            ).strip()
            if not task_id:
                raise RuntimeError(f"No task ID returned: {submit_payload}")
            self.publish_task_id(unique_id, task_id)
            self.publish_live_message(unique_id, f"Task ID: {task_id}\n进度: 已提交\n状态: submitted")

            response_info = {
                "provider": "lingliu_seedance_official",
                "route": "/v1/video/generations",
                "task_id": task_id,
                "model": model,
                "base_url": config.get("base_url", ""),
                "duration": int(duration),
                "ratio": ratio,
                "resolution": resolution,
                "generate_audio": bool(generate_audio),
                "watermark": bool(watermark),
                "seed": int(seed) if int(seed) >= 0 else "auto",
                "local_media_mode": local_media_mode,
                "submit": submit_payload,
            }
            if local_media_info:
                response_info["local_media_uploads"] = local_media_info

            if not wait_for_completion:
                response_info["status"] = "submitted"
                return LingliuNullVideoAdapter("Task submitted. Query later with task_id."), task_id, json.dumps(
                    response_info, ensure_ascii=False, indent=2
                ), "", blank_last_frame

            final_payload = await self.wait_for_task(
                config,
                task_id,
                max_poll_attempts=max_poll_attempts,
                poll_interval=poll_interval,
                unique_id=unique_id,
            )
            video_url = self.extract_video_url(final_payload)
            if not video_url:
                raise RuntimeError(f"No video URL in final payload: {final_payload}")
            response_info["final"] = final_payload
            response_info["status"] = self.extract_status(final_payload)
            video = LingliuVideoAdapter(video_url, mime_type="video/mp4")
            last_frame_image = blank_last_frame
            if return_last_frame:
                last_frame_url_value = self.extract_last_frame_url(final_payload)
                if last_frame_url_value:
                    last_frame_image = await self.download_last_frame(last_frame_url_value)
                    response_info["last_frame_url"] = last_frame_url_value
                else:
                    extracted_last_frame = await asyncio.to_thread(
                        self.extract_last_frame_from_video_url_sync,
                        video_url,
                    )
                    if extracted_last_frame is not None:
                        last_frame_image = extracted_last_frame
                        response_info["last_frame_source"] = "video_last_frame_extracted"
            return video, task_id, json.dumps(response_info, ensure_ascii=False, indent=2), video_url, last_frame_image
        except Exception as exc:
            self.publish_task_id(unique_id, clear=True)
            return LingliuNullVideoAdapter(str(exc)), task_id, self.error_json(exc, task_id=task_id), "", blank_last_frame

