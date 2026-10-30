import asyncio
import io
import json
import mimetypes
import os
import re
import tempfile
import time
import uuid
import wave
from collections.abc import Mapping
from io import BytesIO

import comfy.utils
import numpy as np
import requests
import torch
from PIL import Image
from comfy.comfy_types import IO

from ..upstream_mask import mask_upstream
from ..lingliu_config import LINGLIU_BASE_URL, build_url, lingliu_channel_key
from ..utils import LingliuNullVideoAdapter, LingliuVideoAdapter, blank_image_tensor, image_to_data_url, pil2tensor, tensor2pil
from .nodes_openai_compatible import LingliuBaseNode


def _official_seedance_model(value):
    if str(value or "").strip() == "doubao-seedance-2.0-mini":
        return "doubao-seedance-2-0-mini-260615"
    return value


def _official_seedance_resolution(value):
    if str(value or "").strip().lower() == "4k":
        return "4k"
    return value


def _split_asset_ids(value):
    if not value or not str(value).strip():
        return []
    return [part.strip() for part in re.split(r"[\n,]+", str(value)) if part.strip()]


def _asset_id_to_url(value):
    value = (value or "").strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "asset://")):
        return value
    return f"asset://{value}"


def _parse_asset_bundle_only(bundle_json):
    if not bundle_json or not str(bundle_json).strip():
        return "", "", "", "", ""
    try:
        payload = json.loads(str(bundle_json).strip())
    except Exception:
        return "", "", "", "", ""

    def join_list(key):
        value = payload.get(key)
        if isinstance(value, list):
            return ",".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, str):
            return value.strip()
        return ""

    return (
        str(payload.get("first_frame") or "").strip(),
        str(payload.get("last_frame") or "").strip(),
        join_list("ref_images"),
        join_list("videos"),
        join_list("audios"),
    )


def _probe_video_duration_seconds(file_bytes):
    if not file_bytes:
        return 0.0

    temp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
            temp_path = handle.name
            handle.write(file_bytes)

        try:
            import cv2

            capture = cv2.VideoCapture(temp_path)
            if capture.isOpened():
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
                frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps > 0 and frame_count > 0:
                    return max(0.0, frame_count / fps)
        except Exception:
            pass
        finally:
            try:
                if capture is not None:
                    capture.release()
            except Exception:
                pass

        try:
            import av

            container = av.open(temp_path)
            try:
                stream = next((item for item in container.streams if item.type == "video"), None)
                if stream is not None and stream.duration is not None and stream.time_base is not None:
                    return max(0.0, float(stream.duration * stream.time_base))
            finally:
                container.close()
        except Exception:
            return 0.0
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return 0.0


def _extract_from_payload(payload, candidate_keys):
    normalized_keys = {str(key).strip().lower() for key in candidate_keys if str(key).strip()}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).strip().lower() in normalized_keys:
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    return value
            nested = _extract_from_payload(value, candidate_keys)
            if nested:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _extract_from_payload(item, candidate_keys)
            if nested:
                return nested
    return ""


def _normalize_url_candidate(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        candidate = value.get("url") or value.get("image_url") or value.get("imageUrl") or ""
        if isinstance(candidate, str):
            return candidate.strip()
    return ""


def _extract_last_frame_from_results(results):
    if not isinstance(results, list):
        return ""
    for item in results:
        if not isinstance(item, dict):
            continue
        candidate = (
            item.get("url")
            or item.get("image_url")
            or item.get("imageUrl")
            or ""
        )
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        output_type = str(item.get("outputType") or item.get("type") or "").strip().lower()
        role = str(item.get("role") or item.get("name") or item.get("label") or "").strip().lower()
        if any(token in role for token in ("last_frame", "lastframe", "keyframe", "tail")):
            return candidate.strip()
        if output_type and any(token in output_type for token in ("png", "jpg", "jpeg", "image")):
            return candidate.strip()
    return ""


def _waveform_to_wav_bytes(waveform, sample_rate):
    wf = waveform.detach().cpu().float()
    if wf.dim() == 3:
        wf = wf.squeeze(0)
    if wf.dim() == 1:
        wf = wf.unsqueeze(0)
    if wf.dim() != 2:
        raise ValueError(f"Expected waveform [C, T], got shape {tuple(wf.shape)}")
    channels, _ = wf.shape
    wf = wf.clamp(-1.0, 1.0)
    pcm = (wf.numpy() * 32767.0).astype(np.int16)
    interleaved = pcm[0] if channels == 1 else np.transpose(pcm, (1, 0)).reshape(-1)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_handle:
        wav_handle.setnchannels(int(channels))
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(int(sample_rate))
        wav_handle.writeframes(interleaved.tobytes())
    return buffer.getvalue()


def _media_input_to_bytes(media_input, bytesio_ext=".mp4", label="media"):
    if media_input is None:
        return None, None

    # VIDEO objects such as Video Slice may expose the original stream source;
    # save_to materializes the processed clip that is actually wired in ComfyUI.
    if str(bytesio_ext).lower() == ".mp4" and hasattr(media_input, "save_to"):
        buffer = BytesIO()
        try:
            media_input.save_to(buffer)
            if hasattr(buffer, "seek"):
                buffer.seek(0)
            data = buffer.read()
            if data:
                return data, f"reference_{label}_{abs(hash(data)) % 10**10}{bytesio_ext}"
        except Exception:
            pass

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=bytesio_ext) as handle:
                tmp_path = handle.name
            media_input.save_to(tmp_path)
            if tmp_path and os.path.isfile(tmp_path):
                with open(tmp_path, "rb") as handle:
                    data = handle.read()
                if data:
                    return data, os.path.basename(tmp_path)
        except Exception:
            pass
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    get_stream = getattr(media_input, "get_stream_source", None)
    if callable(get_stream):
        try:
            source = media_input.get_stream_source()
            if isinstance(source, str):
                source = source.strip()
                if source and os.path.isfile(source):
                    with open(source, "rb") as handle:
                        return handle.read(), os.path.basename(source)
                return None, None
            if isinstance(source, BytesIO):
                source.seek(0)
                data = source.read()
                if data:
                    return data, f"reference_{label}_{abs(hash(data)) % 10**10}{bytesio_ext}"
                return None, None
            if hasattr(source, "read"):
                if hasattr(source, "seek"):
                    source.seek(0)
                data = source.read()
                if data:
                    return data, f"reference_{label}_{abs(hash(data)) % 10**10}{bytesio_ext}"
                return None, None
        except Exception:
            pass

    if isinstance(media_input, str):
        path = media_input.strip()
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                return handle.read(), os.path.basename(path)
        if path.lower().startswith(("http://", "https://")):
            response = requests.get(path, timeout=120)
            response.raise_for_status()
            filename = os.path.basename(path.split("?", 1)[0]) or f"reference_{label}{bytesio_ext}"
            return response.content, filename
        return None, None

    if isinstance(media_input, Mapping):
        path = (
            media_input.get("path")
            or media_input.get("file")
            or media_input.get("file_path")
            or media_input.get("filename")
            or ""
        )
        path = str(path).strip() if path else ""
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                return handle.read(), os.path.basename(path)

        remote_url = (
            media_input.get("url")
            or media_input.get("video_url")
            or media_input.get("audio_url")
            or media_input.get("source")
            or media_input.get("src")
            or ""
        )
        remote_url = str(remote_url).strip() if remote_url else ""
        if remote_url.lower().startswith(("http://", "https://")):
            response = requests.get(remote_url, timeout=120)
            response.raise_for_status()
            filename = os.path.basename(remote_url.split("?", 1)[0]) or f"reference_{label}{bytesio_ext}"
            return response.content, filename
        return None, None

    for attr_name in ("path", "file_path"):
        path = getattr(media_input, attr_name, None)
        if path and isinstance(path, str):
            path = path.strip()
            if path and os.path.isfile(path):
                with open(path, "rb") as handle:
                    return handle.read(), os.path.basename(path)

    for attr_name in ("url", "video_url", "audio_url", "source", "src"):
        remote_url = getattr(media_input, attr_name, None)
        if remote_url and isinstance(remote_url, str):
            remote_url = remote_url.strip()
            if remote_url.lower().startswith(("http://", "https://")):
                response = requests.get(remote_url, timeout=120)
                response.raise_for_status()
                filename = os.path.basename(remote_url.split("?", 1)[0]) or f"reference_{label}{bytesio_ext}"
                return response.content, filename

    return None, None


class LingliuDoubaoBase(LingliuBaseNode):
    CATEGORY = "Lingliu/Doubao"
    SEEDANCE_TASK_POLL_INTERVAL = 10
    SEEDANCE_TASK_MAX_WAIT = 600
    ASSET_POLL_INTERVAL = 3
    ASSET_MAX_WAIT = 300

    def auth_headers(self, config, json_mode=False):
        request_config = dict(config or {})
        request_config["api_key"] = lingliu_channel_key(request_config.get("api_key", ""), 6)
        headers = self.headers(request_config)
        if json_mode:
            headers["Content-Type"] = "application/json"
        return headers

    def doubao_endpoint(self, path):
        return build_url(LINGLIU_BASE_URL, path)

    def build_compat_video_payload(
        self,
        *,
        model,
        content,
        duration,
        ratio,
        resolution,
        generate_audio,
        return_last_frame,
        watermark,
        reference_video_seconds=0,
        seed=-1,
        web_search=False,
    ):
        duration_int = int(duration)
        payload = {
            "model": model,
            "content": content,
            "duration": duration_int,
            "seconds": str(duration_int),
            "ratio": ratio,
            "resolution": resolution,
            "generate_audio": bool(generate_audio),
            "return_last_frame": bool(return_last_frame),
            "watermark": bool(watermark),
        }
        if reference_video_seconds:
            payload["reference_video_seconds"] = round(float(reference_video_seconds), 3)
        if web_search:
            payload["tools"] = [{"type": "web_search"}]
        if int(seed) >= 0:
            payload["seed"] = int(seed)
        return payload

    def build_lingliu_official_fallback_payload(
        self,
        *,
        model,
        content,
        duration,
        ratio,
        resolution,
        generate_audio,
        return_last_frame,
        watermark,
        reference_video_seconds=0,
        seed=-1,
    ):
        duration_int = int(duration)
        official_model = _official_seedance_model(model)
        official_resolution = _official_seedance_resolution(resolution)
        prompt = ""
        image_urls = []
        for item in content or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and str(item.get("text") or "").strip():
                prompt = str(item.get("text") or "").strip()
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "").strip()
                if url:
                    image_urls.append(url)
        metadata = {
            "resolution": official_resolution,
            "ratio": ratio,
            "watermark": bool(watermark),
            "generate_audio": bool(generate_audio),
            "return_last_frame": bool(return_last_frame),
            "duration": duration_int,
            "content": content or [],
        }
        if reference_video_seconds:
            metadata["reference_video_seconds"] = round(float(reference_video_seconds), 3)
        payload = {
            "model": official_model,
            "prompt": prompt,
            "seconds": str(duration_int),
            "duration": duration_int,
            "resolution": official_resolution,
            "metadata": metadata,
        }
        if image_urls:
            payload["images"] = image_urls
        if int(seed) >= 0:
            payload["seed"] = int(seed)
        return payload

    def should_fallback_to_lingliu_official(self, exc):
        text = str(exc or "").lower()
        return "channel_selection_failed" in text and "上游" in mask_upstream(text)

    def upload_file_request(self, config, file_content, filename, content_type, timeout=None):
        response = requests.post(
            self.doubao_endpoint("/v1/files"),
            headers={"Authorization": f"Bearer {lingliu_channel_key(config.get('api_key', ''), 6)}"},
            files={"file": (filename, file_content, content_type)},
            timeout=timeout or self.timeout(config),
        )
        response.raise_for_status()
        result = response.json()
        return result.get("url", "")

    async def async_upload_file(self, config, file_content, filename, content_type):
        last_error = None
        base_timeout = self.timeout(config)
        for attempt in range(1, 6):
            current_timeout = min(int(base_timeout * (1.5 ** (attempt - 1))), 300)
            try:
                return await asyncio.to_thread(
                    self.upload_file_request,
                    config,
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
        raise last_error or RuntimeError("Upload failed without a concrete error.")

    async def upload_image_get_url(self, config, image_tensor):
        if image_tensor is None:
            return ""
        pil_image = tensor2pil(image_tensor)[0]
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        file_content = buffer.getvalue()
        filename = f"upload_{abs(hash(file_content)) % 10**10}.png"
        return await self.async_upload_file(config, file_content, filename, "image/png")

    async def upload_video_get_url(self, config, video_input):
        url, _duration_seconds = await self.upload_video_get_url_with_duration(config, video_input)
        return url

    async def upload_video_get_url_with_duration(self, config, video_input):
        file_content, filename = await asyncio.to_thread(_media_input_to_bytes, video_input, ".mp4", "video")
        if not file_content:
            return "", 0.0
        duration_seconds = await asyncio.to_thread(_probe_video_duration_seconds, file_content)
        if not filename:
            filename = f"reference_video_{abs(hash(file_content)) % 10**10}.mp4"
        mime_type = mimetypes.guess_type(filename)[0] or "video/mp4"
        return await self.async_upload_file(config, file_content, filename, mime_type), duration_seconds

    async def upload_audio_get_url(self, config, audio_input):
        if audio_input is None:
            return ""

        file_content, filename = await asyncio.to_thread(_media_input_to_bytes, audio_input, ".wav", "audio")
        if file_content:
            if not filename:
                filename = f"reference_audio_{abs(hash(file_content)) % 10**10}.wav"
            mime_type = mimetypes.guess_type(filename)[0] or "audio/wav"
            return await self.async_upload_file(config, file_content, filename, mime_type)

        if isinstance(audio_input, Mapping) and audio_input.get("waveform") is not None:
            waveform = audio_input["waveform"]
            if torch.is_tensor(waveform):
                sample_rate = int(audio_input.get("sample_rate", 44100))
                file_content = await asyncio.to_thread(_waveform_to_wav_bytes, waveform, sample_rate)
                return await self.async_upload_file(config, file_content, "audio.wav", "audio/wav")

        return ""

    async def download_image_tensor(self, url):
        response = await asyncio.to_thread(requests.get, url, timeout=60)
        response.raise_for_status()
        return pil2tensor(Image.open(BytesIO(response.content)).convert("RGB"))

    def extract_last_frame_from_video_url_sync(self, video_url):
        if not video_url:
            return blank_image_tensor(width=1, height=1, color="black")

        temp_path = None
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

            import cv2

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return pil2tensor(Image.fromarray(frame_rgb))
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


class LingliuDoubaoSeedance2(LingliuDoubaoBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (
                    [
                        "doubao-seedance-2-0-260128",
                        "doubao-seedance-2-0-fast-260128",
                        "doubao-seedance-2.0-mini",
                    ],
                    {"default": "doubao-seedance-2-0-260128"},
                ),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1}),
                "ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "adaptive"], {"default": "16:9"}),
                "resolution": (["720p", "480p", "1080p", "4K"], {"default": "720p"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "ref_image1": ("IMAGE",),
                "ref_image2": ("IMAGE",),
                "ref_image3": ("IMAGE",),
                "ref_image4": ("IMAGE",),
                "ref_image5": ("IMAGE",),
                "ref_image6": ("IMAGE",),
                "ref_image7": ("IMAGE",),
                "ref_image8": ("IMAGE",),
                "ref_image9": ("IMAGE",),
                "video1": (IO.VIDEO, {"tooltip": "Reference video input."}),
                "video2": (IO.VIDEO, {"tooltip": "Reference video input."}),
                "video3": (IO.VIDEO, {"tooltip": "Reference video input."}),
                "audio1": (IO.AUDIO, {"tooltip": "Reference audio input."}),
                "audio2": (IO.AUDIO, {"tooltip": "Reference audio input."}),
                "audio3": (IO.AUDIO, {"tooltip": "Reference audio input."}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "web_search": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "control_after_generate": True}),
                "asset_bundle": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Wire from lingliu_seedance2_asset_bundle output.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "task_id", "response", "video_url", "last_frame_image")
    FUNCTION = "generate_video"
    CATEGORY = "Lingliu/Doubao"

    def _blank_last_frame(self):
        return blank_image_tensor(width=1, height=1, color="black")

    async def generate_video(
        self,
        prompt,
        model,
        duration,
        ratio,
        resolution,
        api_key="",
        first_frame=None,
        last_frame=None,
        ref_image1=None,
        ref_image2=None,
        ref_image3=None,
        ref_image4=None,
        ref_image5=None,
        ref_image6=None,
        ref_image7=None,
        ref_image8=None,
        ref_image9=None,
        video1=None,
        video2=None,
        video3=None,
        audio1=None,
        audio2=None,
        audio3=None,
        generate_audio=True,
        return_last_frame=False,
        web_search=False,
        watermark=False,
        seed=-1,
        asset_bundle="",
        unique_id=None,
    ):
        blank_tensor = self._blank_last_frame()
        task_id = ""
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_seedance2."
                return LingliuNullVideoAdapter(error), "", self.error_json(error), "", blank_tensor

            asset_id_first_frame, asset_id_last_frame, asset_ids_ref_images, asset_ids_ref_videos, asset_ids_ref_audios = (
                _parse_asset_bundle_only(asset_bundle)
            )

            content = [{"type": "text", "text": prompt}]
            frame_count = 0
            ref_count = 0
            video_count = 0
            audio_count = 0

            has_first_tensor = first_frame is not None
            has_last_tensor = last_frame is not None

            if has_first_tensor:
                entry = {"type": "image_url", "image_url": {"url": image_to_data_url(first_frame)}}
                if has_last_tensor:
                    entry["role"] = "first_frame"
                content.append(entry)
                frame_count += 1

            if not has_first_tensor:
                first_frame_url = _asset_id_to_url(asset_id_first_frame)
                if first_frame_url:
                    content.append({"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"})
                    frame_count += 1

            has_first_effective = has_first_tensor or bool(_asset_id_to_url(asset_id_first_frame))

            if has_last_tensor and has_first_effective:
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(last_frame)}, "role": "last_frame"})
                frame_count += 1

            if not has_last_tensor and has_first_effective:
                last_frame_url_from_asset = _asset_id_to_url(asset_id_last_frame)
                if last_frame_url_from_asset:
                    content.append(
                        {"type": "image_url", "image_url": {"url": last_frame_url_from_asset}, "role": "last_frame"}
                    )
                    frame_count += 1

            ref_images = [
                ref_image1,
                ref_image2,
                ref_image3,
                ref_image4,
                ref_image5,
                ref_image6,
                ref_image7,
                ref_image8,
                ref_image9,
            ]
            for image in ref_images:
                if image is None:
                    continue
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}, "role": "reference_image"})
                ref_count += 1

            for asset_id in _split_asset_ids(asset_ids_ref_images):
                asset_url = _asset_id_to_url(asset_id)
                if not asset_url:
                    continue
                content.append({"type": "image_url", "image_url": {"url": asset_url}, "role": "reference_image"})
                ref_count += 1

            reference_video_seconds = 0.0
            video_inputs = [(index, video) for index, video in enumerate([video1, video2, video3], start=1) if video is not None]
            upload_tasks = [self.upload_video_get_url_with_duration(config, video) for _, video in video_inputs]
            uploaded_video_urls = await asyncio.gather(*upload_tasks, return_exceptions=True) if upload_tasks else []
            for (index, _video), result in zip(video_inputs, uploaded_video_urls):
                if isinstance(result, Exception):
                    raise RuntimeError(f"参考视频{index}上传失败: {result}") from result
                if not result:
                    raise RuntimeError(f"参考视频{index}上传失败: 未获得文件 URL。")
                video_url, duration_seconds = result
                if not video_url:
                    raise RuntimeError(f"参考视频{index}上传失败: 未获得文件 URL。")
                content.append({"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"})
                if duration_seconds and duration_seconds > 0:
                    reference_video_seconds += float(duration_seconds)
                video_count += 1

            for asset_id in _split_asset_ids(asset_ids_ref_videos):
                asset_url = _asset_id_to_url(asset_id)
                if not asset_url:
                    continue
                content.append({"type": "video_url", "video_url": {"url": asset_url}, "role": "reference_video"})
                video_count += 1

            audio_inputs = [(index, audio) for index, audio in enumerate([audio1, audio2, audio3], start=1) if audio is not None]
            audio_tasks = [self.upload_audio_get_url(config, audio) for _, audio in audio_inputs]
            uploaded_audio_urls = await asyncio.gather(*audio_tasks, return_exceptions=True) if audio_tasks else []
            for (index, _audio), result in zip(audio_inputs, uploaded_audio_urls):
                if isinstance(result, Exception):
                    raise RuntimeError(f"参考音频{index}上传失败: {result}") from result
                if not result:
                    raise RuntimeError(f"参考音频{index}上传失败: 未获得文件 URL。")
                content.append({"type": "audio_url", "audio_url": {"url": result}, "role": "reference_audio"})
                audio_count += 1

            for asset_id in _split_asset_ids(asset_ids_ref_audios):
                asset_url = _asset_id_to_url(asset_id)
                if not asset_url:
                    continue
                content.append({"type": "audio_url", "audio_url": {"url": asset_url}, "role": "reference_audio"})
                audio_count += 1

            payload = self.build_compat_video_payload(
                model=model,
                content=content,
                duration=duration,
                ratio=ratio,
                resolution=resolution,
                generate_audio=generate_audio,
                return_last_frame=return_last_frame,
                watermark=watermark,
                reference_video_seconds=reference_video_seconds,
                seed=seed,
                web_search=web_search,
            )

            fallback_used = False
            fallback_config = dict(config or {})
            fallback_config["api_key"] = lingliu_channel_key(fallback_config.get("api_key", ""), 5)
            try:
                submit_payload = await self.async_request_json(
                    "POST",
                    self.doubao_endpoint("/seedance/v3/contents/generations/tasks"),
                    config,
                    headers=self.auth_headers(config, json_mode=True),
                    json=payload,
                )
            except Exception as exc:
                if not self.should_fallback_to_lingliu_official(exc):
                    raise
                fallback_used = True
                fallback_payload = self.build_lingliu_official_fallback_payload(
                    model=model,
                    content=content,
                    duration=duration,
                    ratio=ratio,
                    resolution=resolution,
                    generate_audio=generate_audio,
                    return_last_frame=return_last_frame,
                    watermark=watermark,
                    reference_video_seconds=reference_video_seconds,
                    seed=seed,
                )
                submit_payload = await self.async_request_json(
                    "POST",
                    self.doubao_endpoint("/v1/video/generations"),
                    fallback_config,
                    headers=self.headers(fallback_config, {"Content-Type": "application/json"}),
                    json=fallback_payload,
                )

            task_id = submit_payload.get("id", "") or submit_payload.get("task_id", "") or ""
            if not task_id:
                raise RuntimeError(f"No task ID returned: {submit_payload}")
            self.publish_task_id(unique_id, task_id)

            start_time = time.time()
            final_status_data = None
            video_url = ""
            last_frame_url = ""

            while True:
                elapsed = time.time() - start_time
                if elapsed > self.SEEDANCE_TASK_MAX_WAIT:
                    raise TimeoutError(f"Seedance task timed out after {elapsed:.1f}s")

                await asyncio.sleep(self.SEEDANCE_TASK_POLL_INTERVAL)

                try:
                    if fallback_used:
                        status_data = await self.async_request_json(
                            "GET",
                            self.doubao_endpoint(f"/v1/video/generations/{task_id}"),
                            fallback_config,
                            headers=self.headers(fallback_config),
                        )
                    else:
                        status_data = await self.async_request_json(
                            "GET",
                            self.doubao_endpoint(f"/seedance/v3/contents/generations/tasks/{task_id}"),
                            config,
                            headers={"Authorization": f"Bearer {lingliu_channel_key(config.get('api_key', ''), 6)}"},
                        )
                except Exception as exc:
                    if self.is_transient_request_error(exc):
                        continue
                    raise

                final_status_data = status_data
                data_payload = status_data.get("data") if isinstance(status_data.get("data"), dict) else {}
                raw_status = str(status_data.get("status", "") or data_payload.get("status", "")).lower()
                if raw_status == "success":
                    raw_status = "succeeded"
                elif raw_status in ("fail", "failure"):
                    raw_status = "failed"
                elif raw_status == "completed":
                    raw_status = "succeeded"

                if raw_status == "succeeded":
                    video_url = (
                        status_data.get("result_url")
                        or status_data.get("video_url")
                        or _extract_from_payload(status_data, ("metadata.url", "metadata.video_url", "content.video_url"))
                        or data_payload.get("result_url")
                        or data_payload.get("video_url")
                        or _extract_from_payload(data_payload, ("content.video_url", "metadata.url"))
                        or ""
                    )
                    root_content = status_data.get("content")
                    if isinstance(root_content, dict) and not video_url:
                        video_url = root_content.get("video_url") or root_content.get("videoUrl") or ""

                    data = status_data.get("data")
                    if isinstance(data, dict):
                        data_content = data.get("content")
                        if isinstance(data_content, dict) and not video_url:
                            video_url = data_content.get("video_url") or data_content.get("videoUrl") or ""
                        if not video_url:
                            video_url = data.get("video_url") or data.get("videoUrl") or ""

                    if not video_url:
                        results = status_data.get("results", [])
                        if isinstance(results, list):
                            for item in results:
                                if not isinstance(item, dict):
                                    continue
                                candidate = item.get("url", "")
                                output_type = item.get("outputType", "")
                                if output_type in ("mp4", "video") or candidate.endswith(".mp4"):
                                    video_url = candidate
                                    break
                                if candidate and not video_url:
                                    video_url = candidate

                    if return_last_frame:
                        last_frame_url = _normalize_url_candidate(
                            _extract_from_payload(
                            status_data,
                            (
                                "last_frame_url",
                                "lastFrameUrl",
                                "last_frame_image_url",
                                "last_frame_image",
                                "lastframe_url",
                                "keyframe_url",
                                "keyframeUrl",
                            ),
                            )
                        )
                        if not last_frame_url:
                            last_frame_value = _extract_from_payload(status_data, ("last_frame", "lastFrame"))
                            last_frame_url = _normalize_url_candidate(last_frame_value)
                        if not last_frame_url:
                            last_frame_url = _extract_last_frame_from_results(status_data.get("results", []))

                    if video_url:
                        break
                    raise RuntimeError(f"Task succeeded but no video URL found: {status_data}")

                if raw_status == "failed":
                    fail_reason = status_data.get("fail_reason", "") or status_data.get("failReason", "") or "Unknown error"
                    raise RuntimeError(f"Seedance task failed: {fail_reason}")

            last_frame_tensor = blank_tensor
            last_frame_source = ""
            if return_last_frame:
                last_frame_loaded = False
                if last_frame_url:
                    try:
                        last_frame_tensor = await self.download_image_tensor(last_frame_url)
                        last_frame_loaded = True
                        last_frame_source = "last_frame_url"
                    except Exception:
                        last_frame_loaded = False
                if not last_frame_loaded and video_url:
                    try:
                        last_frame_tensor = await asyncio.to_thread(self.extract_last_frame_from_video_url_sync, video_url)
                        last_frame_source = "video_last_frame_extracted"
                    except Exception:
                        last_frame_tensor = blank_tensor

            response_info = {
                "task_id": task_id,
                "model": model,
                "status": "succeeded",
                "video_url": video_url,
                "duration": duration,
                "ratio": ratio,
                "resolution": resolution,
                "generate_audio": generate_audio,
                "return_last_frame": return_last_frame,
                "seed": seed if int(seed) >= 0 else "auto",
                "first_frame": has_first_effective,
                "last_frame_input": has_first_effective and (has_last_tensor or bool(_asset_id_to_url(asset_id_last_frame))),
                "reference_images": ref_count,
                "reference_videos": video_count,
                "reference_audios": audio_count,
                "submit": submit_payload,
                "final": final_status_data,
            }
            if last_frame_url:
                response_info["last_frame_image_url"] = last_frame_url
            if last_frame_source:
                response_info["last_frame_source"] = last_frame_source

            return (
                LingliuVideoAdapter(video_url),
                task_id,
                json.dumps(response_info, ensure_ascii=False, indent=2),
                video_url,
                last_frame_tensor,
            )
        except Exception as exc:
            return (
                LingliuNullVideoAdapter(str(exc)),
                task_id,
                self.error_json(exc, task_id=task_id),
                "",
                blank_tensor,
            )


class LingliuSeedance2AssetBundle(LingliuDoubaoBase):
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "first_frame": ("STRING", {"default": "", "tooltip": "asset_id from lingliu_seedance2_asset_upload"}),
            "last_frame": ("STRING", {"default": "", "tooltip": "asset_id from lingliu_seedance2_asset_upload"}),
        }
        for index in range(1, 10):
            optional[f"ref_image{index}"] = ("STRING", {"default": "", "tooltip": "asset_id from lingliu_seedance2_asset_upload"})
        for index in range(1, 4):
            optional[f"video{index}"] = ("STRING", {"default": "", "tooltip": "asset_id from lingliu_seedance2_asset_upload"})
        for index in range(1, 4):
            optional[f"audio{index}"] = ("STRING", {"default": "", "tooltip": "asset_id from lingliu_seedance2_asset_upload"})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("asset_bundle",)
    FUNCTION = "bundle"
    CATEGORY = "Lingliu/Doubao"

    def bundle(
        self,
        first_frame="",
        last_frame="",
        ref_image1="",
        ref_image2="",
        ref_image3="",
        ref_image4="",
        ref_image5="",
        ref_image6="",
        ref_image7="",
        ref_image8="",
        ref_image9="",
        video1="",
        video2="",
        video3="",
        audio1="",
        audio2="",
        audio3="",
    ):
        payload = {
            "first_frame": (first_frame or "").strip(),
            "last_frame": (last_frame or "").strip(),
            "ref_images": [],
            "videos": [],
            "audios": [],
        }
        for index in range(1, 10):
            value = (locals().get(f"ref_image{index}") or "").strip()
            if value:
                payload["ref_images"].append(value)
        for index in range(1, 4):
            value = (locals().get(f"video{index}") or "").strip()
            if value:
                payload["videos"].append(value)
        for index in range(1, 4):
            value = (locals().get(f"audio{index}") or "").strip()
            if value:
                payload["audios"].append(value)
        return (json.dumps(payload, ensure_ascii=False),)


class LingliuSeedance2AssetUpload(LingliuDoubaoBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "video": (IO.VIDEO, {"tooltip": "Reference video for Seedance asset upload."}),
                "audio": (IO.AUDIO, {"tooltip": "Reference audio for Seedance asset upload."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("asset_id", "status", "response")
    FUNCTION = "upload_asset"
    CATEGORY = "Lingliu/Doubao"

    async def query_asset_status(self, config, asset_id):
        return await self.async_request_json_retrying(
            "POST",
            self.doubao_endpoint("/seedance/v3/assets/query"),
            config,
            headers=self.auth_headers(config, json_mode=True),
            json={"assetId": asset_id},
        )

    async def upload_asset(self, api_key="", name="", image=None, video=None, audio=None):
        try:
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_seedance2_asset_upload."
                return "", "", self.error_json(error)

            asset_type = ""
            media_url = ""
            if image is not None:
                asset_type = "Image"
                media_url = await self.upload_image_get_url(config, image)
            elif video is not None:
                asset_type = "Video"
                media_url = await self.upload_video_get_url(config, video)
            elif audio is not None:
                asset_type = "Audio"
                media_url = await self.upload_audio_get_url(config, audio)
            else:
                raise RuntimeError("Connect image, video, or audio.")

            if not media_url:
                debug_shape = ""
                if video is not None:
                    try:
                        if isinstance(video, dict):
                            debug_shape = f" video_keys={sorted(video.keys())}"
                        else:
                            debug_shape = f" video_type={type(video).__name__}"
                    except Exception:
                        debug_shape = ""
                raise RuntimeError(f"Could not obtain HTTPS URL for asset.{debug_shape}")

            payload = {
                "url": media_url,
                "assetType": asset_type,
                "name": (name or "").strip() or f"asset_{uuid.uuid4().hex[:12]}",
            }
            result = await self.async_request_json_retrying(
                "POST",
                self.doubao_endpoint("/seedance/v3/assets/create"),
                config,
                max_retries=5,
                initial_timeout=self.timeout(config),
                headers=self.auth_headers(config, json_mode=True),
                json=payload,
            )
            if result.get("code") not in (0, None):
                return "", "", json.dumps(result, ensure_ascii=False, indent=2)

            data = result.get("data", {}) if isinstance(result, dict) else {}
            asset_id = data.get("assetId", "")
            status = data.get("status", "")
            if not asset_id:
                raise RuntimeError(f"No asset_id returned: {result}")

            if status == "Active":
                return asset_id, status, json.dumps(result, ensure_ascii=False, indent=2)

            start_time = time.time()
            last_query_result = result
            while True:
                elapsed = time.time() - start_time
                if elapsed > self.ASSET_MAX_WAIT:
                    return (
                        "",
                        status,
                        json.dumps(
                            {"error": f"Asset processing timeout after {elapsed:.1f}s", "asset_id": asset_id, "last_status": status},
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )

                await asyncio.sleep(self.ASSET_POLL_INTERVAL)
                try:
                    query_result = await self.query_asset_status(config, asset_id)
                except Exception as exc:
                    if self.is_transient_request_error(exc):
                        continue
                    raise

                last_query_result = query_result
                if query_result.get("code") not in (0, None):
                    continue
                query_data = query_result.get("data", {})
                status = query_data.get("status", "")
                if status == "Active":
                    return asset_id, status, json.dumps(query_result, ensure_ascii=False, indent=2)
                if status in ("Failed", "Error", "Deleted"):
                    return "", status, json.dumps(query_result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return "", "", self.error_json(exc)
