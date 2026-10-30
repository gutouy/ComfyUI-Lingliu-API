"""Dedicated node for the OpenAI-compatible Grok Imagine Video 1.5 channel."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from urllib.parse import urlparse

import requests
from comfy.comfy_types import IO
from PIL import Image

from ..utils import (
    LingliuNullVideoAdapter,
    LingliuVideoAdapter,
    image_to_png_bytes,
    save_video_natively,
    tensor2pil,
    video_save_options,
)
from .nodes_grok import LingliuGrokBase


class _GrokOpenAIVideoAdapter(LingliuVideoAdapter):
    """Stream a completed video while only authenticating trusted platform hosts."""

    TRUSTED_AUTH_HOSTS = {
        "localhost",
        "127.0.0.1",
        "storage.deepwl.cn",
        "api.lingliuapi.com",
    }

    def __init__(self, video_url, api_key):
        super().__init__(video_url, mime_type="video/mp4")
        parsed = urlparse(str(video_url or ""))
        hostname = (parsed.hostname or "").lower()
        use_auth = hostname in self.TRUSTED_AUTH_HOSTS or hostname.endswith(".lingliuapi.com")
        self._headers = {
            "Authorization": f"Bearer {str(api_key or '').strip()}"
        } if use_auth and str(api_key or "").strip() else {}

    def save_to(self, output, format="auto", codec="auto", metadata=None,
                bit_depth=None, crf=None, color_space=None, preset=None, **kwargs):
        del metadata, kwargs
        options = video_save_options(output, format, codec, bit_depth, crf, color_space, preset)
        if options is not None:
            return save_video_natively(self.save_to, output, options)
        if not self.video_url:
            return super().save_to(output)
        with requests.get(self.video_url, headers=self._headers, stream=True, timeout=300) as response:
            response.raise_for_status()
            if hasattr(output, "write"):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)
                if hasattr(output, "seek"):
                    output.seek(0)
                return True
            with open(output, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        return True


class LingliuGrokImagineVideo15OpenAI(LingliuGrokBase):
    """Generate Grok Imagine Video 1.5 through platform channel 39.

    The channel is called OpenAI-compatible by the platform, but its accepted
    contract is the platform-tested JSON variant: ``seconds`` plus
    ``aspect_ratio``/``resolution``, with URL strings in ``image`` or ``images``.
    The older ``lingliu_grok_imagine_video_1_5`` node remains on channel 37 and
    its native X.AI route.
    """

    # Public catalog alias for channel 39. The platform maps this alias to
    # the original upstream model name before forwarding to xAI.
    MODEL = "grok-imagine-video-1.5-按次计费"
    ROUTE_CHANNEL_ID = 39
    CATEGORY = "Lingliu/Grok"
    MODES = ["文生视频", "首帧图生视频", "参考生视频"]
    RATIOS = ["默认", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
    RESOLUTIONS = ["480p", "720p", "1080p"]
    MAX_REFERENCE_IMAGES = 7
    GENERATE_PATH = "/v1/videos"
    STATUS_PATH_TEMPLATE = "/v1/videos/{task_id}"

    def is_terminal_error_payload(self, payload):
        status = str(self.extract_status(payload) or "").strip().lower()
        if self.is_failure_status(status):
            return True
        detail = str(self.extract_failure_detail(payload) or "").strip().lower()
        if not detail:
            return False
        if status == "":
            return self._has_explicit_error_field(payload)
        return any(marker in detail for marker in (
            "content-moderated", "content moderation", "internal_error",
            "upstream", "上游返回", "invalid",
        ))

    @classmethod
    def _has_explicit_error_field(cls, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).strip().lower() in {"error", "failure", "fail_reason", "failure_reason"}:
                    if item not in (None, "", {}, []):
                        return True
                if cls._has_explicit_error_field(item):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(cls._has_explicit_error_field(item) for item in value)
        return False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "生成模式": (cls.MODES, {"default": "文生视频"}),
                "时长": ("INT", {"default": 6, "min": 1, "max": 15, "step": 1}),
                "比例": (cls.RATIOS, {"default": "默认"}),
                "清晰度": (cls.RESOLUTIONS, {"default": "720p"}),
            },
            "optional": {
                "首帧图": ("IMAGE",),
                "参考图": ("IMAGE", {"tooltip": "参考生视频最多连接 7 张图；IMAGE 批次顺序就是参考顺序。"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 180, "min": 1, "max": 720, "step": 1}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL")
    FUNCTION = "generate"

    def load_lingliu_grok_openai_video_config(self, api_key_override=""):
        config = self.load_lingliu_route_config(api_key_override=api_key_override)
        config["videos_generate_path"] = self.GENERATE_PATH
        config["videos_status_path_template"] = self.STATUS_PATH_TEMPLATE
        return config

    @staticmethod
    def _image_list(value):
        if value is None:
            return []
        if isinstance(value, Image.Image):
            return [value]
        if isinstance(value, (list, tuple)):
            images = []
            for item in value:
                images.extend(LingliuGrokImagineVideo15OpenAI._image_list(item))
            return images
        try:
            return tensor2pil(value)
        except Exception as exc:
            raise ValueError(f"输入不是可识别的 IMAGE：{exc}") from exc

    @classmethod
    def validate_request(cls, *, prompt, mode, duration, ratio, resolution, first_frame, reference_images):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空。")
        mode = str(mode or "").strip()
        if mode not in cls.MODES:
            raise ValueError("生成模式不受支持。")
        duration = int(duration)
        if not 1 <= duration <= 15:
            raise ValueError("时长必须在 1–15 秒之间。")
        ratio = str(ratio or "默认").strip()
        if ratio not in cls.RATIOS:
            raise ValueError("比例不受 Grok Imagine Video 1.5 支持。")
        resolution = str(resolution or "720p").strip().lower()
        if resolution not in cls.RESOLUTIONS:
            raise ValueError("清晰度只能选择 480p、720p 或 1080p。")

        first_images = cls._image_list(first_frame)
        ref_images = cls._image_list(reference_images)
        if len(first_images) > 1:
            raise ValueError("首帧图模式只能连接 1 张首帧图。")
        if len(ref_images) > cls.MAX_REFERENCE_IMAGES:
            raise ValueError("参考生视频最多支持 7 张参考图。")

        if mode == "文生视频":
            if first_images or ref_images:
                raise ValueError("文生视频模式不能连接首帧图或参考图。")
        elif mode == "首帧图生视频":
            if len(first_images) != 1:
                raise ValueError("首帧图生视频模式必须连接 1 张首帧图。")
            if ref_images:
                raise ValueError("首帧图生视频模式不能混用参考图。")
        elif mode == "参考生视频":
            if first_images:
                raise ValueError("参考生视频模式不能连接首帧图。")
            if not ref_images:
                raise ValueError("参考生视频模式至少需要 1 张参考图。")
            if resolution == "1080p":
                raise ValueError("参考生视频最高支持 720p，不能选择 1080p。")

        return {
            "prompt": prompt,
            "mode": mode,
            "duration": duration,
            "ratio": ratio,
            "resolution": resolution,
            "first_images": first_images,
            "reference_images": ref_images,
        }

    @classmethod
    def build_payload(cls, *, prompt, mode, duration, ratio, resolution, first_url="", reference_urls=None):
        payload = {
            "model": cls.MODEL,
            "prompt": str(prompt or "").strip(),
            "seconds": str(int(duration)),
            "resolution": str(resolution or "720p").strip().lower(),
        }
        if str(ratio or "默认").strip() != "默认":
            payload["aspect_ratio"] = str(ratio).strip()
        if mode == "首帧图生视频":
            payload["image"] = str(first_url or "").strip()
        elif mode == "参考生视频":
            payload["images"] = [str(url).strip() for url in (reference_urls or []) if str(url).strip()]
        return payload

    @staticmethod
    def _extract_url(payload):
        if isinstance(payload, Mapping):
            for key in ("url", "file_url", "fileUrl", "download_url", "downloadUrl"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                    return value.strip()
            for value in payload.values():
                result = LingliuGrokImagineVideo15OpenAI._extract_url(value)
                if result:
                    return result
        elif isinstance(payload, (list, tuple)):
            for value in payload:
                result = LingliuGrokImagineVideo15OpenAI._extract_url(value)
                if result:
                    return result
        return ""

    @classmethod
    def first_video_url(cls, payload):
        paths = (
            ("video", "url"),
            ("video_url",),
            ("url",),
            ("data", "video", "url"),
            ("data", "video_url"),
            ("data", "url"),
            ("data", 0, "video", "url"),
            ("data", 0, "video_url"),
            ("data", 0, "url"),
            ("result", "video", "url"),
            ("result", "video_url"),
            ("result", "url"),
            ("output", "video", "url"),
            ("output", "url"),
        )
        for path in paths:
            value = payload
            for key in path:
                if isinstance(value, Mapping):
                    value = value.get(key)
                elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
                    value = value[key]
                else:
                    value = None
                    break
            if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                return value.strip()
        return ""

    @classmethod
    def _redact_urls(cls, value, parent_key=""):
        if isinstance(value, Mapping):
            redacted = {}
            for key, item in value.items():
                normalized = str(key).strip().lower()
                if normalized in {"url", "video_url", "file_url", "download_url", "image", "images", "reference_images"}:
                    if isinstance(item, str) and item.startswith(("http://", "https://")):
                        redacted[key] = "<uploaded-or-result-url>"
                        continue
                    if isinstance(item, (list, tuple)):
                        redacted[key] = ["<uploaded-or-result-url>" for _ in item]
                        continue
                redacted[key] = cls._redact_urls(item, normalized)
            return redacted
        if isinstance(value, (list, tuple)):
            return [cls._redact_urls(item, parent_key) for item in value]
        if isinstance(value, str) and parent_key in {"url", "video_url", "file_url", "download_url"} and value.startswith(("http://", "https://")):
            return "<uploaded-or-result-url>"
        return value

    async def upload_image(self, image, slot, config):
        data = image_to_png_bytes(image)
        payload = await self.async_request_json_retrying(
            "POST",
            self.endpoint_for_path(config, "/v1/files"),
            config,
            max_retries=1,
            initial_timeout=self.timeout(config),
            headers=self.headers(config),
            files={"file": (f"{slot}.png", data, "image/png")},
        )
        url = self._extract_url(payload)
        if not url:
            raise RuntimeError(f"{slot} 上传响应缺少有效 URL。")
        return url

    async def upload_images(self, images, config, prefix):
        urls = []
        for index, image in enumerate(images, start=1):
            urls.append(await self.upload_image(image, f"{prefix}{index}", config))
        return urls

    async def wait_for_video_task(self, config, task_id, *, max_poll_attempts, poll_interval):
        task_url = self.endpoint_from_template(config, "videos_status_path_template", task_id=task_id)
        last_payload = None
        for attempt in range(int(max_poll_attempts)):
            if poll_interval:
                await asyncio.sleep(max(0, int(poll_interval)))
            payload = await self.async_request_json_retrying(
                "GET",
                task_url,
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config),
            )
            last_payload = payload
            video_url = self.first_video_url(payload)
            status = str(self.extract_status(payload) or "").strip().lower()
            if video_url and (not status or self.is_success_status(status)):
                return payload, video_url
            if self.is_terminal_error_payload(payload):
                detail = self.extract_failure_detail(payload)
                raise RuntimeError(f"Grok Imagine Video 1.5 任务失败：{detail or payload}")
            if status in {"failed", "failure", "error", "expired", "cancelled", "canceled", "rejected"}:
                detail = self.extract_failure_detail(payload)
                raise RuntimeError(f"Grok Imagine Video 1.5 任务失败：{detail or payload}")
            if status in {"success", "succeeded", "completed", "complete", "done"} and not video_url:
                raise RuntimeError(f"任务已完成但未返回视频 URL：{payload}")
            if attempt + 1 >= int(max_poll_attempts):
                break
        raise TimeoutError(f"Grok Imagine Video 1.5 任务 {task_id} 轮询超时：{last_payload}")

    async def generate(
        self,
        提示词,
        生成模式,
        时长,
        比例,
        清晰度,
        首帧图=None,
        参考图=None,
        API密钥="",
        等待完成=True,
        最大轮询次数=180,
        轮询间隔=5,
        unique_id=None,
    ):
        task_id = ""
        self.publish_task_id(unique_id, clear=True)
        try:
            if not 1 <= int(最大轮询次数) <= 720:
                raise ValueError("最大轮询次数必须在 1–720 之间。")
            if not 1 <= int(轮询间隔) <= 60:
                raise ValueError("轮询间隔必须在 1–60 秒之间。")
            request = self.validate_request(
                prompt=提示词,
                mode=生成模式,
                duration=时长,
                ratio=比例,
                resolution=清晰度,
                first_frame=首帧图,
                reference_images=参考图,
            )
            config = self.load_lingliu_grok_openai_video_config(api_key_override=API密钥)
            if not (config.get("api_key") or "").strip():
                raise RuntimeError("API key is required for lingliu_grok_imagine_video_1_5_openai.")

            first_urls = await self.upload_images(request["first_images"], config, "首帧图")
            reference_urls = await self.upload_images(request["reference_images"], config, "参考图")
            payload = self.build_payload(
                prompt=request["prompt"],
                mode=request["mode"],
                duration=request["duration"],
                ratio=request["ratio"],
                resolution=request["resolution"],
                first_url=first_urls[0] if first_urls else "",
                reference_urls=reference_urls,
            )
            submit_url = self.endpoint(config, "videos_generate_path")
            submit_payload = await self.async_request_json_retrying(
                "POST",
                submit_url,
                config,
                max_retries=1,
                initial_timeout=self.timeout(config),
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            task_id = self.extract_task_id(submit_payload)
            if task_id:
                self.publish_task_id(unique_id, task_id, append=False)
            video_url = self.first_video_url(submit_payload)
            final_payload = submit_payload
            if not video_url and bool(等待完成):
                if not task_id:
                    raise RuntimeError(f"提交响应未返回 request_id：{submit_payload}")
                final_payload, video_url = await self.wait_for_video_task(
                    config,
                    task_id,
                    max_poll_attempts=最大轮询次数,
                    poll_interval=轮询间隔,
                )

            response_info = {
                "node": "lingliu_grok_imagine_video_1_5_openai",
                "model": self.MODEL,
                "channel_id": self.ROUTE_CHANNEL_ID,
                "mode": request["mode"],
                "endpoint": self.GENERATE_PATH,
                "request": self._redact_urls(payload),
                "reference_image_count": len(reference_urls),
                "submit": self._redact_urls(submit_payload),
                "final": self._redact_urls(final_payload),
                "task_id": task_id,
            }
            if video_url:
                response_info["video_url"] = "<result-url>"
                return (
                    _GrokOpenAIVideoAdapter(video_url, config.get("api_key", "")),
                    task_id,
                    self.safe_json(response_info),
                    video_url,
                )
            return LingliuNullVideoAdapter("任务已提交，尚未返回视频 URL。"), task_id, self.safe_json(response_info), ""
        except Exception as exc:
            return LingliuNullVideoAdapter(str(exc)), task_id, self.error_json(exc, task_id=task_id), ""
