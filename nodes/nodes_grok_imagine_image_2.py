"""Dedicated node for xAI Grok Imagine Image 2.0."""

from __future__ import annotations

import asyncio

from ..utils import blank_image_tensor, image_to_data_url, image_to_png_bytes, parse_image_response, pil2tensor
from .nodes_grok import LingliuGrokBase


class LingliuGrokImagineImage2(LingliuGrokBase):
    MODEL = "grok-imagine-image-2.0"
    # A completed task must not keep a ComfyUI execution occupied forever
    # while a returned result URL is slow or unreachable.
    RESULT_DOWNLOAD_TIMEOUT = 60.0
    ASPECT_RATIOS = [
        "auto",
        "1:1",
        "3:4",
        "4:3",
        "9:16",
        "16:9",
        "2:3",
        "3:2",
        "9:19.5",
        "19.5:9",
        "9:20",
        "20:9",
        "1:2",
        "2:1",
    ]
    RESOLUTIONS = ["1K", "2K"]
    QUALITIES = ["low", "medium"]
    RESPONSE_FORMATS = ["url", "b64_json"]
    MAX_REFERENCE_IMAGES = 3

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "比例": (cls.ASPECT_RATIOS, {"default": "auto"}),
                "清晰度": (cls.RESOLUTIONS, {"default": "1K"}),
                "质量": (cls.QUALITIES, {"default": "medium"}),
                "生成数量": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "响应格式": (cls.RESPONSE_FORMATS, {"default": "url"}),
            },
            "optional": {
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720, "step": 1}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图像URL", "任务ID", "响应")
    FUNCTION = "generate"
    CATEGORY = "Lingliu/Grok"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        del kwargs
        return float("nan")

    @classmethod
    def normalize_resolution(cls, resolution):
        value = str(resolution or "1K").strip().upper()
        if value not in cls.RESOLUTIONS:
            raise ValueError("清晰度只能选择 1K 或 2K。")
        # The local platform adapter expects the enum in lower case while the
        # UI intentionally remains human-readable as 1K/2K.
        return value.lower()

    @classmethod
    def validate_controls(cls, ratio, resolution, quality, count, response_format, max_poll_attempts, poll_interval):
        if str(ratio or "").strip() not in cls.ASPECT_RATIOS:
            raise ValueError("比例不是 Grok Imagine Image 2.0 支持的值。")
        cls.normalize_resolution(resolution)
        if str(quality or "").strip().lower() not in cls.QUALITIES:
            raise ValueError("质量只能选择 low 或 medium。")
        if not 1 <= int(count) <= 10:
            raise ValueError("生成数量必须在 1–10 之间。")
        if str(response_format or "").strip() not in cls.RESPONSE_FORMATS:
            raise ValueError("响应格式只能选择 url 或 b64_json。")
        if not 1 <= int(max_poll_attempts) <= 720:
            raise ValueError("最大轮询次数必须在 1–720 之间。")
        if not 1 <= int(poll_interval) <= 60:
            raise ValueError("轮询间隔必须在 1–60 秒之间。")

    @classmethod
    def collect_reference_images(cls, *images):
        references = []
        for image in images:
            if image is None:
                continue
            batch_count = int(image.shape[0]) if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for index in range(batch_count):
                if len(references) >= cls.MAX_REFERENCE_IMAGES:
                    raise ValueError("Grok Imagine Image 2.0 最多支持 3 张参考图。")
                single_image = image[index : index + 1] if batch_count > 1 else image
                references.append(image_to_data_url(single_image))
        return references

    @classmethod
    def build_payload(cls, *, prompt, ratio, resolution, quality, count, response_format, reference_images):
        payload = {
            "model": cls.MODEL,
            "prompt": str(prompt or ""),
            "n": int(count),
            "aspect_ratio": str(ratio).strip(),
            "resolution": cls.normalize_resolution(resolution),
            "quality": str(quality).strip().lower(),
            "response_format": str(response_format).strip(),
        }
        if len(reference_images or []) == 1:
            payload["image"] = {"type": "image_url", "url": reference_images[0]}
        elif reference_images:
            payload["images"] = [
                {"type": "image_url", "url": str(url)}
                for url in reference_images
            ]
        return payload

    @classmethod
    def collect_reference_image_bytes(cls, *images):
        references = []
        for image in images:
            if image is None:
                continue
            batch_count = int(image.shape[0]) if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for index in range(batch_count):
                if len(references) >= cls.MAX_REFERENCE_IMAGES:
                    raise ValueError("Grok Imagine Image 2.0 最多支持 3 张参考图。")
                single_image = image[index : index + 1] if batch_count > 1 else image
                references.append(image_to_png_bytes(single_image))
        return references

    @classmethod
    def build_multipart_request(cls, *, prompt, ratio, resolution, quality, count, response_format, reference_image_bytes):
        fields = {
            "model": cls.MODEL,
            "prompt": str(prompt or ""),
            "n": str(int(count)),
            "aspect_ratio": str(ratio).strip(),
            "resolution": cls.normalize_resolution(resolution),
            "quality": str(quality).strip().lower(),
            "response_format": str(response_format).strip(),
        }
        files = [
            ("image", (f"reference_image_{index}.png", data, "image/png"))
            for index, data in enumerate(reference_image_bytes or [], start=1)
        ]
        return fields, files

    async def decode_images(self, payload, timeout):
        # The platform's async image-edit contract nests the final image under
        # task/result/data envelopes.  The shared parser intentionally handles
        # only the OpenAI top-level ``data[]`` shape, so normalize every image
        # URL/base64 leaf here without changing other image nodes.
        image_items = []
        seen = set()

        def walk(value):
            if isinstance(value, dict):
                b64_json = value.get("b64_json")
                if isinstance(b64_json, str) and b64_json.strip():
                    marker = ("b64_json", b64_json)
                    if marker not in seen:
                        seen.add(marker)
                        image_items.append({"b64_json": b64_json})
                url = value.get("url")
                if isinstance(url, str) and url.strip():
                    marker = ("url", url.strip())
                    if marker not in seen:
                        seen.add(marker)
                        image_items.append({"url": url.strip()})
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(payload)
        try:
            requested_timeout = float(timeout)
        except (TypeError, ValueError):
            requested_timeout = self.RESULT_DOWNLOAD_TIMEOUT
        download_timeout = min(
            max(requested_timeout, 0.1),
            float(self.RESULT_DOWNLOAD_TIMEOUT),
        )

        async def parse_with_bound(items):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        parse_image_response,
                        {"data": items},
                        download_timeout,
                    ),
                    timeout=download_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "Grok Imagine Image 2.0 图片结果下载超时："
                    f"{download_timeout:g} 秒。"
                ) from exc

        images, urls = await parse_with_bound(image_items)
        if images:
            return images, urls
        image_urls, _video_urls = self.extract_media_urls(payload)
        if image_urls:
            return await parse_with_bound([{"url": url} for url in image_urls])
        return [], []

    async def request_image(
        self,
        *,
        config,
        url,
        payload,
        wait_for_completion,
        max_poll_attempts,
        poll_interval,
        unique_id,
        multipart_data=None,
        multipart_files=None,
    ):
        request_kwargs = {
            "max_retries": 1,
            "initial_timeout": self.timeout(config),
            "headers": self.headers(config),
        }
        if multipart_files:
            request_kwargs.update({"data": multipart_data or {}, "files": multipart_files})
        else:
            request_kwargs["headers"] = self.headers(config, {"Content-Type": "application/json"})
            request_kwargs["json"] = payload
        response_payload = await self.async_request_json_retrying(
            "POST",
            url,
            config,
            **request_kwargs,
        )
        images, urls = await self.decode_images(response_payload, self.timeout(config))
        task_id = self.extract_task_id(response_payload)
        if images:
            return images, urls, task_id, response_payload
        if not task_id:
            raise RuntimeError(f"Grok Imagine Image 2.0 未返回图片或任务 ID：{response_payload}")
        self.publish_task_id(unique_id, task_id, label="Task ID", append=False)
        if not wait_for_completion:
            return [], [], task_id, response_payload

        last_payload = response_payload
        for attempt in range(int(max_poll_attempts)):
            query_payload = await self.async_request_json_retrying(
                "GET",
                f"{config['base_url'].rstrip('/')}/v1/images/tasks/{task_id}",
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config),
            )
            last_payload = query_payload
            images, urls = await self.decode_images(query_payload, self.timeout(config))
            if images:
                return images, urls, task_id, query_payload
            status = str(self.extract_status(query_payload) or "").strip().lower()
            if status in {"failed", "failure", "error", "cancelled", "canceled", "rejected", "expired"}:
                raise RuntimeError(f"Grok Imagine Image 2.0 任务失败：{query_payload}")
            if status in {"success", "succeeded", "completed", "complete", "done", "finished"}:
                raise RuntimeError(f"Grok Imagine Image 2.0 任务成功但未返回图片：{query_payload}")
            if attempt + 1 < int(max_poll_attempts):
                await asyncio.sleep(int(poll_interval))
        raise TimeoutError(f"Grok Imagine Image 2.0 任务 {task_id} 轮询超时：{last_payload}")

    async def generate(
        self,
        提示词,
        比例,
        清晰度,
        质量,
        生成数量,
        响应格式,
        参考图1=None,
        参考图2=None,
        参考图3=None,
        API密钥="",
        等待完成=True,
        最大轮询次数=120,
        轮询间隔=5,
        unique_id=None,
    ):
        self.validate_controls(比例, 清晰度, 质量, 生成数量, 响应格式, 最大轮询次数, 轮询间隔)
        prompt = str(提示词 or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空。")
        config = self.load_lingliu_route_config(api_key_override=API密钥)
        if not (config.get("api_key") or "").strip():
            raise RuntimeError("API key is required for lingliu_grok_imagine_image_2.")
        reference_image_bytes = self.collect_reference_image_bytes(参考图1, 参考图2, 参考图3)
        multipart_data = None
        multipart_files = None
        if reference_image_bytes:
            multipart_data, multipart_files = self.build_multipart_request(
                prompt=prompt,
                ratio=比例,
                resolution=清晰度,
                quality=质量,
                count=生成数量,
                response_format=响应格式,
                reference_image_bytes=reference_image_bytes,
            )
            payload = self.build_payload(
                prompt=prompt,
                ratio=比例,
                resolution=清晰度,
                quality=质量,
                count=生成数量,
                response_format=响应格式,
                reference_images=[],
            )
            if len(reference_image_bytes) == 1:
                payload["image"] = {"type": "multipart_file", "filename": "reference_image_1.png"}
            else:
                payload["images"] = [
                    {"type": "multipart_file", "filename": f"reference_image_{index}.png"}
                    for index in range(1, len(reference_image_bytes) + 1)
                ]
        else:
            payload = self.build_payload(
                prompt=prompt,
                ratio=比例,
                resolution=清晰度,
                quality=质量,
                count=生成数量,
                response_format=响应格式,
                reference_images=[],
            )
        path = "/v1/images/edits" if reference_image_bytes else "/v1/images/generations"
        submit_url = f"{config['base_url'].rstrip('/')}{path}"
        if reference_image_bytes:
            # Reference edits are an asynchronous platform task.  Keep the
            # text-only generations contract unchanged.
            submit_url += "?async=true"
        images, urls, task_id, response_payload = await self.request_image(
            config=config,
            url=submit_url,
            payload=payload,
            wait_for_completion=bool(等待完成),
            max_poll_attempts=最大轮询次数,
            poll_interval=轮询间隔,
            unique_id=unique_id,
            multipart_data=multipart_data,
            multipart_files=multipart_files,
        )
        if images:
            image_output = pil2tensor(images)
        else:
            image_output = blank_image_tensor(width=1, height=1, color="black")
        response_info = {
            "node": "lingliu_grok_imagine_image_2",
            "model": self.MODEL,
            "request": payload,
            "endpoint": path + ("?async=true" if reference_image_bytes else ""),
            "reference_image_count": len(reference_image_bytes),
            "task_id": task_id,
            "response": response_payload,
        }
        return image_output, "\n".join(urls), task_id, self.safe_json(response_info)
