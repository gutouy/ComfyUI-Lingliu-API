"""Dedicated Lingliu node for the GPT Image 2.5 model family.

This module intentionally owns the GPT Image 2.5 contract.  The older GPT
Image nodes use different request shapes and are left untouched so existing
workflows keep their original behaviour.
"""

from __future__ import annotations

import asyncio
from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    normalize_provider,
    provider_display_name,
    provider_defaults,
)
from ..utils import blank_image_tensor, image_to_png_bytes, parse_image_response, pil2tensor
from .nodes_openai import LingliuOpenAIBase
from .lingliu_image_channels import (
    CHANNELS,
    CHANNEL_DEFAULT_LEGACY,
    CHANNEL_DEFAULT_NEW_NODE,
    CHANNEL_JL,
    JL_QUALITIES,
    ImageChannelError,
    ImageSubmitUncertain,
    config_with_channel,
    jl_contract,
    jl_reference_bytes,
    resolve_route,
    submit_once,
    validate_jl_ratio,
    validate_jl_resolution,
)


class LingliuGptImage25(LingliuOpenAIBase):
    """GPT Image 2.5 Flare/Sunburst via the Lingliu API routes."""

    CATEGORY = "Lingliu/OpenAI"
    MODELS = [
        "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-1k",
        "gpt-image-2.5-flare-1k",
        "gpt-image-2.5-sunburst-1k",
        "gpt-image-2.5-flare-all",
        "gpt-image-2.5-sunburst-all",
        "gpt-image-2.5-flare-hq2",
        "gpt-image-2.5-sunburst-hq2",
        "gpt-image-2.5-flare-hq3",
        "gpt-image-2.5-sunburst-hq3",
        "gpt-image-2.5-flare-official",
        "gpt-image-2.5-sunburst-official",
        "gpt-image-2.5-flare-enterprise",
        "gpt-image-2.5-sunburst-enterprise",
    ]
    # V17.4: these aliases carry their own platform channel, so the 渠道 widget
    # is ignored and the request stays unsuffixed.  -all goes through the JL
    # contract; -hq2/-hq3 through _generate_fixed.
    FIXED_CHANNEL_MODELS = {
        "gpt-image-2.5-flare-all": "优惠渠道2",
        "gpt-image-2.5-sunburst-all": "优惠渠道2",
        "gpt-image-2.5-flare-hq2": "高质渠道2",
        "gpt-image-2.5-sunburst-hq2": "高质渠道2",
        "gpt-image-2.5-flare-hq3": "优惠渠道3",
        "gpt-image-2.5-sunburst-hq3": "优惠渠道3",
        "gpt-image-2.5-flare-official": "官方",
        "gpt-image-2.5-sunburst-official": "官方",
        "gpt-image-2.5-flare-enterprise": "官方-企业",
        "gpt-image-2.5-sunburst-enterprise": "官方-企业",
    }
    # 官方/官方-企业 price by size × quality and reject auto, so the quality
    # list is explicit per model family.
    OFFICIAL_QUALITIES = {
        "gpt-image-2": ["low", "medium", "high"],
        "gpt-image-2.5": ["low", "medium", "high", "xhigh", "max"],
    }
    HQ3_RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "5:4", "4:5"]
    HQ3_REFERENCE_BUDGET = 50 << 20
    MODEL_TOOLTIP = (
        "-all=优惠渠道2，-hq2=高质渠道2，-hq3=优惠渠道3：自带渠道（忽略「渠道」选项），按张数逐张提交，"
        "质量只支持 auto/low/medium/high，背景/输出格式/内容审核需保持默认。高质渠道2 的 2K/4K 暂不稳定。"
        "-official=官方，-enterprise=官方-企业：同样自带渠道，质量只支持 low/medium/high/xhigh/max（无 auto），"
        "按尺寸×质量×参考图计价，2K/4K 搭配高质量较慢。"
    )
    ROUTES = ["灵流"]
    ROUTE_PROVIDERS = {"灵流": LINGLIU_PROVIDER}
    # 5:4, 4:5 and 21:9 exist only on JL; LL keeps the SIZE_MAP ratios.
    ASPECT_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "21:9"]
    RESOLUTIONS = ["1K", "2K", "4K"]
    QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]
    OUTPUT_FORMATS = ["png", "jpeg", "webp"]
    BACKGROUNDS = ["auto", "opaque", "transparent"]
    MODERATION = ["auto", "low"]
    MAX_REFERENCE_IMAGES = 16
    SIZE_MAP = {
        "1:1": {"1K": "1024x1024", "2K": "2048x2048", "4K": "2880x2880"},
        "16:9": {"1K": "1536x864", "2K": "2048x1152", "4K": "3840x2160"},
        "9:16": {"1K": "864x1536", "2K": "1152x2048", "4K": "2160x3840"},
        "3:2": {"1K": "1536x1024", "2K": "2048x1360", "4K": "3312x2208"},
        "2:3": {"1K": "1024x1536", "2K": "1360x2048", "4K": "2208x3312"},
        "4:3": {"1K": "1536x1152", "2K": "2048x1536", "4K": "3264x2448"},
        "3:4": {"1K": "1152x1536", "2K": "1536x2048", "4K": "2448x3264"},
    }
    # -hq2/-hq3 accept the platform's full GPT Image 2.5 size table.
    FIXED_SIZE_MAP = {
        **SIZE_MAP,
        "21:9": {"1K": "1536x656", "2K": "2048x880", "4K": "3840x1648"},
        "5:4": {"1K": "1280x1024", "2K": "2048x1632", "4K": "3200x2560"},
        "4:5": {"1K": "1024x1280", "2K": "1632x2048", "4K": "2560x3200"},
    }
    TERMINAL_SUCCESS = {"success", "succeeded", "completed", "complete", "done", "finished"}
    TERMINAL_FAILURE = {
        "failed",
        "failure",
        "error",
        "cancelled",
        "canceled",
        "rejected",
        "expired",
    }
    NON_TERMINAL = {"pending", "queued", "processing", "running", "in_progress", "submitted"}
    # Task errors name the model family; the GPT Image 2 node overrides it.
    TASK_LABEL = "GPT Image 2.5"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "线路": (cls.ROUTES, {"default": "灵流"}),
                "模型": (cls.MODELS, {"default": cls.MODELS[0], "tooltip": cls.MODEL_TOOLTIP}),
                "比例": (cls.ASPECT_RATIOS, {"default": "1:1"}),
                "清晰度": (cls.RESOLUTIONS, {"default": "1K"}),
                "质量": (cls.QUALITIES, {"default": "auto"}),
                "生成数量": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "输出格式": (cls.OUTPUT_FORMATS, {"default": "png"}),
                "背景": (cls.BACKGROUNDS, {"default": "auto"}),
                "内容审核": (cls.MODERATION, {"default": "auto"}),
            },
            "optional": {
                "参考图像": ("IMAGE",),
                "输出压缩": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 300, "min": 1, "max": 1000, "step": 1}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                # V16.3: keep the channel last so legacy widgets_values stay aligned.
                "渠道": (CHANNELS, {"default": CHANNEL_DEFAULT_NEW_NODE}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图像URL", "任务ID", "响应")
    FUNCTION = "generate"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        del kwargs
        return float("nan")

    @classmethod
    def _batch_count(cls, value):
        if value is None:
            return 0
        if hasattr(value, "shape") and len(value.shape) == 4:
            return int(value.shape[0])
        return 1

    @classmethod
    def validate_controls(
        cls,
        route,
        model,
        ratio,
        resolution,
        quality,
        count,
        output_format,
        background,
        moderation,
        compression,
        max_poll_attempts,
        poll_interval,
    ):
        if str(route or "").strip() not in cls.ROUTES:
            raise ValueError("线路只能选择灵流。")
        if str(model or "").strip() not in cls.MODELS:
            raise ValueError("模型不是 GPT Image 2.5 支持的公开模型。")
        if str(ratio or "").strip() not in cls.ASPECT_RATIOS:
            raise ValueError("比例不是 GPT Image 2.5 支持的值。")
        if str(resolution or "").strip().upper() not in cls.RESOLUTIONS:
            raise ValueError("清晰度只能选择 1K、2K 或 4K。")
        if str(quality or "").strip().lower() not in cls.QUALITIES:
            raise ValueError("质量不是 GPT Image 2.5 支持的值。")
        if not 1 <= int(count) <= 10:
            raise ValueError("生成数量必须在 1–10 之间。")
        normalized_format = str(output_format or "").strip().lower()
        if normalized_format not in cls.OUTPUT_FORMATS:
            raise ValueError("输出格式只能选择 png、jpeg 或 webp。")
        normalized_background = str(background or "").strip().lower()
        if normalized_background not in cls.BACKGROUNDS:
            raise ValueError("背景只能选择 auto、opaque 或 transparent。")
        if normalized_background == "transparent" and normalized_format not in {"png", "webp"}:
            raise ValueError("透明背景只能与 png 或 webp 输出格式组合。")
        if str(moderation or "").strip().lower() not in cls.MODERATION:
            raise ValueError("内容审核只能选择 auto 或 low。")
        if not 0 <= int(compression) <= 100:
            raise ValueError("输出压缩必须在 0–100 之间。")
        if not 1 <= int(max_poll_attempts) <= 1000:
            raise ValueError("最大轮询次数必须在 1–1000 之间。")
        if not 1 <= int(poll_interval) <= 60:
            raise ValueError("轮询间隔必须在 1–60 秒之间。")

    @classmethod
    def size_for(cls, ratio, resolution):
        ratio = str(ratio or "").strip()
        resolution = str(resolution or "").strip().upper()
        try:
            return cls.SIZE_MAP[ratio][resolution]
        except KeyError as exc:
            raise ValueError("无法根据比例和清晰度计算 GPT Image 2.5 尺寸。") from exc

    @classmethod
    def build_json_payload(
        cls,
        *,
        model,
        prompt,
        count,
        ratio,
        resolution,
        quality,
        output_format,
        background,
        moderation,
        compression=100,
    ):
        output_format = str(output_format).strip().lower()
        payload = {
            "model": str(model).strip(),
            "prompt": str(prompt or ""),
            "n": int(count),
            "size": cls.size_for(ratio, resolution),
            "quality": str(quality).strip().lower(),
            "background": str(background).strip().lower(),
            "output_format": output_format,
            "moderation": str(moderation).strip().lower(),
        }
        if output_format in {"jpeg", "webp"}:
            payload["output_compression"] = int(compression)
        return payload

    @classmethod
    def collect_reference_image_bytes(cls, reference_image):
        count = cls._batch_count(reference_image)
        if count > cls.MAX_REFERENCE_IMAGES:
            raise ValueError("GPT Image 2.5 最多支持 16 张参考图。")
        if count <= 0:
            return []
        return [
            image_to_png_bytes(reference_image[index : index + 1])
            for index in range(count)
        ]

    @staticmethod
    def extract_file_url(payload):
        """Extract a public URL from a /v1/files response envelope."""
        def walk(value):
            if isinstance(value, dict):
                for key in ("url", "download_url", "file_url", "fileUrl"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip().startswith(("http://", "https://")):
                        return candidate.strip()
                for child in value.values():
                    candidate = walk(child)
                    if candidate:
                        return candidate
            elif isinstance(value, list):
                for child in value:
                    candidate = walk(child)
                    if candidate:
                        return candidate
            return ""

        url = walk(payload)
        if url:
            return url
        raise RuntimeError("/v1/files 上传响应缺少有效公网 URL。")

    async def upload_image_to_file_url(self, config, image_bytes, *, filename, mime="image/png"):
        response_payload = await self.async_request_json_retrying(
            "POST",
            build_url(config["base_url"], "/v1/files"),
            config,
            max_retries=3,
            initial_timeout=min(self.timeout(config), 180),
            headers=self.headers(config),
            data={"purpose": "assistants"},
            files={"file": (filename, image_bytes, mime)},
        )
        return self.extract_file_url(response_payload)

    async def upload_reference_images(self, config, reference_image_bytes, *, jpeg=False):
        """Upload one file at a time and preserve the input batch order."""
        urls = []
        extension, mime = ("jpg", "image/jpeg") if jpeg else ("png", "image/png")
        for index, image_bytes in enumerate(reference_image_bytes or [], start=1):
            urls.append(
                await self.upload_image_to_file_url(
                    config,
                    image_bytes,
                    filename=f"gpt_image_25_reference_{index}.{extension}",
                    mime=mime,
                )
            )
        return urls

    def load_gpt_image_25_config(self, route, api_key_override=""):
        selected_provider = self.ROUTE_PROVIDERS.get(str(route or "").strip())
        if not selected_provider:
            raise RuntimeError("线路只能选择灵流。")
        runtime = self.load_config(api_key_override=str(api_key_override or ""))
        configured_provider = normalize_provider(runtime.get("api_provider"))
        if configured_provider != selected_provider:
            raise RuntimeError(
                f"节点选择的线路为“{route}”，但灵流API设置当前线路为“{provider_display_name(configured_provider)}”。"
                "请先在 lingliu_api_set 中切换到相同线路。"
            )
        config = provider_defaults(selected_provider)
        config["api_provider"] = selected_provider
        config["api_key"] = str(runtime.get("api_key") or "").strip()
        config["timeout"] = int(runtime.get("timeout") or config.get("timeout") or 300)
        return config

    @staticmethod
    def _redact_payload(value):
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                if str(key).lower() in {"b64_json", "base64", "data_uri"} and isinstance(item, str):
                    redacted[key] = f"<base64_data:{len(item)} chars>"
                else:
                    redacted[key] = LingliuGptImage25._redact_payload(item)
            return redacted
        if isinstance(value, list):
            return [LingliuGptImage25._redact_payload(item) for item in value]
        return value

    @classmethod
    def _image_items(cls, payload):
        """Find image result leaves in direct and nested platform envelopes."""
        items = []
        seen = set()

        def walk(value):
            if isinstance(value, dict):
                for key in ("b64_json", "url"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        marker = (key, candidate.strip())
                        if marker not in seen:
                            seen.add(marker)
                            items.append({key: candidate.strip()})
                for nested in value.values():
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)

        walk(payload)
        return items

    async def decode_images(self, payload, timeout):
        image_items = self._image_items(payload)
        if not image_items:
            return [], []
        images, urls = await asyncio.to_thread(parse_image_response, {"data": image_items}, timeout)
        return images, urls

    async def _request_once(self, method, url, config, *, json_payload=None, max_retries=1):
        kwargs = {"max_retries": max_retries, "initial_timeout": self.timeout(config), "headers": self.headers(config)}
        kwargs["headers"] = self.headers(config, {"Content-Type": "application/json"})
        kwargs["json"] = json_payload
        return await self.async_request_json_retrying(method, url, config, **kwargs)

    async def wait_for_task(self, config, task_id, max_poll_attempts, poll_interval):
        query_url = build_url(config["base_url"], f"/v1/images/tasks/{task_id}")
        last_payload = None
        for attempt in range(int(max_poll_attempts)):
            if attempt > 0 or poll_interval:
                await asyncio.sleep(max(1, int(poll_interval)))
            last_payload = await self.async_request_json_retrying(
                "GET",
                query_url,
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config),
            )
            images, urls = await self.decode_images(last_payload, self.timeout(config))
            if images:
                return images, urls, last_payload
            status = str(self.extract_status(last_payload) or "").strip().lower()
            if status in self.TERMINAL_FAILURE:
                raise RuntimeError(f"{self.TASK_LABEL} 任务失败：{self._redact_payload(last_payload)}")
            if status in self.TERMINAL_SUCCESS:
                raise RuntimeError(f"{self.TASK_LABEL} 任务已成功，但响应中没有可用图像。")
            if status and status not in self.NON_TERMINAL:
                # Unknown states are deliberately treated as pending.  This
                # avoids misclassifying a new platform status as a failure.
                continue
        raise TimeoutError(f"{self.TASK_LABEL} 任务 {task_id} 轮询超时：{self._redact_payload(last_payload)}")

    async def generate(
        self,
        提示词,
        线路,
        模型,
        比例,
        清晰度,
        质量,
        生成数量,
        输出格式,
        背景,
        内容审核,
        参考图像=None,
        输出压缩=100,
        API密钥="",
        等待完成=True,
        最大轮询次数=300,
        轮询间隔=5,
        渠道=CHANNEL_DEFAULT_LEGACY,
        unique_id=None,
    ):
        self.validate_controls(
            线路,
            模型,
            比例,
            清晰度,
            质量,
            生成数量,
            输出格式,
            背景,
            内容审核,
            输出压缩,
            最大轮询次数,
            轮询间隔,
        )
        prompt = str(提示词 or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空。")
        config = self.load_gpt_image_25_config(线路, api_key_override=API密钥)
        if not config.get("api_key"):
            raise RuntimeError("API密钥不能为空，请在 lingliu_api_set 或节点中填写 API 密钥。")

        fixed_label = self.FIXED_CHANNEL_MODELS.get(str(模型).strip())
        if fixed_label:
            # The alias selects the channel; 渠道 is ignored and the key stays unsuffixed.
            handler = self._generate_jl if str(模型).strip().endswith("-all") else self._generate_fixed
            return await handler(
                config, None, prompt, 线路, 模型, 比例, 清晰度, 质量, 生成数量,
                输出格式, 背景, 内容审核, 参考图像, 等待完成, 最大轮询次数, 轮询间隔, unique_id,
                channel_label=fixed_label,
            )
        channel, channel_id = await resolve_route(config, 模型, 渠道)
        if channel == CHANNEL_JL:
            return await self._generate_jl(
                config, channel_id, prompt, 线路, 模型, 比例, 清晰度, 质量, 生成数量,
                输出格式, 背景, 内容审核, 参考图像, 等待完成, 最大轮询次数, 轮询间隔, unique_id,
            )
        if str(比例).strip() not in self.SIZE_MAP:
            raise ValueError(f"LL 渠道不支持比例 {比例}，请切换到 JL 或更换比例。")

        payload = self.build_json_payload(
            model=模型,
            prompt=prompt,
            count=生成数量,
            ratio=比例,
            resolution=清晰度,
            quality=质量,
            output_format=输出格式,
            background=背景,
            moderation=内容审核,
            compression=输出压缩,
        )
        reference_image_bytes = self.collect_reference_image_bytes(参考图像)
        has_references = bool(reference_image_bytes)
        # The LL GPT Image 2.5 contract accepts public references on the
        # generations endpoint.  The edits endpoint is multipart-only and
        # rejects JSON URL references, so both modes use generations here.
        endpoint_path = "/v1/images/generations"
        # GPT Image 2.5 must use the platform's asynchronous image contract.
        # The query flag is deliberately scoped to this node; legacy image
        # nodes retain their historical synchronous/async behaviour.
        endpoint = self.append_query_param(build_url(config["base_url"], endpoint_path), "async", "true")
        request_payload = payload
        if has_references:
            reference_urls = await self.upload_reference_images(config, reference_image_bytes)
            payload = dict(payload)
            payload["image_urls"] = reference_urls
            request_payload = dict(payload)
            request_payload["image_urls"] = [
                f"<uploaded_reference_{index}>"
                for index in range(1, len(reference_urls) + 1)
            ]

        # Uploads above used the raw key; only the create is pinned to LL.
        response_payload = await self._request_once(
            "POST",
            endpoint,
            config_with_channel(config, channel_id),
            json_payload=payload,
            max_retries=1,
        )
        images, urls = await self.decode_images(response_payload, self.timeout(config))
        task_id = self.extract_task_id(response_payload)
        if task_id:
            self.publish_task_id(unique_id, task_id, label="Task ID", append=False)
        if not images and task_id and bool(等待完成):
            try:
                images, urls, response_payload = await self.wait_for_task(
                    config,
                    task_id,
                    最大轮询次数,
                    轮询间隔,
                )
            except Exception as exc:
                response_info = {
                    "node": "lingliu_gpt_image_25",
                    "model": str(模型),
                    "route": str(线路),
                    "request": self._redact_payload(request_payload),
                    "endpoint": endpoint_path + "?async=true",
                    "reference_image_count": len(reference_image_bytes),
                    "task_id": task_id,
                    "status": "error",
                    "error": str(exc),
                    "response": self._redact_payload(response_payload),
                }
                return blank_image_tensor(width=1, height=1, color="black"), "", task_id, self.safe_json(response_info)

        response_info = {
            "node": "lingliu_gpt_image_25",
            "model": str(模型),
            "route": str(线路),
            "request": self._redact_payload(request_payload),
            "endpoint": endpoint_path + "?async=true",
            "reference_image_count": len(reference_image_bytes),
            "task_id": task_id,
            "response": self._redact_payload(response_payload),
        }
        if images:
            image_output = pil2tensor(images)
        else:
            image_output = blank_image_tensor(width=1, height=1, color="black")
        return image_output, "\n".join(urls), task_id, self.safe_json(response_info)

    async def _generate_jl(
        self,
        config,
        channel_id,
        prompt,
        route,
        model,
        ratio,
        resolution,
        quality,
        count,
        output_format,
        background,
        moderation,
        reference_image,
        wait,
        max_poll_attempts,
        poll_interval,
        unique_id,
        channel_label=CHANNEL_JL,
    ):
        """JL: image_size + aspect_ratio, one submit per image."""
        contract = jl_contract(model)
        # 优惠渠道2 (-all) reuses this path; its errors name that channel instead of JL.
        label = "JL 渠道" if channel_label == CHANNEL_JL else channel_label
        resolution = str(resolution).strip().upper()
        try:
            validate_jl_resolution(model, resolution)
            validate_jl_ratio(model, ratio)
        except ImageChannelError as exc:
            if channel_label == CHANNEL_JL:
                raise
            raise ImageChannelError(str(exc).replace("JL 渠道", label, 1)) from exc
        quality = str(quality).strip().lower()
        if quality not in JL_QUALITIES:
            raise ImageChannelError(f"{label}的质量只能选择 {'、'.join(JL_QUALITIES)}。")
        if (
            str(background).strip().lower() != "auto"
            or str(output_format).strip().lower() != "png"
            or str(moderation).strip().lower() != "auto"
        ):
            raise ImageChannelError(f"{label}只支持 背景=auto、输出格式=png、内容审核=auto。")
        if self._batch_count(reference_image) > contract["max_refs"]:
            raise ImageChannelError(f"{label}的 {model} 最多支持 {contract['max_refs']} 张参考图。")
        reference_count = self._batch_count(reference_image)
        reference_image_bytes = await asyncio.to_thread(
            lambda: [jl_reference_bytes(reference_image[index : index + 1]) for index in range(reference_count)]
        )
        payload = {
            "model": str(model).strip(),
            "prompt": prompt,
            "image_size": resolution,
            "aspect_ratio": str(ratio).strip(),
            "quality": quality,
        }
        request_payload = dict(payload)
        if reference_image_bytes:
            # /v1/files must use the raw key, otherwise the upload is routed to the JL channel.
            payload["image_urls"] = await self.upload_reference_images(config, reference_image_bytes, jpeg=True)
            request_payload["image_urls"] = [
                f"<uploaded_reference_{index}>" for index in range(1, len(reference_image_bytes) + 1)
            ]
        endpoint_path = "/v1/images/generations"
        endpoint = self.append_query_param(build_url(config["base_url"], endpoint_path), "async", "true")
        submit_config = config_with_channel(config, channel_id)
        headers = self.headers(submit_config, {"Content-Type": "application/json"})

        all_images, all_urls, task_ids, errors, responses = [], [], [], [], []
        for index in range(int(count)):
            task_id = ""
            try:
                response_payload = await submit_once(
                    self, endpoint, submit_config, timeout=self.timeout(config), headers=headers, json=payload,
                )
                task_id = self.extract_task_id(response_payload)
                if not task_id:
                    raise RuntimeError(f"创建响应缺少 task_id：{self._redact_payload(response_payload)}")
                task_ids.append(task_id)
                self.publish_task_id(unique_id, task_id, label=f"Task ID #{index + 1}", append=index > 0)
                if not bool(wait):
                    responses.append(self._redact_payload(response_payload))
                    continue
                images, urls, final_payload = await self.wait_for_task(config, task_id, max_poll_attempts, poll_interval)
                all_images.extend(images)
                all_urls.extend(urls)
                responses.append(self._redact_payload(final_payload))
            except ImageSubmitUncertain as exc:
                task_ids.append(exc.task_id)
                self.publish_task_id(unique_id, exc.task_id, label=f"Task ID #{index + 1}", append=index > 0)
                errors.append({"index": index, "task_id": exc.task_id, "error": str(exc)})
                break
            except Exception as exc:
                # Stop on the first failure: later images would be charged separately.
                errors.append({"index": index, "task_id": task_id, "error": str(exc)})
                break

        response_info = {
            "node": "lingliu_gpt_image_25",
            "model": str(model),
            "route": str(route),
            "channel": channel_label,
            "request": request_payload,
            "endpoint": endpoint_path + "?async=true",
            "reference_image_count": len(reference_image_bytes),
            "requested_count": int(count),
            "task_ids": task_ids,
            "errors": errors,
            "responses": responses,
        }
        if errors:
            response_info["status"] = "error"
        image_output = pil2tensor(all_images) if all_images else blank_image_tensor(width=1, height=1, color="black")
        return image_output, "\n".join(all_urls), "\n".join(task_ids), self.safe_json(response_info)

    @classmethod
    def official_qualities(cls, model):
        """Quality list for -official/-enterprise, or None for other aliases."""
        model = str(model).strip()
        for suffix in ("-official", "-enterprise"):
            if model.endswith(suffix):
                family = "gpt-image-2.5" if model.startswith("gpt-image-2.5") else "gpt-image-2"
                return cls.OFFICIAL_QUALITIES[family]
        return None

    async def _generate_fixed(
        self,
        config,
        channel_id,
        prompt,
        route,
        model,
        ratio,
        resolution,
        quality,
        count,
        output_format,
        background,
        moderation,
        reference_image,
        wait,
        max_poll_attempts,
        poll_interval,
        unique_id,
        channel_label,
    ):
        """-hq2/-hq3/-official/-enterprise on this node; channel_id is unused because the alias picks the channel."""
        result = await self.submit_fixed_channel(
            config, prompt, model, ratio, resolution, quality, count, output_format, background,
            moderation, reference_image, wait, max_poll_attempts, poll_interval, unique_id, channel_label,
        )
        response_info = {
            "node": "lingliu_gpt_image_25",
            "model": str(model).strip(),
            "route": str(route),
            "channel": channel_label,
            "request": result["request"],
            "endpoint": result["endpoint"],
            "reference_image_count": result["reference_image_count"],
            "requested_count": int(count),
            "task_ids": result["task_ids"],
            "errors": result["errors"],
            "responses": result["responses"],
        }
        if result["errors"]:
            response_info["status"] = "error"
        images = result["images"]
        image_output = pil2tensor(images) if images else blank_image_tensor(width=1, height=1, color="black")
        return image_output, "\n".join(result["urls"]), "\n".join(result["task_ids"]), self.safe_json(response_info)

    async def submit_fixed_channel(
        self,
        config,
        prompt,
        model,
        ratio,
        resolution,
        quality,
        count,
        output_format,
        background,
        moderation,
        reference_image,
        wait,
        max_poll_attempts,
        poll_interval,
        unique_id,
        channel_label,
    ):
        """-hq2/-hq3: size + explicit resolution tier, n=1, one submit per image.

        -hq3 answers the create synchronously with the image; -hq2 returns a
        task ID that is polled.  Both shapes are accepted.  The GPT Image 2
        node reuses this for gpt-image-2-hq2/-hq3.
        """
        model = str(model).strip()
        ratio = str(ratio).strip()
        resolution = str(resolution).strip().upper()
        ratios = self.HQ3_RATIOS if model.endswith("-hq3") else list(self.FIXED_SIZE_MAP)
        if ratio not in ratios:
            raise ImageChannelError(f"{channel_label}的 {model} 不支持比例 {ratio}，可用：{'、'.join(ratios)}。")
        quality = str(quality).strip().lower()
        qualities = self.official_qualities(model) or JL_QUALITIES
        if quality not in qualities:
            raise ImageChannelError(f"{channel_label}的质量只能选择 {'、'.join(qualities)}。")
        if (
            str(background).strip().lower() != "auto"
            or str(output_format).strip().lower() != "png"
            or str(moderation).strip().lower() != "auto"
        ):
            raise ImageChannelError(f"{channel_label}只支持 背景=auto、输出格式=png、内容审核=auto。")
        reference_count = self._batch_count(reference_image)
        if reference_count > self.MAX_REFERENCE_IMAGES:
            raise ImageChannelError(f"{channel_label}最多支持 {self.MAX_REFERENCE_IMAGES} 张参考图。")
        reference_image_bytes = await asyncio.to_thread(
            lambda: [jl_reference_bytes(reference_image[index : index + 1]) for index in range(reference_count)]
        )
        if model.endswith("-hq3") and sum(len(item) for item in reference_image_bytes) > self.HQ3_REFERENCE_BUDGET:
            raise ImageChannelError(f"{channel_label}的参考图总大小不能超过 50MB，请减少参考图。")
        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": self.FIXED_SIZE_MAP[ratio][resolution],
            "resolution": resolution.lower(),
            "quality": quality,
        }
        request_payload = dict(payload)
        if reference_image_bytes:
            payload["image_urls"] = await self.upload_reference_images(config, reference_image_bytes, jpeg=True)
            request_payload["image_urls"] = [
                f"<uploaded_reference_{index}>" for index in range(1, len(reference_image_bytes) + 1)
            ]
        endpoint_path = "/v1/images/generations"
        endpoint = self.append_query_param(build_url(config["base_url"], endpoint_path), "async", "true")
        # The alias picks the channel, so the key stays unsuffixed.
        headers = self.headers(config, {"Content-Type": "application/json"})
        # -hq3 holds the create open until the image is ready.
        submit_timeout = max(self.timeout(config), 600) if model.endswith("-hq3") else self.timeout(config)

        all_images, all_urls, task_ids, errors, responses = [], [], [], [], []
        for index in range(int(count)):
            task_id = ""
            try:
                response_payload = await submit_once(
                    self, endpoint, config, timeout=submit_timeout, headers=headers, json=payload,
                )
                task_id = self.extract_task_id(response_payload)
                if task_id:
                    task_ids.append(task_id)
                    self.publish_task_id(unique_id, task_id, label=f"Task ID #{index + 1}", append=index > 0)
                images, urls = await self.decode_images(response_payload, self.timeout(config))
                if images:
                    all_images.extend(images)
                    all_urls.extend(urls)
                    responses.append(self._redact_payload(response_payload))
                    continue
                if not task_id:
                    raise RuntimeError(f"创建响应缺少 task_id 和图像：{self._redact_payload(response_payload)}")
                if not bool(wait):
                    responses.append(self._redact_payload(response_payload))
                    continue
                images, urls, final_payload = await self.wait_for_task(config, task_id, max_poll_attempts, poll_interval)
                all_images.extend(images)
                all_urls.extend(urls)
                responses.append(self._redact_payload(final_payload))
            except ImageSubmitUncertain as exc:
                task_ids.append(exc.task_id)
                self.publish_task_id(unique_id, exc.task_id, label=f"Task ID #{index + 1}", append=index > 0)
                errors.append({"index": index, "task_id": exc.task_id, "error": str(exc)})
                break
            except Exception as exc:
                # Stop on the first failure: later images would be charged separately.
                errors.append({"index": index, "task_id": task_id, "error": str(exc)})
                break

        return {
            "images": all_images,
            "urls": all_urls,
            "task_ids": task_ids,
            "errors": errors,
            "responses": responses,
            "request": request_payload,
            "endpoint": endpoint_path + "?async=true",
            "reference_image_count": len(reference_image_bytes),
        }


__all__ = ["LingliuGptImage25"]
