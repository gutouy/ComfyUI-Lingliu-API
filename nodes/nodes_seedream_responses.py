import asyncio

from ..lingliu_config import LINGLIU_PROVIDER, build_url, provider_defaults
from ..utils import blank_image_tensor, content_to_text, image_to_png_bytes, parse_image_response, pil2tensor
from .nodes_openai_compatible import LingliuBaseNode


class LingliuPlatformBase(LingliuBaseNode):
    LINGLIU_BASE_URL = "https://api.lingliuapi.com"

    def load_lingliu_platform_config(self, api_key_override=""):
        runtime_config = self.load_config(api_key_override=api_key_override)
        config = provider_defaults(LINGLIU_PROVIDER)
        config["base_url"] = self.LINGLIU_BASE_URL
        config["api_provider"] = LINGLIU_PROVIDER
        config["api_key"] = str(runtime_config.get("api_key") or "").strip()
        config["timeout"] = int(runtime_config.get("timeout") or config.get("timeout") or 300)
        config["files_path"] = "/v1/files"
        config["responses_path"] = "/v1/responses"
        return config

    def endpoint_for_path(self, config, path):
        return build_url(config.get("base_url"), path)

    def extract_file_url(self, payload):
        if isinstance(payload, dict):
            for key in ("url", "download_url", "file_url"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            data = payload.get("data")
            if isinstance(data, dict):
                return self.extract_file_url(data)
        raise RuntimeError(f"/v1/files did not return a usable url: {payload}")

    async def upload_image_to_file_url(self, config, image, *, filename="image.png"):
        files = [
            (
                "file",
                (
                    filename,
                    image_to_png_bytes(image),
                    "image/png",
                ),
            )
        ]
        response_payload = await self.async_request_json_retrying(
            "POST",
            self.endpoint_for_path(config, "/v1/files"),
            config,
            max_retries=3,
            initial_timeout=min(self.timeout(config), 180),
            headers=self.headers(config),
            data={"purpose": "assistants"},
            files=files,
        )
        return self.extract_file_url(response_payload), response_payload

    async def upload_image_batch_to_urls(self, config, *images, max_images=16, prefix="image"):
        urls = []
        upload_payloads = []
        for image in images:
            if image is None:
                continue
            batch_count = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for index in range(batch_count):
                if len(urls) >= max_images:
                    raise ValueError(f"最多支持 {max_images} 张参考图。")
                single_image = image[index : index + 1] if batch_count > 1 else image
                url, payload = await self.upload_image_to_file_url(
                    config,
                    single_image,
                    filename=f"{prefix}_{len(urls) + 1}.png",
                )
                urls.append(url)
                upload_payloads.append(payload)
        return urls, upload_payloads


class LingliuSeedreamV5Image(LingliuPlatformBase):
    CATEGORY = "灵流API/图像"
    MODELS = ["seedream-v5-pro", "doubao-seedream-5-0-260128"]
    RESPONSE_FORMATS = ["url", "b64_json"]
    OUTPUT_FORMATS = ["png", "jpg"]
    SEQUENTIAL_MODES = ["disabled", "auto"]
    RATIO_CHOICES = ["自动", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21"]
    MODEL_RESOLUTIONS = {
        "seedream-v5-pro": ["1K", "2K"],
        "doubao-seedream-5-0-260128": ["2K", "3K", "4K"],
    }
    SIZE_BY_MODEL_RESOLUTION_RATIO = {
        "seedream-v5-pro": {
            "1K": {
                "1:1": "1024x1024",
                "4:3": "1152x864",
                "3:4": "864x1152",
                "3:2": "1248x832",
                "2:3": "832x1248",
                "16:9": "1424x800",
                "9:16": "800x1424",
                "21:9": "1568x672",
                "9:21": "672x1568",
            },
            "2K": {
                "1:1": "2048x2048",
                "4:3": "2368x1776",
                "3:4": "1776x2368",
                "3:2": "2496x1664",
                "2:3": "1664x2496",
                "16:9": "2816x1584",
                "9:16": "1584x2816",
                "21:9": "3136x1344",
                "9:21": "1344x3136",
            },
        },
        "doubao-seedream-5-0-260128": {
            "2K": {
                "1:1": "2048x2048",
                "4:3": "2368x1776",
                "3:4": "1776x2368",
                "3:2": "2496x1664",
                "2:3": "1664x2496",
                "16:9": "2720x1536",
                "9:16": "1536x2720",
                "21:9": "3136x1344",
                "9:21": "1344x3136",
            },
            "3K": {
                "1:1": "3072x3072",
                "4:3": "3456x2592",
                "3:4": "2592x3456",
                "3:2": "3744x2496",
                "2:3": "2496x3744",
                "16:9": "4096x2304",
                "9:16": "2304x4096",
                "21:9": "4704x2016",
                "9:21": "2016x4704",
            },
            "4K": {
                "1:1": "4096x4096",
                "4:3": "4704x3520",
                "3:4": "3520x4704",
                "3:2": "4992x3328",
                "2:3": "3328x4992",
                "16:9": "4096x2304",
                "9:16": "2304x4096",
                "21:9": "4704x2016",
                "9:21": "2016x4704",
            },
        },
    }
    MAX_REFERENCE_IMAGES = 14

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "api_key": ("STRING", {"default": ""}),
            "清晰度": (["1K", "2K", "3K", "4K"], {"default": "2K"}),
            "尺寸": (cls.RATIO_CHOICES, {"default": "自动"}),
            "response_format": (cls.RESPONSE_FORMATS, {"default": "url"}),
            "output_format": (cls.OUTPUT_FORMATS, {"default": "png"}),
            "水印": ("BOOLEAN", {"default": False}),
            "组图生成": (cls.SEQUENTIAL_MODES, {"default": "disabled"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
        }
        for index in range(1, cls.MAX_REFERENCE_IMAGES + 1):
            optional[f"参考图{index}"] = ("IMAGE",)
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "模型": (cls.MODELS, {"default": cls.MODELS[0]}),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图片URL", "原始响应", "实际尺寸")
    FUNCTION = "generate"

    def normalize_model(self, model):
        model = str(model or "").strip()
        return model if model in self.MODELS else self.MODELS[0]

    def normalize_resolution(self, model, resolution):
        model = self.normalize_model(model)
        allowed = self.MODEL_RESOLUTIONS[model]
        resolution = str(resolution or "").strip().upper()
        return resolution if resolution in allowed else allowed[0]

    def normalize_ratio(self, ratio):
        ratio = str(ratio or "").strip()
        return ratio if ratio in self.RATIO_CHOICES and ratio != "自动" else "自动"

    def resolve_size(self, model, resolution, ratio):
        model = self.normalize_model(model)
        resolution = self.normalize_resolution(model, resolution)
        ratio = self.normalize_ratio(ratio)
        if ratio == "自动":
            return resolution
        return self.SIZE_BY_MODEL_RESOLUTION_RATIO[model][resolution].get(ratio, resolution)

    def build_payload(
        self,
        *,
        prompt,
        model,
        resolution="2K",
        ratio="自动",
        reference_images=None,
        response_format="url",
        output_format="png",
        watermark=False,
        sequential_image_generation="disabled",
        seed=0,
    ):
        model = self.normalize_model(model)
        resolution = self.normalize_resolution(model, resolution)
        resolved_size = self.resolve_size(model, resolution, ratio)
        payload = {
            "model": model,
            "prompt": str(prompt or ""),
            "size": resolved_size,
            "response_format": str(response_format or "url").strip() or "url",
            "output_format": str(output_format or "png").strip() or "png",
            "watermark": bool(watermark),
        }
        references = list(reference_images or [])
        if references:
            payload["image"] = references
        sequential = str(sequential_image_generation or "").strip()
        if sequential and sequential != "disabled":
            payload["sequential_image_generation"] = sequential
        if int(seed) > 0:
            payload["seed"] = int(seed)
        return payload

    async def generate(
        self,
        提示词,
        模型,
        api_key="",
        清晰度="2K",
        尺寸="自动",
        response_format="url",
        output_format="png",
        水印=False,
        组图生成="disabled",
        seed=0,
        unique_id=None,
        **kwargs,
    ):
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_lingliu_platform_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "灵流 Seedream 节点需要 API 密钥。"
                return blank_image_tensor(), "", self.error_json(error), ""
            reference_urls, upload_payloads = await self.upload_image_batch_to_urls(
                config,
                *[kwargs.get(f"参考图{index}") for index in range(1, self.MAX_REFERENCE_IMAGES + 1)],
                max_images=self.MAX_REFERENCE_IMAGES,
                prefix="seedream_reference",
            )
            payload = self.build_payload(
                prompt=提示词,
                model=模型,
                resolution=清晰度,
                ratio=尺寸,
                reference_images=reference_urls,
                response_format=response_format,
                output_format=output_format,
                watermark=水印,
                sequential_image_generation=组图生成,
                seed=seed,
            )
            response_payload = await self.async_request_json_retrying(
                "POST",
                self.endpoint_for_path(config, "/v1/images/generations"),
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            images, urls = await asyncio.to_thread(parse_image_response, response_payload, self.timeout(config))
            if not images:
                raise RuntimeError(f"Seedream 未返回可解析图片：{response_payload}")
            response_size = ""
            if isinstance(response_payload, dict):
                items = response_payload.get("data")
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    response_size = str(items[0].get("size") or "")
            raw = {
                "request": {
                    "model": payload.get("model"),
                    "size": payload.get("size"),
                    "resolved_size": self.resolve_size(模型, 清晰度, 尺寸),
                    "resolution": self.normalize_resolution(模型, 清晰度),
                    "ratio": 尺寸,
                    "reference_image_count": len(reference_urls),
                    "reference_transport": "/v1/files url",
                    "response_format": payload.get("response_format"),
                    "output_format": payload.get("output_format"),
                    "watermark": payload.get("watermark"),
                    "sequential_image_generation": payload.get("sequential_image_generation", "disabled"),
                },
                "uploads": upload_payloads,
                "response": response_payload,
            }
            return pil2tensor(images), "\n".join(urls), self.safe_json(raw), response_size or raw["request"].get("resolved_size") or payload.get("size", "")
        except Exception as exc:
            return blank_image_tensor(), "", self.error_json(exc), ""


class LingliuResponsesVision(LingliuPlatformBase):
    CATEGORY = "灵流API/LLM"
    MAX_IMAGES = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "模型": ("STRING", {"default": "gpt-5.5"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "图像批次": ("IMAGE",),
                "系统提示词": ("STRING", {"default": "", "multiline": True}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "图像上限": ("INT", {"default": 16, "min": 1, "max": cls.MAX_IMAGES}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("文本", "原始响应")
    FUNCTION = "respond"

    def build_payload(
        self,
        *,
        prompt,
        model,
        system_prompt="",
        image_urls=None,
        max_output_tokens=2048,
        temperature=0.7,
        top_p=1.0,
    ):
        user_content = [{"type": "input_text", "text": str(prompt or "")}]
        for image_url in image_urls or []:
            user_content.append({"type": "input_image", "image_url": image_url})
        input_items = []
        if str(system_prompt or "").strip():
            input_items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": str(system_prompt).strip()}],
                }
            )
        input_items.append({"role": "user", "content": user_content})
        return {
            "model": str(model or "").strip(),
            "input": input_items,
            "max_output_tokens": int(max_output_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

    def extract_response_text(self, payload):
        if isinstance(payload, dict):
            if payload.get("output_text"):
                return str(payload.get("output_text") or "")
            texts = []
            for item in payload.get("output", []) or []:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []) or []:
                    if not isinstance(content, dict):
                        continue
                    if content.get("text"):
                        texts.append(str(content.get("text")))
                    elif content.get("type") in {"output_text", "text"} and content.get("value"):
                        texts.append(str(content.get("value")))
            if texts:
                return "\n".join(texts).strip()
            choices = payload.get("choices", [])
            if choices and isinstance(choices[0], dict):
                return content_to_text(choices[0].get("message", {}).get("content"))
        return content_to_text(payload)

    async def respond(
        self,
        提示词,
        模型,
        api_key="",
        图像批次=None,
        系统提示词="",
        max_output_tokens=2048,
        temperature=0.7,
        top_p=1.0,
        图像上限=16,
    ):
        try:
            config = self.load_lingliu_platform_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "灵流 Responses 节点需要 API 密钥。"
                return f"Error: {error}", self.error_json(error)
            image_urls, upload_payloads = await self.upload_image_batch_to_urls(
                config,
                图像批次,
                max_images=max(1, min(self.MAX_IMAGES, int(图像上限))),
                prefix="responses_reference",
            )
            payload = self.build_payload(
                prompt=提示词,
                model=模型,
                system_prompt=系统提示词,
                image_urls=image_urls,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            response_payload = await self.async_request_json_retrying(
                "POST",
                self.endpoint_for_path(config, "/v1/responses"),
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            text = self.extract_response_text(response_payload)
            if not text:
                raise RuntimeError(f"Responses API 未返回文本：{response_payload}")
            raw = {
                "request": {
                    "model": payload.get("model"),
                    "image_count": len(image_urls),
                    "image_transport": "/v1/files url",
                    "max_output_tokens": payload.get("max_output_tokens"),
                    "temperature": payload.get("temperature"),
                    "top_p": payload.get("top_p"),
                },
                "uploads": upload_payloads,
                "response": response_payload,
            }
            return text, self.safe_json(raw)
        except Exception as exc:
            return f"Error: {exc}", self.error_json(exc)
