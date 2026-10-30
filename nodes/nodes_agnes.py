import asyncio
import base64
from urllib.parse import urlencode

from comfy.comfy_types import IO
from PIL import Image

from ..lingliu_config import LINGLIU_PROVIDER, build_url, lingliu_channel_key, provider_defaults
from ..utils import (
    LingliuNullVideoAdapter,
    LingliuVideoAdapter,
    blank_image_tensor,
    content_to_text,
    image_to_data_url,
    image_to_png_bytes,
    load_image_from_candidate,
    parse_image_response,
    pil2tensor,
    tensor2pil,
)
from .nodes_openai_compatible import LingliuBaseNode


class LingliuAgnesBase(LingliuBaseNode):
    CATEGORY = "灵流API/Agnes"
    LINGLIU_BASE_URL = "https://api.lingliuapi.com"
    AGNES_CHANNEL_ID = 11
    ASPECT_RATIOS = ["1:1", "2:3", "3:4", "4:5", "9:16", "9:21", "3:2", "4:3", "5:4", "16:9", "21:9"]

    def load_lingliu_agnes_config(self, api_key_override=""):
        runtime_config = self.load_config(api_key_override=api_key_override)
        config = provider_defaults(LINGLIU_PROVIDER)
        config["base_url"] = self.LINGLIU_BASE_URL
        config["api_key"] = lingliu_channel_key(runtime_config.get("api_key", ""), self.AGNES_CHANNEL_ID)
        config["timeout"] = int(runtime_config.get("timeout") or config.get("timeout") or 300)
        config["chat_path"] = "/v1/chat/completions"
        config["images_generate_path"] = "/v1/images/generations"
        config["videos_generate_path"] = "/v1/videos"
        config["videos_status_path_template"] = "/v1/videos/{task_id}"
        config["agnes_video_status_path_template"] = "/agnesapi?{query}"
        return config

    def endpoint_from_template(self, config, template_key, **values):
        template = str(config.get(template_key) or "").strip()
        return build_url(config.get("base_url"), template.format(**values))

    def compute_size(self, quality, aspect_ratio, *, quality_map=None):
        quality_map = quality_map or {"1K": 1024}
        base = int(quality_map.get(str(quality or "").strip(), next(iter(quality_map.values()))))
        ratio = str(aspect_ratio or "1:1").strip()
        if ratio not in self.ASPECT_RATIOS:
            ratio = "1:1"
        width_ratio, height_ratio = [int(part) for part in ratio.split(":", 1)]
        if width_ratio >= height_ratio:
            height = base
            width = base * width_ratio // height_ratio
        else:
            width = base
            height = base * height_ratio // width_ratio
        width = max(64, (width // 8) * 8)
        height = max(64, (height // 8) * 8)
        return f"{width}x{height}"

    def normalize_size_text(self, size):
        raw = str(size or "").strip().lower()
        if "x" not in raw:
            return ""
        left, right = raw.split("x", 1)
        try:
            width = max(64, (int(left.strip()) // 8) * 8)
            height = max(64, (int(right.strip()) // 8) * 8)
        except ValueError:
            return ""
        return f"{width}x{height}"

    def resolve_size(self, *, quality, aspect_ratio, size="", quality_map=None):
        manual_size = self.normalize_size_text(size)
        if manual_size and manual_size != "0x0":
            return manual_size
        return self.compute_size(quality, aspect_ratio, quality_map=quality_map)

    def parse_size_text(self, size):
        normalized = self.normalize_size_text(size)
        if not normalized:
            raise ValueError(f"Invalid size: {size}")
        width_text, height_text = normalized.split("x", 1)
        return int(width_text), int(height_text)

    def fit_pil_to_size(self, image, width, height, background="white"):
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, background)
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        else:
            image = image.convert("RGB")
        src_width, src_height = image.size
        scale = min(width / src_width, height / src_height)
        fitted_width = max(1, int(round(src_width * scale)))
        fitted_height = max(1, int(round(src_height * scale)))
        image = image.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), background)
        left = (width - fitted_width) // 2
        top = (height - fitted_height) // 2
        canvas.paste(image, (left, top))
        return canvas

    def collect_image_data_urls(self, *images, max_images=4, target_size="", normalize=False):
        references = []
        target_width = target_height = None
        if normalize and target_size:
            target_width, target_height = self.parse_size_text(target_size)
        for image in images:
            if image is None:
                continue
            batch_count = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for index in range(batch_count):
                if len(references) >= max_images:
                    raise ValueError(f"Agnes supports at most {max_images} reference images.")
                single_image = image[index : index + 1] if batch_count > 1 else image
                if normalize and target_width and target_height:
                    single_image = self.fit_pil_to_size(tensor2pil(single_image)[0], target_width, target_height)
                references.append(image_to_data_url(single_image))
        return references

    def collect_image_base64_values(self, *images, max_images=4):
        references = []
        for image in images:
            if image is None:
                continue
            batch_count = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for index in range(batch_count):
                if len(references) >= max_images:
                    raise ValueError(f"Agnes supports at most {max_images} reference images.")
                single_image = image[index : index + 1] if batch_count > 1 else image
                references.append(base64.b64encode(image_to_png_bytes(single_image)).decode("ascii"))
        return references

    def first_video_url(self, payload):
        _image_urls, video_urls = self.extract_media_urls(payload)
        return video_urls[0] if video_urls else ""


class LingliuAgnesChat(LingliuAgnesBase):
    MODELS = ["agnes-2.0-flash"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant.", "multiline": True}),
                "image_url": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "raw_json")
    FUNCTION = "chat"

    def build_user_content(self, prompt, image_url="", image=None):
        references = []
        if str(image_url or "").strip():
            references.append(str(image_url).strip())
        if image is not None:
            references.extend(self.collect_image_data_urls(image, max_images=1))
        if not references:
            return str(prompt or "")
        content = [{"type": "text", "text": str(prompt or "")}]
        for url in references:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def build_chat_payload(
        self,
        *,
        prompt,
        system_prompt,
        model,
        temperature,
        top_p,
        max_tokens,
        seed=0,
        image_url="",
        image=None,
        enable_thinking=False,
    ):
        messages = []
        if str(system_prompt or "").strip():
            messages.append({"role": "system", "content": str(system_prompt).strip()})
        messages.append({"role": "user", "content": self.build_user_content(prompt, image_url=image_url, image=image)})
        payload = {
            "model": str(model or "").strip() or self.MODELS[0],
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if int(seed) > 0:
            payload["seed"] = int(seed)
        if enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        return payload

    async def chat(
        self,
        prompt,
        model,
        api_key="",
        system_prompt="You are a helpful AI assistant.",
        image_url="",
        image=None,
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        seed=0,
        enable_thinking=False,
    ):
        try:
            config = self.load_lingliu_agnes_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_agnes_chat."
                return f"Error: {error}", self.error_json(error)
            payload = self.build_chat_payload(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                image_url=image_url,
                image=image,
                enable_thinking=enable_thinking,
            )
            response_payload = await self.async_request_json(
                "POST",
                self.endpoint(config, "chat_path"),
                config,
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            choices = response_payload.get("choices", []) if isinstance(response_payload, dict) else []
            if not choices:
                raise RuntimeError(f"No choices returned from Agnes chat endpoint: {response_payload}")
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            response_text = content_to_text(message.get("content"))
            if not response_text and isinstance(choices[0], dict) and choices[0].get("text"):
                response_text = str(choices[0]["text"])
            return response_text, self.safe_json({"request": self.public_payload(payload), "response": response_payload})
        except Exception as exc:
            return f"Error: {exc}", self.error_json(exc)


class LingliuAgnesImageReverse(LingliuAgnesBase):
    MODELS = ["agnes-2.0-flash"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 65536}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "brief_prompt", "raw_json")
    FUNCTION = "reverse"

    def build_reverse_payload(self, *, image, model, detail="detailed", temperature=0.3, max_tokens=2048):
        image_urls = self.collect_image_data_urls(image, max_images=1)
        if not image_urls:
            raise ValueError("Image is required for Agnes reverse prompt.")
        if detail == "brief":
            system_prompt = (
                "You are an expert at analyzing images and writing prompts for AI image generation models. "
                "Write a concise prompt describing the key elements of this image. "
                "Output ONLY the prompt, no additional commentary."
            )
        else:
            system_prompt = (
                "You are an expert at analyzing images and writing prompts for AI image generation models. "
                "Describe the image in extreme detail: subject, composition, lighting, color palette, style, mood, "
                "camera angle, depth of field, textures, and any distinctive elements. "
                "Output ONLY the prompt, no additional commentary."
            )
        return {
            "model": str(model or "").strip() or self.MODELS[0],
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_urls[0]}},
                        {"type": "text", "text": "Please describe this image as an AI image generation prompt."},
                    ],
                },
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

    def response_text(self, payload):
        choices = payload.get("choices", []) if isinstance(payload, dict) else []
        if not choices:
            raise RuntimeError(f"No choices returned from Agnes reverse prompt endpoint: {payload}")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        text = content_to_text(message.get("content"))
        if not text and isinstance(choices[0], dict) and choices[0].get("text"):
            text = str(choices[0]["text"])
        return text

    async def reverse(self, image, model, api_key="", temperature=0.3, max_tokens=2048):
        try:
            config = self.load_lingliu_agnes_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_agnes_image_reverse."
                return f"Error: {error}", f"Error: {error}", self.error_json(error)
            detailed_payload = self.build_reverse_payload(
                image=image,
                model=model,
                detail="detailed",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            brief_payload = self.build_reverse_payload(
                image=image,
                model=model,
                detail="brief",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            detailed_response = await self.async_request_json(
                "POST",
                self.endpoint(config, "chat_path"),
                config,
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=detailed_payload,
            )
            brief_response = await self.async_request_json(
                "POST",
                self.endpoint(config, "chat_path"),
                config,
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=brief_payload,
            )
            detailed = self.response_text(detailed_response)
            brief = self.response_text(brief_response)
            raw = {
                "detailed": {"request": self.public_payload(detailed_payload), "response": detailed_response},
                "brief": {"request": self.public_payload(brief_payload), "response": brief_response},
            }
            return detailed, brief, self.safe_json(raw)
        except Exception as exc:
            return f"Error: {exc}", f"Error: {exc}", self.error_json(exc)


class LingliuAgnesImage(LingliuAgnesBase):
    MODELS = ["agnes-image-2.1-flash", "agnes-image-2.0-flash"]
    QUALITY_MAP = {"1K": 1024, "2K": 2048, "4K": 4096}
    QUALITIES = ["1K", "2K", "4K"]
    MAX_REFERENCE_IMAGES = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "quality": (cls.QUALITIES, {"default": "1K"}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "1:1"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "size": ("STRING", {"default": "", "multiline": False}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "normalize_reference": ("BOOLEAN", {"default": True}),
                "reference_image1": ("IMAGE",),
                "reference_image2": ("IMAGE",),
                "reference_image3": ("IMAGE",),
                "reference_image4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response", "resolution")
    FUNCTION = "generate"

    def normalize_size_choice(self, size, quality="1K", aspect_ratio="1:1"):
        return self.resolve_size(
            quality=quality,
            aspect_ratio=aspect_ratio,
            size=size,
            quality_map=self.QUALITY_MAP,
        )

    def build_image_payload(
        self,
        *,
        prompt,
        model,
        quality="1K",
        aspect_ratio="1:1",
        size="",
        reference_images=None,
        seed=0,
        response_format="url",
        n=1,
        strength=1.0,
    ):
        references = list(reference_images or [])
        resolved_size = self.normalize_size_choice(size, quality=quality, aspect_ratio=aspect_ratio)
        prompt_text = str(prompt or "")
        if references and float(strength) < 1.0:
            prompt_text = (
                f"{prompt_text}\n"
                f"[Guidance: transformation strength = {float(strength):.0%}. "
                "Lower strength means stay closer to the original.]"
            )
        payload = {
            "model": str(model or "").strip() or self.MODELS[0],
            "prompt": prompt_text,
            "size": resolved_size,
            "n": max(1, min(4, int(n))),
        }
        extra_body = {}
        if references:
            extra_body["image"] = references
        response_format = str(response_format or "").strip() or "url"
        if response_format == "b64_json" and not references:
            payload["return_base64"] = True
        elif response_format:
            extra_body["response_format"] = response_format
        if extra_body:
            payload["extra_body"] = extra_body
        if int(seed) > 0 and not references:
            payload["seed"] = int(seed)
        return payload

    async def generate(
        self,
        prompt,
        model,
        quality="1K",
        aspect_ratio="1:1",
        api_key="",
        response_format="url",
        size="",
        n=1,
        strength=1.0,
        normalize_reference=True,
        reference_image1=None,
        reference_image2=None,
        reference_image3=None,
        reference_image4=None,
        seed=0,
        unique_id=None,
    ):
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_lingliu_agnes_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_agnes_image."
                return blank_image_tensor(), "", "", self.error_json(error), ""
            resolved_size = self.normalize_size_choice(size, quality=quality, aspect_ratio=aspect_ratio)
            references = self.collect_image_data_urls(
                reference_image1,
                reference_image2,
                reference_image3,
                reference_image4,
                max_images=self.MAX_REFERENCE_IMAGES,
                target_size=resolved_size,
                normalize=bool(normalize_reference),
            )
            payload = self.build_image_payload(
                prompt=prompt,
                model=model,
                quality=quality,
                aspect_ratio=aspect_ratio,
                size=size,
                reference_images=references,
                seed=seed,
                response_format=response_format,
                n=n,
                strength=strength,
            )
            response_payload = await self.async_request_json_retrying(
                "POST",
                self.endpoint(config, "images_generate_path"),
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            images, urls = await asyncio.to_thread(parse_image_response, response_payload, self.timeout(config))
            if not images:
                raise RuntimeError(f"Agnes image response did not contain an image: {response_payload}")
            task_id = self.extract_task_id(response_payload)
            if task_id:
                self.publish_task_id(unique_id, task_id)
            response_info = {
                "request": {
                    "model": payload.get("model", ""),
                    "size": payload.get("size", ""),
                    "quality": quality,
                    "aspect_ratio": aspect_ratio,
                    "n": payload.get("n"),
                    "reference_image_count": len(references),
                    "normalize_reference": bool(normalize_reference),
                    "response_format": response_format,
                },
                "response": response_payload,
            }
            return pil2tensor(images), "\n".join(urls), task_id, self.safe_json(response_info), payload.get("size", "")
        except Exception as exc:
            return blank_image_tensor(), "", "", self.error_json(exc), ""


class LingliuAgnesVideo(LingliuAgnesBase):
    MODELS = ["agnes-video-v2.0"]
    QUALITY_MAP = {"1K": 1024, "2K": 2048}
    QUALITIES = ["1K", "2K"]
    MODES = ["auto", "ti2vid", "multi_reference", "keyframes"]
    MAX_REFERENCE_IMAGES = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "quality": (cls.QUALITIES, {"default": "1K"}),
                "aspect_ratio": (cls.ASPECT_RATIOS, {"default": "16:9"}),
                "mode": (cls.MODES, {"default": cls.MODES[0]}),
                "num_frames": ("INT", {"default": 121, "min": 9, "max": 441, "step": 8}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "reference_image_url": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "", "multiline": False}),
                "reference_image1": ("IMAGE",),
                "reference_image2": ("IMAGE",),
                "reference_image3": ("IMAGE",),
                "reference_image4": ("IMAGE",),
                "wait_for_completion": ("BOOLEAN", {"default": True}),
                "max_poll_attempts": ("INT", {"default": 180, "min": 1, "max": 720}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 30}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response", "video_url", "resolution")
    FUNCTION = "generate"

    def normalize_size_choice(self, size, quality="1K", aspect_ratio="16:9"):
        return self.resolve_size(
            quality=quality,
            aspect_ratio=aspect_ratio,
            size=size,
            quality_map=self.QUALITY_MAP,
        )

    def parse_size(self, size, quality="1K", aspect_ratio="16:9"):
        raw = self.normalize_size_choice(size, quality=quality, aspect_ratio=aspect_ratio)
        return self.parse_size_text(raw)

    def normalize_num_frames(self, num_frames):
        value = max(9, min(441, int(num_frames)))
        remainder = (value - 1) % 8
        if remainder:
            value -= remainder
        return max(9, value)

    def normalize_mode(self, mode, reference_count):
        raw = str(mode or "auto").strip().lower() or "auto"
        aliases = {
            "text_to_video": "auto",
            "txt2vid": "auto",
            "t2v": "auto",
            "image_to_video": "ti2vid",
            "img2vid": "ti2vid",
            "i2v": "ti2vid",
            "multi_image": "multi_reference",
            "multi-reference": "multi_reference",
        }
        raw = aliases.get(raw, raw)
        if raw == "auto":
            if reference_count <= 0:
                return ""
            if reference_count == 1:
                return "ti2vid"
            return "multi_reference"
        if raw in {"ti2vid", "multi_reference", "keyframes"}:
            return raw
        return ""

    def build_video_payload(
        self,
        *,
        model,
        prompt,
        quality="1K",
        aspect_ratio="16:9",
        size="",
        reference_images=None,
        mode="text_to_video",
        num_frames=121,
        frame_rate=24,
        num_inference_steps=0,
        seed=0,
        negative_prompt="",
    ):
        resolved_size = self.normalize_size_choice(size, quality=quality, aspect_ratio=aspect_ratio)
        width, height = self.parse_size(resolved_size, quality=quality, aspect_ratio=aspect_ratio)
        references = [str(item).strip() for item in (reference_images or []) if str(item).strip()]
        normalized_mode = self.normalize_mode(mode, len(references))
        payload = {
            "model": str(model or "").strip() or self.MODELS[0],
            "prompt": str(prompt or ""),
            "width": width,
            "height": height,
            "num_frames": self.normalize_num_frames(num_frames),
            "frame_rate": float(frame_rate),
        }
        if references:
            payload["mode"] = normalized_mode or "ti2vid"
            if payload["mode"] == "ti2vid":
                payload["image"] = references[0]
            else:
                payload["image"] = references[0]
                payload["extra_body"] = {"image": references, "mode": payload["mode"]}
        elif normalized_mode:
            payload["mode"] = normalized_mode
        return payload

    async def wait_for_video_task(self, config, task_id, video_id="", *, model="agnes-video-v2.0", max_poll_attempts=180, poll_interval=5):
        last_payload = None
        for _ in range(int(max_poll_attempts)):
            await asyncio.sleep(max(1, int(poll_interval)))
            urls_to_try = []
            if video_id:
                query = urlencode({"video_id": video_id, "model_name": model})
                urls_to_try.append(self.endpoint_from_template(config, "agnes_video_status_path_template", query=query))
            urls_to_try.append(self.endpoint_from_template(config, "videos_status_path_template", task_id=task_id))
            poll_errors = []
            for task_url in urls_to_try:
                try:
                    payload = await self.async_request_json("GET", task_url, config, headers=self.headers(config))
                    last_payload = payload
                    break
                except Exception as exc:
                    poll_errors.append(str(exc))
                    payload = None
            if payload is None:
                if poll_errors:
                    last_payload = {"poll_errors": poll_errors, "task_id": task_id, "video_id": video_id}
                continue
            video_url = self.first_video_url(payload)
            status = self.extract_status(payload)
            if video_url and (not status or self.is_success_status(status) or str(self.extract_progress(payload)) == "100"):
                return payload, video_url
            if self.is_failure_status(status):
                detail = self.extract_failure_detail(payload)
                raise RuntimeError(f"Agnes video task failed: {detail or payload}")
        raise TimeoutError(f"Agnes video task {task_id or video_id} timed out. Last payload: {last_payload}")

    async def generate(
        self,
        prompt,
        model,
        quality="1K",
        aspect_ratio="16:9",
        mode="text_to_video",
        num_frames=121,
        frame_rate=24,
        api_key="",
        negative_prompt="",
        reference_image_url="",
        size="",
        reference_image1=None,
        reference_image2=None,
        reference_image3=None,
        reference_image4=None,
        num_inference_steps=0,
        seed=0,
        wait_for_completion=True,
        max_poll_attempts=180,
        poll_interval=5,
        unique_id=None,
    ):
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_lingliu_agnes_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_agnes_video."
                return LingliuNullVideoAdapter(error), "", self.error_json(error), "", ""
            resolved_size = self.normalize_size_choice(size, quality=quality, aspect_ratio=aspect_ratio)
            references = []
            if str(reference_image_url or "").strip():
                references.append(str(reference_image_url).strip())
            references.extend(
                self.collect_image_base64_values(
                    reference_image1,
                    reference_image2,
                    reference_image3,
                    reference_image4,
                    max_images=self.MAX_REFERENCE_IMAGES,
                )
            )
            payload = self.build_video_payload(
                model=model,
                prompt=prompt,
                quality=quality,
                aspect_ratio=aspect_ratio,
                size=size,
                reference_images=references,
                mode=mode,
                num_frames=num_frames,
                frame_rate=frame_rate,
            )
            submit_payload = await self.async_request_json_retrying(
                "POST",
                self.endpoint(config, "videos_generate_path"),
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            task_id = self.extract_task_id(submit_payload)
            video_id = str(submit_payload.get("video_id", "") or "").strip() if isinstance(submit_payload, dict) else ""
            if task_id:
                self.publish_task_id(unique_id, task_id)
            video_url = self.first_video_url(submit_payload)
            final_payload = submit_payload
            if wait_for_completion and not video_url:
                if not task_id and not video_id:
                    raise RuntimeError(f"Agnes video submit response did not include task_id or video_id: {submit_payload}")
                final_payload, video_url = await self.wait_for_video_task(
                    config,
                    task_id,
                    video_id=video_id,
                    model=str(model or "").strip() or self.MODELS[0],
                    max_poll_attempts=max_poll_attempts,
                    poll_interval=poll_interval,
                )
            response_info = {
                "request": {
                    "model": payload.get("model", ""),
                    "size": resolved_size,
                    "quality": quality,
                    "aspect_ratio": aspect_ratio,
                    "mode": payload.get("mode", ""),
                    "reference_image_count": len(references),
                    "num_frames": payload.get("num_frames"),
                    "frame_rate": payload.get("frame_rate"),
                },
                "submit": submit_payload,
                "final": final_payload,
            }
            if not video_url:
                return LingliuNullVideoAdapter("No video URL yet."), task_id or video_id, self.safe_json(response_info), "", resolved_size
            response_info["video_url"] = video_url
            return LingliuVideoAdapter(video_url, mime_type="video/mp4"), task_id or video_id, self.safe_json(response_info), video_url, resolved_size
        except Exception as exc:
            return LingliuNullVideoAdapter(str(exc)), "", self.error_json(exc), "", ""
