import asyncio
import base64
import json
import time
from io import BytesIO

import comfy.utils
import requests
import torch
from PIL import Image
from comfy.comfy_types import IO

from ..lingliu_config import (
    LINGLIU_PROVIDER,
    PUBLIC_PROVIDER_DISPLAY_CHOICES,
    build_url,
    get_config,
    mask_provider_text,
    normalize_provider,
    provider_display_name,
)
from ..utils import (
    LingliuNullVideoAdapter,
    LingliuVideoAdapter,
    blank_image_tensor,
    download_image_content,
    extract_image_candidates_from_text,
    image_to_data_url,
    image_to_png_bytes,
    load_image_from_candidate,
    pil2tensor,
    tensor2pil,
    video_to_data_url,
    video_to_remote_url,
)
from .nodes_openai_compatible import LingliuBaseNode
from .lingliu_reference_upload import log as log_reference, upload_references
from .lingliu_image_channels import (
    CHANNELS,
    CHANNEL_DEFAULT_LEGACY,
    CHANNEL_DEFAULT_NEW_NODE,
    CHANNEL_JL,
    ImageChannelError,
    ImageSubmitUncertain,
    config_with_channel,
    image_tensor_size,
    jl_contract,
    jl_reference_bytes,
    nearest_ratio,
    parse_flat_image_task,
    resolve_route,
    submit_once,
    validate_jl_ratio,
    validate_jl_references,
    validate_jl_resolution,
    wait_flat_image_task,
)


class LingliuGoogleBase(LingliuBaseNode):
    CATEGORY = "Lingliu/Google"
    IMAGE_TASK_MAX_RETRIES = 360
    VIDEO_TASK_MAX_RETRIES = 360
    POLL_INTERVAL_SECONDS = 2

    def default_image(self, *images, width=1024, height=1024):
        for image in images:
            if image is not None:
                if hasattr(image, "shape") and len(image.shape) == 4 and image.shape[0] > 0:
                    return image[:1]
                return image
        return blank_image_tensor(width=width, height=height)

    def auth_headers(self, config, json_mode=False):
        headers = self.headers(config)
        if json_mode:
            headers["Content-Type"] = "application/json"
        return headers

    def task_endpoint(self, config, template_key, task_id):
        template = config.get(template_key, "")
        return build_url(config.get("base_url"), template.format(task_id=task_id))

    async def async_request_json(self, method, url, config, **kwargs):
        return await asyncio.to_thread(self.request_json, method, url, config, **kwargs)

    def is_deadline_retryable_failure(self, payload, reason=""):
        detail = " ".join(
            str(part or "")
            for part in (
                reason,
                self.extract_failure_detail(payload),
                payload,
            )
        ).lower()
        retry_tokens = (
            "deadline",
            "deadline exceeded",
            "timed out",
            "timeout",
            "operation completed before deadline",
            "\u622a\u6b62\u65f6\u95f4",
            "\u64cd\u4f5c\u5b8c\u6210\u524d",
            "\u5df2\u8fc7\u671f",
        )
        return any(token in detail for token in retry_tokens)

    async def wait_for_image_task(self, config, task_id, timeout=600, max_retries=None):

        if max_retries is None:
            max_retries = self.IMAGE_TASK_MAX_RETRIES
        last_payload = None
        consecutive_poll_errors = 0
        for _ in range(max_retries):
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
            try:
                payload = await self.async_request_json(
                    "GET",
                    self.task_endpoint(config, "images_task_status_path_template", task_id),
                    config,
                    headers=self.auth_headers(config),
                )
                consecutive_poll_errors = 0
            except Exception as exc:
                if self.is_transient_request_error(exc):
                    consecutive_poll_errors += 1
                    last_payload = {"warning": str(exc), "task_id": task_id, "transient_poll_errors": consecutive_poll_errors}
                    if consecutive_poll_errors < 12:
                        continue
                raise
            last_payload = payload
            # Platform-native image tasks (JL) use a flat envelope:
            # {"status": "completed", "data": [{"url": ...}]}.
            flat = parse_flat_image_task(payload)
            if flat is not None:
                flat_state, flat_items, flat_error = flat
                if flat_state == "success" and flat_items:
                    return flat_items, payload
                if flat_state == "failure":
                    raise RuntimeError(f"Task failed: {flat_error}")
                continue
            outer_data = payload.get("data", {}) if isinstance(payload, dict) else {}
            status = outer_data.get("status", "")
            if status == "SUCCESS":
                inner_data = outer_data.get("data", {})
                return inner_data.get("data", []), payload
            if status == "FAILURE":
                fail_reason = outer_data.get("fail_reason", "Unknown error")
                if self.is_deadline_retryable_failure(outer_data, fail_reason):
                    continue
                raise RuntimeError(f"Task failed: {fail_reason}")
        raise TimeoutError(
            f"Task {task_id} timed out after {max_retries} polls "
            f"({max_retries * self.POLL_INTERVAL_SECONDS} seconds). Last payload: {last_payload}"
        )

    async def decode_image_items(self, image_items, timeout=600):
        tensors = {}
        first_image_url = ""
        url_items = []
        b64_items = []

        for index, item in enumerate(image_items):
            if not isinstance(item, dict):
                continue
            if item.get("b64_json"):
                b64_items.append((index, item["b64_json"]))
            elif item.get("url"):
                url_items.append((index, item["url"]))
                if not first_image_url:
                    first_image_url = item["url"]

        for index, b64_data in b64_items:
            raw = base64.b64decode(b64_data)
            image = Image.open(BytesIO(raw)).convert("RGB")
            tensors[index] = pil2tensor(image)
            if not first_image_url:
                first_image_url = f"data:image/png;base64,{b64_data}"

        if url_items:
            tasks = [asyncio.to_thread(download_image_content, url, timeout) for _, url in url_items]
            contents = await asyncio.gather(*tasks)
            for (index, url), content in zip(url_items, contents):
                image = Image.open(BytesIO(content)).convert("RGB")
                tensors[index] = pil2tensor(image)
                if not first_image_url:
                    first_image_url = url

        ordered = [tensors[i] for i in sorted(tensors)]
        if not ordered:
            raise RuntimeError("Failed to decode any generated images from task result.")
        return torch.cat(ordered, dim=0), first_image_url


def current_model_provider_default():
    return provider_display_name("lingliu")


BANANA_EDIT_MODEL_GROUPS = {
    "lingliu": [
        "nano-banana",
        "nano-banana-hd",
    ],
}
BANANA_EDIT_MODEL_CHOICES = list(dict.fromkeys(
    BANANA_EDIT_MODEL_GROUPS["lingliu"]
))


BANANA2_EDIT_MODEL_GROUPS = {
    "lingliu": [
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-image",
        "nano-banana-2",
        "nano-banana-2-2k",
        "nano-banana-2-4k",
        "nano-banana-2.1",
        "nano-banana-pro",
        "nano-banana-pro-2k",
        "nano-banana-pro-4k",
        "gemini-3.1-flash-image-preview-2k",
        "gemini-3.1-flash-image-preview-4k",
        "gemini-3.1-flash-image-2k",
        "gemini-3.1-flash-image-4k",
    ],
}
BANANA2_EDIT_MODEL_CHOICES = list(dict.fromkeys(
    BANANA2_EDIT_MODEL_GROUPS["lingliu"]
))


class LingliuGoogleVeo3(LingliuGoogleBase):
    LINGLIU_VEO_MODELS = [
        "veo3.1-lite",
        "veo3.1-fast",
        "veo3.1",
        "veo3.1-pro",
        "veo3.1-4k",
        "veo3.1-pro-4k",
        "veo3.1-components",
        "veo3.1-components-4k",
        "veo3",
        "veo3-fast",
        "veo3-pro",
        "veo3-fast-frames",
        "veo3-pro-frames",
        "fal-ai/veo3.1",
        "fal-ai/veo3.1/fast",
    ]
    LINGLIU_VEO_MODEL_ALIASES = {
        "veo3": "veo3.1",
        "veo3-fast": "veo3.1-fast",
        "veo3-pro": "veo3.1-pro",
        "veo3-fast-frames": "veo3.1",
        "veo3-pro-frames": "veo3.1-pro",
        "veo3.1-4K": "veo3.1-4k",
        "veo3.1-components-4K": "veo3.1-components-4k",
        "veo3.1-fast-4K": "veo3.1-fast-4k",
        "veo3.1-fast-components": "veo3.1-components",
        "veo3.1-fast-components-4K": "veo3.1-components-4k",
        "veo3.1-fast-components-4k": "veo3.1-components-4k",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.LINGLIU_VEO_MODELS, {"default": "veo3.1-fast"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "duration": (["4s", "6s", "8s"], {"default": "8s"}),
                "resolution": (["720p", "1080p", "4k"], {"default": "720p"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "4"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "enable_upsample": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = "Lingliu/Google"

    SUPPORTED_RATIO_MODELS = {
        "veo3",
        "veo3-fast",
        "veo3-pro",
        "veo3-fast-frames",
        "veo3-pro-frames",
        "veo3.1",
        "veo3.1-fast",
        "veo3.1-pro",
        "veo3.1-lite",
        "veo3.1-components",
        "veo3.1-4k",
        "veo3.1-pro-4k",
        "veo3.1-components-4k",
        "veo3.1-fast-4k",
        "veo3.1-fast-components",
        "veo3.1-fast-components-4k",
    }

    def normalize_lingliu_veo_model(self, model):
        raw = str(model or "").strip()
        if raw.startswith("fal-ai/veo3.1"):
            return raw
        return self.LINGLIU_VEO_MODEL_ALIASES.get(raw, raw)

    def extract_video_generation_url(self, payload):
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                for key in ("output", "video_url", "url", "output_url"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            for key in ("output", "video_url", "url", "output_url"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        _image_urls, video_urls = self.extract_media_urls(payload)
        return video_urls[0] if video_urls else ""

    async def generate_video(
        self,
        prompt,
        model="veo3",
        enhance_prompt=False,
        aspect_ratio="16:9",
        api_key="",
        image1=None,
        image2=None,
        image3=None,
        duration="8s",
        resolution="720p",
        generate_audio=True,
        safety_tolerance="4",
        seed=0,
        enable_upsample=False,
        unique_id=None,
    ):
        task_id = ""
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_google_veo3."
                return LingliuNullVideoAdapter(error), "", self.safe_json({"error": error})



            submitted_model = self.normalize_lingliu_veo_model(model)
            payload = {
                "prompt": prompt,
                "model": submitted_model,
                "enhance_prompt": bool(enhance_prompt),
            }
            if int(seed) > 0:
                payload["seed"] = int(seed)
            if submitted_model in self.SUPPORTED_RATIO_MODELS:
                payload["aspect_ratio"] = aspect_ratio
                if enable_upsample:
                    payload["enable_upsample"] = True

            image_data = []
            for image in [image1, image2, image3]:
                if image is None:
                    continue
                batch_size = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
                for batch_index in range(batch_size):
                    slice_image = image[batch_index : batch_index + 1] if batch_size > 1 else image
                    encoded = base64.b64encode(image_to_png_bytes(slice_image)).decode("utf-8")
                    image_data.append(f"data:image/png;base64,{encoded}")
            if image_data:
                payload["images"] = image_data

            submit_payload = await self.async_request_json(
                "POST",
                self.endpoint(config, "videos_generate_path"),
                config,
                headers=self.auth_headers(config, json_mode=True),
                json=payload,
            )
            task_id = submit_payload.get("task_id", "")
            if not task_id:
                raise RuntimeError(f"No task_id returned from video generation endpoint: {submit_payload}")
            self.publish_task_id(unique_id, task_id)

            video_url = ""
            last_payload = None
            consecutive_poll_errors = 0
            for _ in range(self.VIDEO_TASK_MAX_RETRIES):
                await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
                try:
                    last_payload = await self.async_request_json(
                        "GET",
                        self.task_endpoint(config, "videos_status_path_template", task_id),
                        config,
                        headers=self.auth_headers(config),
                    )
                    consecutive_poll_errors = 0
                except Exception as exc:
                    if self.is_transient_request_error(exc):
                        consecutive_poll_errors += 1
                        last_payload = {
                            "warning": str(exc),
                            "task_id": task_id,
                            "transient_poll_errors": consecutive_poll_errors,
                        }
                        if consecutive_poll_errors < 12:
                            continue
                    raise
                status = self.extract_status(last_payload) or last_payload.get("status", "")
                if self.is_success_status(status):
                    video_url = self.extract_video_generation_url(last_payload)
                    if video_url:
                        break
                if self.is_failure_status(status):
                    fail_reason = self.extract_failure_detail(last_payload) or last_payload.get("fail_reason", "Unknown error")
                    if self.is_deadline_retryable_failure(last_payload, fail_reason):
                        continue
                    raise RuntimeError(f"Video generation failed: {fail_reason}")
            if not video_url:
                raise TimeoutError(
                    f"Failed to fetch generated video URL for task {task_id} after "
                    f"{self.VIDEO_TASK_MAX_RETRIES * self.POLL_INTERVAL_SECONDS} seconds. "
                    f"Last payload: {last_payload}"
                )

            response_summary = {
                "task_id": task_id,
                "requested_model": model,
                "model": submitted_model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio if submitted_model in self.SUPPORTED_RATIO_MODELS else "",
                "enable_upsample": bool(enable_upsample) if submitted_model in self.SUPPORTED_RATIO_MODELS else False,
                "video_url": video_url,
                "images_count": len(image_data),
            }
            return LingliuVideoAdapter(video_url), video_url, self.safe_json(response_summary)
        except Exception as exc:
            return (
                LingliuNullVideoAdapter(mask_provider_text(exc)),
                "",
                self.safe_json({"error": str(exc), "task_id": task_id}),
            )


class LingliuNanoBanana(LingliuGoogleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model": (
                    [
                        "nano-banana-2",
                        "gemini-3-pro-image-preview",
                        "gemini-3.1-flash-image",
                        "gemini-2.5-flash-image",
                        "nano-banana",
                        "nano-banana-hd",
                        "gemini-2.5-flash-image-preview",
                    ],
                    {"default": "nano-banana"},
                ),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_tokens": ("INT", {"default": 32768, "min": 1, "max": 32768}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "process"
    CATEGORY = "Lingliu/Google"

    def _stream_chat_text_sync(self, config, payload):
        full_response = ""
        response = requests.post(
            self.endpoint(config, "chat_path"),
            headers=self.auth_headers(config, json_mode=True),
            json=payload,
            stream=True,
            timeout=self.timeout(config),
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            line_text = line.decode("utf-8", errors="ignore").strip()
            if not line_text.startswith("data: "):
                continue
            data = line_text[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if isinstance(delta.get("content"), str):
                full_response += delta["content"]
            elif isinstance(delta.get("content"), list):
                for item in delta["content"]:
                    if isinstance(item, dict) and item.get("text"):
                        full_response += item["text"]
        return full_response

    async def stream_chat_text(self, config, payload):
        return await asyncio.to_thread(self._stream_chat_text_sync, config, payload)

    async def process(
        self,
        text,
        model="nano-banana",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        temperature=1.0,
        top_p=0.95,
        api_key="",
        seed=0,
        max_tokens=32768,
    ):
        config = self.load_chat_config(api_key_override=api_key)
        fallback_image = self.default_image(image1, image2, image3, image4, width=512, height=512)
        if not (config.get("api_key") or "").strip():
            return fallback_image, "API key is required for lingliu_nano_banana.", ""

        content = [{"type": "text", "text": text}]
        for image in [image1, image2, image3, image4]:
            if image is None:
                continue
            batch_size = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for batch_index in range(batch_size):
                slice_image = image[batch_index : batch_index + 1] if batch_size > 1 else image
                encoded = base64.b64encode(image_to_png_bytes(slice_image)).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": True,
        }
        if int(seed) > 0:
            payload["seed"] = int(seed)

        try:
            response_text = await self.stream_chat_text(config, payload)
        except Exception as exc:
            return fallback_image, f"API Error: {mask_provider_text(exc)}", ""

        candidates = extract_image_candidates_from_text(response_text)
        for candidate in candidates:
            try:
                image = load_image_from_candidate(candidate, timeout=self.timeout(config))
                return pil2tensor(image), response_text, candidate
            except Exception:
                continue

        return fallback_image, response_text, ""


class LingliuNanoBananaEdit(LingliuGoogleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
                "model_provider": (PUBLIC_PROVIDER_DISPLAY_CHOICES, {"default": current_model_provider_default()}),
                "model": (BANANA_EDIT_MODEL_CHOICES, {"default": "nano-banana"}),
                "aspect_ratio": (
                    ["16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9"],
                    {"default": "1:1"},
                ),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "webhook": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response")
    FUNCTION = "generate_image"
    CATEGORY = "Lingliu/Google"

    async def generate_image(
        self,
        prompt,
        mode="text2img",
        model_provider="lingliu",
        model="nano-banana",
        aspect_ratio="1:1",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        api_key="",
        response_format="url",
        seed=0,
        webhook="",
        unique_id=None,
    ):
        task_id = ""
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                return blank_image_tensor(), "API key is required for lingliu_nano_banana_edit."
            model_provider = normalize_provider(model_provider)
            actual_provider = self.provider_name(config)
            if model_provider != actual_provider:
                return (
                    blank_image_tensor(),
                    f"模型线路“{provider_display_name(model_provider)}”与当前全局线路“{provider_display_name(actual_provider)}”不一致。",
                )

            query_params = {"async": "true"}
            if webhook.strip():
                query_params["webhook"] = webhook.strip()



            if mode == "text2img":
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio,
                    "response_format": response_format,
                }
                if int(seed) > 0:
                    payload["seed"] = int(seed)
                submit_payload = await self.async_request_json(
                    "POST",
                    self.endpoint(config, "images_generate_path"),
                    config,
                    headers=self.auth_headers(config, json_mode=True),
                    json=payload,
                    params=query_params,
                )
            else:
                files = []
                for image in [image1, image2, image3, image4]:
                    if image is None:
                        continue
                    files.append(("image", ("image.png", image_to_png_bytes(image), "image/png")))
                submit_payload = await self.async_request_json(
                    "POST",
                    self.endpoint(config, "images_edit_path"),
                    config,
                    headers=self.auth_headers(config),
                    data={
                        "prompt": prompt,
                        "model": model,
                        "aspect_ratio": aspect_ratio,
                        "response_format": response_format,
                        **({"seed": str(seed)} if int(seed) > 0 else {}),
                    },
                    files=files,
                    params=query_params,
                )

            task_id = submit_payload.get("task_id")
            if not task_id:
                return blank_image_tensor(), f"No task_id returned from image task endpoint: {submit_payload}"
            self.publish_task_id(unique_id, task_id)

            image_items, _ = await self.wait_for_image_task(config, task_id, timeout=self.timeout(config))
            tensor, _ = await self.decode_image_items(image_items, timeout=self.timeout(config))
            response_info = (
                f"Generated {tensor.shape[0]} image(s) using {model}\n"
                f"Aspect ratio: {aspect_ratio}\n"
                f"Task ID: {task_id}\n"
                + (f"Seed: {seed}\n" if int(seed) > 0 else "")
            )
            return tensor, response_info
        except Exception as exc:
            response_info = f"Error: {mask_provider_text(exc)}"
            if task_id:
                response_info += f"\nTask ID: {task_id}"
            return blank_image_tensor(), response_info


class LingliuNanoBanana2Edit(LingliuGoogleBase):
    GEMINI_ONLY_RATIOS = {"1:4", "4:1", "1:8", "8:1"}
    GEMINI_IMAGE_SIZE_MODELS = {"gemini-3.1-flash-image-preview", "gemini-3.1-flash-image"}
    # -2k/-4k pick their tier from the model name and never send image_size.
    GEMINI_TIER_MODELS = {
        "gemini-3.1-flash-image-preview-2k",
        "gemini-3.1-flash-image-preview-4k",
        "gemini-3.1-flash-image-2k",
        "gemini-3.1-flash-image-4k",
    }
    IMAGE_SIZE_MODELS = {"nano-banana-2", "nano-banana-2.1", "nano-banana-pro"} | GEMINI_IMAGE_SIZE_MODELS
    # JL picks the tier from image_size; -2k/-4k are kept only for older prompts.
    JL_IMAGE_SIZE_MODELS = {"nano-banana-2", "nano-banana-2.1", "nano-banana-pro"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "mode": (["text2img", "img2img"], {"default": "text2img"}),
                "model_provider": (PUBLIC_PROVIDER_DISPLAY_CHOICES, {"default": current_model_provider_default()}),
                "model": (BANANA2_EDIT_MODEL_CHOICES, {"default": "nano-banana-2"}),
                "aspect_ratio": (
                    ["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9", "1:4", "4:1", "1:8", "8:1"],
                    {"default": "auto"},
                ),
                "image_size": (["1K", "2K", "4K"], {"default": "2K"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "response_format": (["url", "b64_json"], {"default": "url"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "webhook": ("STRING", {"default": ""}),
                # V16.3: keep the channel last so legacy widgets_values stay aligned.
                "渠道": (CHANNELS, {"default": CHANNEL_DEFAULT_NEW_NODE}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "response", "image_url")
    FUNCTION = "generate_image"
    CATEGORY = "Lingliu/Google"

    async def generate_image(
        self,
        prompt,
        mode="text2img",
        model_provider="lingliu",
        model="nano-banana-2",
        aspect_ratio="auto",
        image_size="2K",
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        image11=None,
        image12=None,
        image13=None,
        image14=None,
        api_key="",
        response_format="url",
        seed=0,
        webhook="",
        渠道=CHANNEL_DEFAULT_LEGACY,
        unique_id=None,
    ):
        task_id = ""
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                return blank_image_tensor(), "API key is required for lingliu_nano_banana2_edit.", ""
            model_provider = normalize_provider(model_provider)
            actual_provider = self.provider_name(config)
            if model_provider != actual_provider:
                return (
                    blank_image_tensor(),
                    f"模型线路“{provider_display_name(model_provider)}”与当前全局线路“{provider_display_name(actual_provider)}”不一致。",
                    "",
                )

            channel, channel_id = await resolve_route(config, model, 渠道)
            if channel == CHANNEL_JL:
                return await self._generate_jl(
                    config, channel_id, prompt, mode, model, aspect_ratio, image_size,
                    [image1, image2, image3, image4, image5, image6, image7,
                     image8, image9, image10, image11, image12, image13, image14],
                    response_format, seed, unique_id,
                )
            # LL: pin the original channel for submits only; polling keeps the raw key.
            submit_config = config_with_channel(config, channel_id)

            gemini_models = self.GEMINI_IMAGE_SIZE_MODELS | self.GEMINI_TIER_MODELS
            if aspect_ratio in self.GEMINI_ONLY_RATIOS and model not in gemini_models:
                message = (
                    f"Aspect ratio '{aspect_ratio}' is only supported by gemini-3.1 image models. "
                    f"Current model: '{model}'."
                )
                return blank_image_tensor(), message, ""

            query_params = {"async": "true"}
            if webhook.strip():
                query_params["webhook"] = webhook.strip()

            all_images = [
                image1,
                image2,
                image3,
                image4,
                image5,
                image6,
                image7,
                image8,
                image9,
                image10,
                image11,
                image12,
                image13,
                image14,
            ]



            if mode == "text2img":
                payload = {
                    "prompt": prompt,
                    "model": model,
                    "aspect_ratio": aspect_ratio,
                    "response_format": response_format,
                }
                if model in self.IMAGE_SIZE_MODELS:
                    payload["image_size"] = image_size
                if int(seed) > 0:
                    payload["seed"] = int(seed)
                submit_payload = await self.async_request_json(
                    "POST",
                    self.endpoint(config, "images_generate_path"),
                    submit_config,
                    headers=self.auth_headers(submit_config, json_mode=True),
                    json=payload,
                    params=query_params,
                )
                input_image_count = 0
            else:
                references = [image for image in all_images if image is not None]
                encode_started = time.perf_counter()
                blobs = await asyncio.to_thread(lambda: [image_to_png_bytes(image) for image in references])
                input_image_count = len(blobs)
                if blobs:
                    log_reference(
                        f"参考图 {len(blobs)} 张编码 {time.perf_counter() - encode_started:.1f}s，"
                        f"共 {sum(map(len, blobs)) / 1048576:.1f}MB"
                    )
                reference_urls = None
                if blobs and self.provider_name(config) in (LINGLIU_PROVIDER):
                    reference_urls = await upload_references(config["base_url"], config.get("api_key", ""), blobs)
                submit_started = time.perf_counter()
                if reference_urls:
                    # Same contract the canvas uses for Banana URL references; "auto" is omitted
                    # so the upstream follows the reference image.
                    payload = {"prompt": prompt, "model": model, "image": reference_urls, "response_format": response_format}
                    if aspect_ratio != "auto":
                        payload["aspect_ratio"] = aspect_ratio
                    if model in self.IMAGE_SIZE_MODELS:
                        payload["image_size"] = image_size
                    if int(seed) > 0:
                        payload["seed"] = int(seed)
                    submit_payload = await self.async_request_json(
                        "POST",
                        self.endpoint(config, "images_generate_path"),
                        submit_config,
                        headers=self.auth_headers(submit_config, json_mode=True),
                        json=payload,
                        params=query_params,
                    )
                    log_reference(f"提交到拿到任务号 {time.perf_counter() - submit_started:.1f}s（参考图直传）")
                    reference_mode = "direct"
                else:
                    reference_mode = "multipart"
            if mode != "text2img" and reference_mode == "multipart":
                files = [("image", (f"image_{index}.png", data, "image/png")) for index, data in enumerate(blobs)]
                submit_payload = await self.async_request_json(
                    "POST",
                    self.endpoint(config, "images_edit_path"),
                    submit_config,
                    headers=self.auth_headers(submit_config),
                    data={
                        "prompt": prompt,
                        "model": model,
                        "aspect_ratio": aspect_ratio,
                        "response_format": response_format,
                        **({"image_size": image_size} if model in self.IMAGE_SIZE_MODELS else {}),
                        **({"seed": str(seed)} if int(seed) > 0 else {}),
                    },
                    files=files,
                    params=query_params,
                )
                if blobs:
                    log_reference(f"提交到拿到任务号 {time.perf_counter() - submit_started:.1f}s（参考图随请求上传）")

            task_id = submit_payload.get("task_id")
            if not task_id:
                return blank_image_tensor(), f"No task_id returned from image task endpoint: {submit_payload}", ""
            self.publish_task_id(unique_id, task_id)

            image_items, _ = await self.wait_for_image_task(config, task_id, timeout=self.timeout(config))
            tensor, first_image_url = await self.decode_image_items(image_items, timeout=self.timeout(config))
            response_info = f"Generated {tensor.shape[0]} image(s) using {model}\n"
            if model in self.IMAGE_SIZE_MODELS:
                response_info += f"Image size: {image_size}\n"
            response_info += f"Aspect ratio: {aspect_ratio}\n"
            response_info += f"Task ID: {task_id}\n"
            if mode == "img2img":
                response_info += f"Input images: {input_image_count}\n"
                if input_image_count:
                    response_info += f"Reference upload: {reference_mode}\n"
            if int(seed) > 0:
                response_info += f"Seed: {seed}\n"
            if channel_id:
                response_info += "Channel: LL\n"
            return tensor, response_info, first_image_url
        except ImageSubmitUncertain as exc:
            self.publish_task_id(unique_id, exc.task_id)
            return blank_image_tensor(), f"Error: {exc}\nTask ID: {exc.task_id}", ""
        except Exception as exc:
            response_info = f"Error: {mask_provider_text(exc)}"
            if task_id:
                response_info += f"\nTask ID: {task_id}"
            return blank_image_tensor(), response_info, ""

    async def _generate_jl(self, config, channel_id, prompt, mode, model, aspect_ratio, image_size,
                           all_images, response_format, seed, unique_id):
        """JL: only platform-accepted fields, one submit, flat polling."""
        jl_contract(model)
        uses_image_size = model in self.JL_IMAGE_SIZE_MODELS
        if uses_image_size:
            resolution = str(image_size or "").strip().upper()
            validate_jl_resolution(model, resolution)
        else:
            resolution = jl_contract(model)["resolutions"][0]
        validate_jl_ratio(model, aspect_ratio)
        references = [image for image in all_images if image is not None] if mode != "text2img" else []
        if mode != "text2img" and not references:
            raise ImageChannelError("图生图模式至少需要连接一张参考图。")
        reference_bytes = await asyncio.to_thread(lambda: [jl_reference_bytes(image) for image in references])
        validate_jl_references(model, reference_bytes)
        ratio = str(aspect_ratio or "auto").strip()
        if ratio == "auto" and resolution != "1K":
            # The platform rejects an auto ratio above 1K; derive it locally.
            width, height = image_tensor_size(references[0]) if references else (1, 1)
            ratio = nearest_ratio(width, height, jl_contract(model)["ratios"])
        fields = {"prompt": prompt, "model": model, "response_format": response_format}
        if ratio != "auto":
            fields["aspect_ratio"] = ratio
        if uses_image_size:
            fields["image_size"] = resolution
        submit_config = config_with_channel(config, channel_id)
        if references:
            files = [
                ("image", (f"image_{index}.jpg", data, "image/jpeg"))
                for index, data in enumerate(reference_bytes)
            ]
            submit_payload = await submit_once(
                self, self.append_async(self.endpoint(config, "images_edit_path")), submit_config,
                timeout=self.timeout(config), headers=self.auth_headers(submit_config), data=fields, files=files,
            )
        else:
            submit_payload = await submit_once(
                self, self.append_async(self.endpoint(config, "images_generate_path")), submit_config,
                timeout=self.timeout(config), headers=self.auth_headers(submit_config, json_mode=True), json=fields,
            )
        task_id = self.extract_task_id(submit_payload)
        if not task_id:
            raise RuntimeError(f"No task_id returned from image task endpoint: {self.public_payload(submit_payload)}")
        self.publish_task_id(unique_id, task_id)
        try:
            image_items, _ = await wait_flat_image_task(
                self, config, task_id,
                max_poll_attempts=self.IMAGE_TASK_MAX_RETRIES,
                poll_interval=self.POLL_INTERVAL_SECONDS,
            )
            tensor, first_image_url = await self.decode_image_items(image_items, timeout=self.timeout(config))
        except Exception as exc:
            return blank_image_tensor(), f"Error: {mask_provider_text(exc)}\nTask ID: {task_id}", ""
        response_info = f"Generated {tensor.shape[0]} image(s) using {model}\n"
        response_info += "Channel: JL\n"
        if uses_image_size:
            response_info += f"Image size: {resolution}\n"
        response_info += f"Aspect ratio: {ratio}\n"
        response_info += f"Task ID: {task_id}\n"
        if references:
            response_info += f"Input images: {len(references)}\n"
        if int(seed) > 0:
            response_info += "Seed: JL 不支持 seed，未发送\n"
        return tensor, response_info, first_image_url

    @staticmethod
    def append_async(url):
        return f"{url}{'&' if '?' in url else '?'}async=true"


class LingliuGeminiTextOnly(LingliuGoogleBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gemini-2.5-pro"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "stream": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "Lingliu/Google"

    def _stream_text_response_sync(self, config, payload):
        full_response = ""
        response = requests.post(
            self.endpoint(config, "chat_path"),
            headers=self.auth_headers(config, json_mode=True),
            json=payload,
            stream=True,
            timeout=self.timeout(config),
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            line_text = line.decode("utf-8", errors="ignore").strip()
            if not line_text.startswith("data: "):
                continue
            data = line_text[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if isinstance(delta.get("content"), str):
                full_response += delta["content"]
        return full_response

    async def _stream_text_response(self, config, payload):
        return await asyncio.to_thread(self._stream_text_response_sync, config, payload)

    async def generate_text(
        self,
        prompt,
        model,
        temperature,
        top_p,
        max_tokens,
        seed,
        image=None,
        video=None,
        api_key="",
        system_prompt="",
        stream=False,
    ):
        config = self.load_chat_config(api_key_override=api_key)
        if not (config.get("api_key") or "").strip():
            return ("API key is required for lingliu_gemini_text_only.",)

        content = [{"type": "text", "text": prompt}]

        if image is not None:
            images = tensor2pil(image)
            for pil_image in images:
                encoded = base64.b64encode(image_to_png_bytes(pil_image)).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

        if video is not None:
            remote_url = video_to_remote_url(video)
            if remote_url:
                # gpt-best's Gemini video analysis docs use image_url with an mp4 URL.
                content.append({"type": "image_url", "image_url": {"url": remote_url}})
            else:
                data_url = video_to_data_url(video)
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
        }
        if int(seed) > 0:
            payload["seed"] = int(seed)

        try:
            if stream:
                text = await self._stream_text_response(config, payload)
            else:
                response_payload = await self.async_request_json(
                    "POST",
                    self.endpoint(config, "chat_path"),
                    config,
                    headers=self.auth_headers(config, json_mode=True),
                    json=payload,
                )
                text = response_payload["choices"][0]["message"]["content"]
            return (text,)
        except Exception as exc:
            return (f"Error: {mask_provider_text(exc)}",)
