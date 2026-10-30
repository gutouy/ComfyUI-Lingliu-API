import asyncio
import base64
import json
import math
import os
from io import BytesIO
from urllib.parse import quote

import requests
import torch
from PIL import Image
from comfy.utils import common_upscale

from ..lingliu_config import (
    PUBLIC_PROVIDER_DISPLAY_CHOICES,
    build_url,
    get_config,
    normalize_provider,
    provider_display_name,
)
from ..utils import (
    blank_image_tensor,
    content_to_text,
    extract_image_candidates_from_text,
    image_to_data_url,
    image_to_png_bytes,
    load_image_from_candidate,
    parse_image_response,
    pil2tensor,
    tensor2pil,
)
from .nodes_openai_compatible import LingliuBaseNode, LingliuImageEdit, LingliuImageGenerate
from .lingliu_image_channels import (
    CHANNELS,
    CHANNEL_DEFAULT_LEGACY,
    CHANNEL_DEFAULT_NEW_NODE,
    CHANNEL_JL,
    JL_QUALITIES,
    ImageChannelError,
    ImageSubmitUncertain,
    channel_applies,
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


class LingliuTerminalTaskFailure(RuntimeError):
    """Raised when an upstream async task has reached a final failed state."""


def current_public_provider_default():
    # New provider-selecting nodes start on the production Lingliu route.
    # Existing workflow values remain authoritative when they are loaded.
    return provider_display_name("lingliu")


def downscale_input(image, total_pixels=2048 * 2048):
    samples = image.movedim(-1, 1)
    scale_by = math.sqrt(total_pixels / (samples.shape[3] * samples.shape[2]))
    if scale_by >= 1:
        return image
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    scaled = common_upscale(samples, width, height, "lanczos", "disabled")
    return scaled.movedim(1, -1)


class LingliuOpenAIBase(LingliuBaseNode):
    CATEGORY = "Lingliu/OpenAI"
    IMAGE_TASK_MAX_RETRIES = 360
    POLL_INTERVAL_SECONDS = 2
    TRANSIENT_ERROR_LIMIT = 12

    def default_image(self, *images, width=1024, height=1024):
        for image in images:
            if image is not None and hasattr(image, "shape") and len(image.shape) == 4 and image.shape[0] > 0:
                return image[:1]
        return blank_image_tensor(width=width, height=height)

    def auth_headers(self, config, json_mode=False):
        headers = self.headers(config)
        if json_mode:
            headers["Content-Type"] = "application/json"
        return headers

    def task_endpoint(self, config, task_id):
        template = config.get("images_task_status_path_template", "")
        return build_url(config.get("base_url"), template.format(task_id=task_id))

    def append_query_param(self, url, key, value):
        if not value:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{key}={quote(str(value), safe='')}"

    def request_json_with_timeout(self, method, url, timeout, **kwargs):
        response = requests.request(method, url, timeout=timeout, **kwargs)
        self._raise_for_status_with_details(response, url)
        return self._decode_json_response(response, url)

    async def async_request_json_with_timeout(self, method, url, timeout, **kwargs):
        return await asyncio.to_thread(self.request_json_with_timeout, method, url, timeout, **kwargs)

    async def async_request_json_retrying(
        self,
        method,
        url,
        config,
        *,
        max_retries=5,
        initial_timeout=300,
        retry_status_codes=(429, 500, 502, 503, 504),
        **kwargs,
    ):
        last_error = None
        for attempt in range(1, max_retries + 1):
            current_timeout = min(int(initial_timeout * (1.5 ** (attempt - 1))), 1200)
            try:
                return await self.async_request_json_with_timeout(method, url, current_timeout, **kwargs)
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                last_error = exc
                if status_code not in retry_status_codes or attempt == max_retries:
                    raise
            except Exception as exc:
                last_error = exc
                if not self.is_transient_request_error(exc) or attempt == max_retries:
                    raise
            await asyncio.sleep(min(2 ** (attempt - 1), 60))
        raise last_error or RuntimeError(f"Failed to request {url}")

    async def wait_for_image_task(self, config, task_id, max_poll_attempts=None, poll_interval=2):

        if max_poll_attempts is None:
            max_poll_attempts = self.IMAGE_TASK_MAX_RETRIES
        last_payload = None
        consecutive_poll_errors = 0
        for _ in range(int(max_poll_attempts)):
            await asyncio.sleep(max(1, int(poll_interval)))
            try:
                payload = await self.async_request_json(
                    "GET",
                    self.task_endpoint(config, task_id),
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
                    if consecutive_poll_errors < self.TRANSIENT_ERROR_LIMIT:
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
                    raise LingliuTerminalTaskFailure(f"Task {task_id} failed: {flat_error}")
                continue
            outer_data = payload.get("data", {}) if isinstance(payload, dict) else {}
            status = ""
            if isinstance(outer_data, dict):
                status = str(outer_data.get("status", "") or outer_data.get("state", "") or "")
            status = status or self.extract_status(payload)

            if self.is_success_status(status):
                inner_data = outer_data.get("data", {}) if isinstance(outer_data, dict) else {}
                if isinstance(inner_data, dict):
                    return inner_data.get("data", []), payload
                if isinstance(inner_data, list):
                    return inner_data, payload
                image_urls, _video_urls = self.extract_media_urls(payload)
                if image_urls:
                    return [{"url": url} for url in image_urls], payload
                raise RuntimeError(f"Task {task_id} succeeded but no images were returned. Payload: {payload}")

            if self.is_failure_status(status):
                failure_context = outer_data if isinstance(outer_data, dict) and outer_data else payload
                fail_reason = self.extract_failure_detail(failure_context) or "Unknown error"
                raise LingliuTerminalTaskFailure(f"Task {task_id} failed: {fail_reason}")

        raise TimeoutError(
            f"Task {task_id} timed out after {max_poll_attempts} polls. "
            f"Last payload: {last_payload}"
        )

    async def decode_image_items(self, image_items, timeout=300):
        if not image_items:
            raise RuntimeError("No images were returned from the task result.")

        payload = {"data": image_items}
        images, urls = await asyncio.to_thread(parse_image_response, payload, timeout)
        if not images:
            raise RuntimeError("Failed to decode any generated images from task result.")
        first_url = urls[0] if urls else ""
        if not first_url and isinstance(image_items[0], dict) and image_items[0].get("b64_json"):
            first_url = f"data:image/png;base64,{image_items[0]['b64_json']}"
        return pil2tensor(images), first_url

    def image_tensor_to_data_uri(self, image_tensor):
        return image_to_data_url(image_tensor)

    def image_tensor_to_data_uris(self, image_tensor):
        if image_tensor is None:
            return []
        return [image_to_data_url(pil_image) for pil_image in tensor2pil(image_tensor)]

    def normalize_file_paths(self, files):
        if not files:
            return []
        if isinstance(files, str):
            stripped = files.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except Exception:
                pass
            return [line.strip() for line in stripped.splitlines() if line.strip()]
        if isinstance(files, (list, tuple, set)):
            return [str(item) for item in files if str(item).strip()]
        return [str(files)]

    def file_to_data_url(self, file_path):
        with open(file_path, "rb") as handle:
            raw = handle.read()
        mime_type = "application/octet-stream"
        try:
            import mimetypes

            mime_type = mimetypes.guess_type(file_path)[0] or mime_type
        except Exception:
            pass
        encoded = json.dumps(raw.decode("utf-8", errors="ignore"))
        del encoded
        import base64

        return f"data:{mime_type};base64,{base64.b64encode(raw).decode('utf-8')}", mime_type

    def extract_response_image_candidates(self, message_content):
        candidates = []
        text_value = content_to_text(message_content)
        for candidate in extract_image_candidates_from_text(text_value):
            if candidate not in candidates:
                candidates.append(candidate)

        if isinstance(message_content, list):
            for item in message_content:
                if not isinstance(item, dict):
                    continue
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    candidate = image_url.get("url", "")
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
                elif isinstance(image_url, str) and image_url not in candidates:
                    candidates.append(image_url)
        return candidates

    def mask_to_png_bytes(self, mask, image):
        if mask is None:
            return None
        if image is None:
            raise ValueError("Cannot use a mask without an input image.")
        if image.shape[0] != 1:
            raise ValueError("Mask requires a single input image.")
        if mask.shape[1:] != image.shape[1:-1]:
            raise ValueError("Mask and Image must be the same size.")

        _batch, height, width = mask.shape
        rgba_mask = torch.zeros(height, width, 4, device="cpu")
        rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
        scaled_mask = downscale_input(rgba_mask.unsqueeze(0)).squeeze()
        return image_to_png_bytes(scaled_mask.unsqueeze(0))

    @staticmethod
    def build_jl_reference_files(image):
        if image is None:
            return []
        return [
            ("image", (f"image_{index}.jpg", jl_reference_bytes(downscale_input(image[index : index + 1])), "image/jpeg"))
            for index in range(image.shape[0])
        ]

    def build_edit_file_payload(self, image, mask=None, include_blank=True):
        files = []
        batch_size = 0

        if image is None:
            if not include_blank:
                return files
            files.append(("image", ("blank.png", image_to_png_bytes(blank_image_tensor()), "image/png")))
            batch_size = 1
        else:
            batch_size = image.shape[0]
            for index in range(batch_size):
                single_image = image[index : index + 1]
                scaled = downscale_input(single_image)
                files.append(("image", (f"image_{index}.png", image_to_png_bytes(scaled), "image/png")))

        mask_bytes = self.mask_to_png_bytes(mask, image) if mask is not None else None
        if mask_bytes is not None:
            files.append(("mask", ("mask.png", mask_bytes, "image/png")))

        return files


class LingliuGptImage(LingliuImageGenerate):
    CATEGORY = "Lingliu/OpenAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-1", "gpt-image-1.5"], {"default": "gpt-image-1"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }


class LingliuGptImageEdit(LingliuImageEdit):
    CATEGORY = "Lingliu/OpenAI"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "model": (["gpt-image-1", "gpt-image-1.5"], {"default": "gpt-image-1"}),
            },
            "optional": {
                "mask": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "background": (["auto", "transparent", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }


class LingliuChatGPTApi(LingliuOpenAIBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-4o-image"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "files": ("FILES",),
                "image_url": ("STRING", {"default": ""}),
                "images": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 16384}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image_download_timeout": ("INT", {"default": 600, "min": 30, "max": 1800}),
                "clear_chats": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "response", "image_urls", "raw_json")
    FUNCTION = "process"
    CATEGORY = "Lingliu/OpenAI"

    async def process(
        self,
        prompt,
        model,
        api_key="",
        files=None,
        image_url="",
        images=None,
        temperature=0.7,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=-1,
        image_download_timeout=600,
        clear_chats=True,
    ):
        del clear_chats
        try:
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_chatgpt_api."
                return self.default_image(images, width=512, height=512), error, "", self.error_json(error)

            content = [{"type": "text", "text": prompt}]

            if images is not None:
                batch_size = images.shape[0]
                max_images = min(batch_size, 4)
                for index in range(max_images):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_tensor_to_data_uri(images[index : index + 1])},
                        }
                    )
                if batch_size > max_images:
                    content.append(
                        {
                            "type": "text",
                            "text": f"{batch_size - max_images} additional images were omitted due to API limits.",
                        }
                    )
            elif image_url.strip():
                content.append({"type": "image_url", "image_url": {"url": image_url.strip()}})

            for file_path in self.normalize_file_paths(files):
                if not os.path.exists(file_path):
                    continue
                data_url, mime_type = await asyncio.to_thread(self.file_to_data_url, file_path)
                if mime_type.startswith("image/"):
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                else:
                    content.append({"type": "text", "text": f"Attached file: {os.path.basename(file_path)}"})
                    content.append(
                        {
                            "type": "file_url",
                            "file_url": {"url": data_url, "name": os.path.basename(file_path)},
                        }
                    )

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "top_p": float(top_p),
                "frequency_penalty": float(frequency_penalty),
                "presence_penalty": float(presence_penalty),
            }
            if int(seed) >= 0:
                payload["seed"] = int(seed)

            response_payload = await self.async_request_json(
                "POST",
                self.endpoint(config, "chat_path"),
                config,
                headers=self.auth_headers(config, json_mode=True),
                json=payload,
            )

            choices = response_payload.get("choices", []) if isinstance(response_payload, dict) else []
            if not choices:
                raise RuntimeError("No choices returned from ChatGPT API.")

            message = choices[0].get("message", {})
            message_content = message.get("content", "")
            response_text = content_to_text(message_content)
            if not response_text and choices[0].get("text"):
                response_text = str(choices[0]["text"])

            image_candidates = self.extract_response_image_candidates(message_content or response_text)
            image_urls = []
            image_tensors = []
            if image_candidates:
                tasks = [
                    asyncio.to_thread(load_image_from_candidate, candidate, image_download_timeout)
                    for candidate in image_candidates
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for candidate, result in zip(image_candidates, results):
                    if isinstance(result, Exception):
                        continue
                    image_urls.append(candidate)
                    image_tensors.append(pil2tensor(result))

            if image_tensors:
                combined = torch.cat(image_tensors, dim=0)
            else:
                combined = self.default_image(images, width=512, height=512)

            return (
                combined,
                response_text,
                "\n".join(image_urls),
                self.safe_json(response_payload),
            )
        except Exception as exc:
            return self.default_image(images, width=512, height=512), f"Error: {self.public_payload(str(exc))}", "", self.error_json(exc)


class LingliuGptImage2(LingliuOpenAIBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-image-2"], {"default": "gpt-image-2"}),
                "aspect_ratio": (
                    ["auto", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"],
                    {"default": "1:1"},
                ),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "webhook": ("STRING", {"default": ""}),
                "max_poll_attempts": ("INT", {"default": 300, "min": 10, "max": 1000}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 60}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "Lingliu/OpenAI"

    async def generate(
        self,
        prompt,
        model,
        aspect_ratio,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        api_key="",
        webhook="",
        max_poll_attempts=300,
        poll_interval=5,
        seed=0,
        unique_id=None,
    ):
        task_id = ""
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_gpt_image_2."
                return blank_image_tensor(), "", self.error_json(error)

            ratio = str(aspect_ratio).split(" ")[0].strip()
            submit_url = self.endpoint(config, "images_generate_path")

            payload_prompt = prompt if ratio == "auto" else f"{prompt}\n\nAspect ratio: {ratio}"
            payload = {
                "model": model,
                "prompt": payload_prompt,
                "aspect_ratio": ratio,
            }
            if int(seed) > 0:
                payload["seed"] = int(seed)

            images_list = []
            for image in [image1, image2, image3, image4]:
                if image is None:
                    continue
                batch_size = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
                for batch_index in range(batch_size):
                    slice_image = image[batch_index : batch_index + 1] if batch_size > 1 else image
                    images_list.append(self.image_tensor_to_data_uri(slice_image))
            if images_list:
                payload["image"] = images_list
            submit_url = self.append_query_param(submit_url, "async", "true")
            submit_url = self.append_query_param(submit_url, "webhook", webhook.strip())

            submit_payload = await self.async_request_json_retrying(
                "POST",
                submit_url,
                config,
                max_retries=5,
                initial_timeout=self.timeout(config),
                headers=self.auth_headers(config, json_mode=True),
                json=payload,
            )
            task_id = self.extract_task_id(submit_payload)
            if not task_id:
                raise RuntimeError(f"No task_id in response: {submit_payload}")
            self.publish_task_id(unique_id, task_id)

            image_items, final_payload = await self.wait_for_image_task(
                config,
                task_id,
                max_poll_attempts=max_poll_attempts,
                poll_interval=poll_interval,
            )
            images_tensor, first_url = await self.decode_image_items(image_items, timeout=self.timeout(config))
            response = {
                "task_id": task_id,
                "submit": submit_payload,
                "final": final_payload,
            }
            return images_tensor, first_url, self.safe_json(response)
        except Exception as exc:
            return blank_image_tensor(), "", self.error_json(exc, task_id=task_id)


class LingliuGptImage2Official(LingliuOpenAIBase):
    GPT_IMAGE2_SIZE_MAP = {
        "1:1": {
            "1K": "1024x1024",
            "2K": "2048x2048",
            "4K": "2880x2880",
        },
        "16:9": {
            "1K": "1280x720",
            "2K": "2560x1440",
            "4K": "3840x2160",
        },
        "9:16": {
            "1K": "720x1280",
            "2K": "1440x2560",
            "4K": "2160x3840",
        },
        "4:3": {
            "1K": "1152x864",
            "2K": "2304x1728",
            "4K": "3264x2448",
        },
        "3:4": {
            "1K": "864x1152",
            "2K": "1728x2304",
            "4K": "2448x3264",
        },
        "3:2": {
            "1K": "1248x832",
            "2K": "2496x1664",
            "4K": "3504x2336",
        },
        "2:3": {
            "1K": "832x1248",
            "2K": "1664x2496",
            "4K": "2336x3504",
        },
        "5:4": {
            "1K": "1120x896",
            "2K": "2240x1792",
            "4K": "3200x2560",
        },
        "4:5": {
            "1K": "896x1120",
            "2K": "1792x2240",
            "4K": "2560x3200",
        },
        "21:9": {
            "1K": "1456x624",
            "2K": "3024x1296",
            "4K": "3696x1584",
        },
    }
    GPT_IMAGE2_ASPECT_RATIOS = ["auto"] + list(GPT_IMAGE2_SIZE_MAP.keys())
    GPT_IMAGE2_RESOLUTIONS = ["1K", "2K", "4K"]
    GPT_IMAGE2_MODEL_GROUPS = {
        "lingliu": ["gpt-image-2", "gpt-image-2-all", "gpt-image-2-1k", "gpt-image-2-hq2", "gpt-image-2-hq3", "gpt-image-2-official", "gpt-image-2-enterprise"],
    }
    GPT_IMAGE2_MODELS = [
        "gpt-image-2",
        "gpt-image-2-all",
        "gpt-image-2-1k",
        "gpt-image-2-hq2",
        "gpt-image-2-hq3",
        "gpt-image-2-official",
        "gpt-image-2-enterprise",
    ]
    # V17.4: aliases that carry their own platform channel; 渠道 is ignored
    # and the key stays unsuffixed.  Same contract as the GPT Image 2.5 node.
    GPT_IMAGE2_FIXED_CHANNEL_MODELS = {
        "gpt-image-2-hq2": "高质渠道2",
        "gpt-image-2-hq3": "优惠渠道3",
        "gpt-image-2-official": "官方",
        "gpt-image-2-enterprise": "官方-企业",
    }
    GPT_IMAGE2_MODEL_TOOLTIP = (
        "-hq2=高质渠道2，-hq3=优惠渠道3：自带渠道（忽略「渠道」选项），按 n 逐张提交，"
        "不支持 mask 和 auto 比例，quality 只支持 auto/low/medium/high，background/output_format/moderation 需保持默认。"
        "-official=官方，-enterprise=官方-企业：同样自带渠道，quality 只支持 low/medium/high（无 auto），"
        "按尺寸×质量×参考图计价，2K/4K 搭配 high 较慢。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_provider": (PUBLIC_PROVIDER_DISPLAY_CHOICES, {"default": current_public_provider_default()}),
                "model": (cls.GPT_IMAGE2_MODELS, {"default": "gpt-image-2", "tooltip": cls.GPT_IMAGE2_MODEL_TOOLTIP}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "api_key": ("STRING", {"default": ""}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8}),
                "quality": (["auto", "high", "medium", "low"], {"default": "auto"}),
                "aspect_ratio": (cls.GPT_IMAGE2_ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (cls.GPT_IMAGE2_RESOLUTIONS, {"default": "2K"}),
                "background": (["auto", "opaque"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "async_mode": ("BOOLEAN", {"default": True}),
                "webhook": ("STRING", {"default": ""}),
                "max_poll_attempts": ("INT", {"default": 300, "min": 10, "max": 1000}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 60}),
                "max_retries": ("INT", {"default": 5, "min": 1, "max": 10}),
                "initial_timeout": ("INT", {"default": 900, "min": 60, "max": 1200}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                # V16.3: keep the channel last so legacy widgets_values stay aligned.
                "渠道": (CHANNELS, {"default": CHANNEL_DEFAULT_NEW_NODE}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "Lingliu/OpenAI"

    def resolve_size(self, aspect_ratio, resolution):
        if str(aspect_ratio).strip().lower() == "auto":
            return "auto"
        ratio_map = self.GPT_IMAGE2_SIZE_MAP.get(aspect_ratio) or self.GPT_IMAGE2_SIZE_MAP["1:1"]
        return ratio_map.get(resolution) or ratio_map["2K"]

    def validate_selected_provider_and_model(self, config, model_provider, model):
        selected_provider = normalize_provider(model_provider)
        configured_provider = self.provider_name(config)
        if selected_provider != configured_provider:
            raise RuntimeError(
                "线路不一致："
                f"当前节点选择的是 {provider_display_name(selected_provider)}，"
                f"但 lingliu_api_set 设置的是 {provider_display_name(configured_provider)}。"
                "请把节点线路和 set-api 节点线路保持一致后再运行。"
            )
        model = str(model or "").strip() or "gpt-image-2"
        allowed_models = self.GPT_IMAGE2_MODEL_GROUPS.get(selected_provider, [])
        if model not in allowed_models:
            raise RuntimeError(
                f"{provider_display_name(selected_provider)} 不支持模型 {model}。"
                f"{provider_display_name(selected_provider)} 可用模型：{', '.join(allowed_models)}。"
            )
        return selected_provider, model

    def build_payload(self, prompt, model, image, mask, n, quality, aspect_ratio, resolution, background, output_format, output_compression, moderation, seed):
        resolved_size = self.resolve_size(aspect_ratio, resolution)
        data = {
            "prompt": prompt,
            "model": model,
            "n": str(n),
            "quality": quality,
            "moderation": moderation,
            "size": resolved_size,
        }
        if background != "auto":
            data["background"] = background
        if output_format != "png":
            data["output_format"] = output_format
        if int(output_compression) != 100:
            data["output_compression"] = str(output_compression)
        if int(seed) > 0:
            data["seed"] = str(seed)
        files = self.build_edit_file_payload(image, mask=mask, include_blank=True)
        return data, files, resolved_size

    def _summarize_single_result(self, result):
        summary = {}
        if "task_id" in result:
            summary["task_id"] = result.get("task_id", "")
        if "request_id" in result:
            summary["request_id"] = result.get("request_id", "")
        if "fal_endpoint_id" in result:
            summary["fal_endpoint_id"] = result.get("fal_endpoint_id", "")
        if "submit" in result:
            summary["submit"] = result.get("submit")
        if "final" in result:
            summary["final"] = result.get("final")
        if "response" in result:
            summary["response"] = result.get("response")
        if "aspect_ratio" in result:
            summary["aspect_ratio"] = result.get("aspect_ratio")
        if "resolution" in result:
            summary["resolution"] = result.get("resolution")
        if "resolved_size" in result:
            summary["resolved_size"] = result.get("resolved_size")
        if "model" in result:
            summary["model"] = result.get("model")
        summary["urls"] = list(result.get("urls", []) or [])
        summary["image_count"] = len(result.get("images", []) or [])
        return summary

    async def _run_single_async_request(
        self,
        config,
        prompt,
        model,
        image,
        mask,
        quality,
        aspect_ratio,
        resolution,
        background,
        output_format,
        output_compression,
        moderation,
        webhook,
        max_poll_attempts,
        poll_interval,
        max_retries,
        initial_timeout,
        seed,
        unique_id=None,
        request_index=0,
        channel_id=None,
    ):
        resolved_size = self.resolve_size(aspect_ratio, resolution)


        data, request_files, resolved_size = self.build_payload(
            prompt,
            model,
            image,
            mask,
            1,
            quality,
            aspect_ratio,
            resolution,
            background,
            output_format,
            output_compression,
            moderation,
            seed,
        )
        submit_url = self.append_query_param(self.endpoint(config, "images_edit_path"), "async", "true")
        submit_url = self.append_query_param(submit_url, "webhook", webhook.strip())
        submit_config = config_with_channel(config, channel_id)
        submit_payload = await self.async_request_json_retrying(
            "POST",
            submit_url,
            submit_config,
            max_retries=max_retries,
            initial_timeout=initial_timeout,
            headers=self.headers(submit_config),
            data=data,
            files=request_files,
        )
        task_id = submit_payload.get("task_id") or submit_payload.get("data") or ""
        if not task_id:
            raise RuntimeError(f"No task_id in response: {submit_payload}")
        self.publish_task_id(unique_id, task_id, label=f"Task ID #{request_index + 1}")

        image_items, final_payload = await self.wait_for_image_task(
            config,
            task_id,
            max_poll_attempts=max_poll_attempts,
            poll_interval=poll_interval,
        )
        payload = {"data": image_items}
        images, urls = await asyncio.to_thread(parse_image_response, payload, initial_timeout)
        if not images:
            raise RuntimeError("Async task succeeded but no images were returned.")
        return {
            "task_id": task_id,
            "submit": submit_payload,
            "final": final_payload,
            "images": images,
            "urls": urls,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "resolved_size": resolved_size,
            "model": model,
        }

    async def _run_single_sync_request(
        self,
        config,
        prompt,
        model,
        image,
        mask,
        quality,
        aspect_ratio,
        resolution,
        background,
        output_format,
        output_compression,
        moderation,
        max_retries,
        initial_timeout,
        seed,
        channel_id=None,
    ):
        resolved_size = self.resolve_size(aspect_ratio, resolution)


        data, request_files, resolved_size = self.build_payload(
            prompt,
            model,
            image,
            mask,
            1,
            quality,
            aspect_ratio,
            resolution,
            background,
            output_format,
            output_compression,
            moderation,
            seed,
        )
        submit_config = config_with_channel(config, channel_id)
        response_payload = await self.async_request_json_retrying(
            "POST",
            self.endpoint(config, "images_edit_path"),
            submit_config,
            max_retries=max_retries,
            initial_timeout=initial_timeout,
            headers=self.headers(submit_config),
            data=data,
            files=request_files,
        )
        images, urls = await asyncio.to_thread(parse_image_response, response_payload, initial_timeout)
        if not images:
            raise RuntimeError("No images returned from gpt-image-2 official endpoint.")
        return {
            "response": response_payload,
            "images": images,
            "urls": urls,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "resolved_size": resolved_size,
            "model": model,
        }

    async def generate(
        self,
        prompt,
        model_provider=None,
        model="gpt-image-2",
        image=None,
        mask=None,
        api_key="",
        n=1,
        quality="auto",
        aspect_ratio="1:1",
        resolution="2K",
        background="auto",
        output_format="png",
        output_compression=100,
        moderation="auto",
        async_mode=True,
        webhook="",
        max_poll_attempts=300,
        poll_interval=5,
        max_retries=5,
        initial_timeout=900,
        seed=0,
        渠道=CHANNEL_DEFAULT_LEGACY,
        unique_id=None,
    ):
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_gpt_image_2_official."
                return blank_image_tensor(), "", self.error_json(error)
            selected_provider, model = self.validate_selected_provider_and_model(config, model_provider, model)

            fixed_label = self.GPT_IMAGE2_FIXED_CHANNEL_MODELS.get(model)
            if fixed_label:
                return await self._generate_fixed(
                    config, fixed_label, selected_provider, prompt, model, image, mask, n, quality,
                    aspect_ratio, resolution, background, output_format, moderation,
                    max_poll_attempts, poll_interval, unique_id,
                )
            channel, channel_id = await resolve_route(config, model, 渠道)
            if channel == CHANNEL_JL:
                return await self._generate_jl(
                    config, channel_id, selected_provider, prompt, model, image, mask, n, quality,
                    aspect_ratio, resolution, background, output_format, moderation,
                    max_poll_attempts, poll_interval, initial_timeout, unique_id,
                )
            # Lingliu lines submit each image once: a retried create can be
            # charged twice.  Other lines keep their historical retries.
            single_submit = channel_applies(selected_provider)
            if single_submit:
                max_retries = 1

            request_count = max(1, int(n))
            request_plan = []
            tasks = []
            for index in range(request_count):
                request_seed = int(seed) + index if int(seed) > 0 else 0
                request_plan.append({"index": index, "seed": request_seed})
                if async_mode:
                    tasks.append(
                        self._run_single_async_request(
                            config,
                            prompt,
                            model,
                            image,
                            mask,
                            quality,
                            aspect_ratio,
                            resolution,
                            background,
                            output_format,
                            output_compression,
                            moderation,
                            webhook,
                            max_poll_attempts,
                            poll_interval,
                            max_retries,
                            initial_timeout,
                            request_seed,
                            unique_id,
                            index,
                            channel_id=channel_id,
                        )
                    )
                else:
                    tasks.append(
                        self._run_single_sync_request(
                            config,
                            prompt,
                            model,
                            image,
                            mask,
                            quality,
                            aspect_ratio,
                            resolution,
                            background,
                            output_format,
                            output_compression,
                            moderation,
                            max_retries,
                            initial_timeout,
                            request_seed,
                            channel_id=channel_id,
                        )
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_images = []
            all_urls = []
            successful_responses = []
            serializable_responses = []
            errors = []
            for index, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(
                        {
                            "index": index,
                            "error": str(result),
                            "terminal": isinstance(result, LingliuTerminalTaskFailure),
                        }
                    )
                    continue
                all_images.extend(result.get("images", []))
                all_urls.extend(result.get("urls", []))
                successful_responses.append(result)
                serializable_responses.append(self._summarize_single_result(result))

            if errors and not single_submit:
                recovered = []
                recovered_serializable = []
                remaining_errors = []
                for error_item in errors:
                    request_index = int(error_item["index"])
                    if error_item.get("terminal"):
                        remaining_errors.append(error_item)
                        continue
                    plan = request_plan[request_index]
                    try:
                        if async_mode:
                            recovered_result = await self._run_single_async_request(
                                config,
                                prompt,
                                model,
                                image,
                                mask,
                                quality,
                                aspect_ratio,
                                resolution,
                                background,
                                output_format,
                                output_compression,
                                moderation,
                                webhook,
                                max_poll_attempts,
                                poll_interval,
                                max_retries,
                                initial_timeout,
                                plan["seed"],
                                unique_id,
                                request_index,
                                channel_id=channel_id,
                            )
                        else:
                            recovered_result = await self._run_single_sync_request(
                                config,
                                prompt,
                                model,
                                image,
                                mask,
                                quality,
                                aspect_ratio,
                                resolution,
                                background,
                                output_format,
                                output_compression,
                                moderation,
                                max_retries,
                                initial_timeout,
                                plan["seed"],
                                channel_id=channel_id,
                            )
                        recovered.append(recovered_result)
                        recovered_serializable.append(self._summarize_single_result(recovered_result))
                    except Exception as exc:
                        remaining_errors.append({"index": request_index, "error": str(exc)})
                for result in recovered:
                    all_images.extend(result.get("images", []))
                    all_urls.extend(result.get("urls", []))
                    successful_responses.append(result)
                serializable_responses.extend(recovered_serializable)
                errors = remaining_errors

            if not all_images:
                raise RuntimeError(
                    json.dumps(
                        {
                            "requested_n": request_count,
                            "successful_requests": 0,
                            "failed_requests": len(errors),
                            "errors": errors,
                        },
                        ensure_ascii=False,
                    )
                )

            raw_payload = {
                "requested_n": request_count,
                "model_provider": provider_display_name(selected_provider),
                "model": model,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "resolved_size": self.resolve_size(aspect_ratio, resolution),
                "successful_requests": len(successful_responses),
                "failed_requests": len(errors),
                "errors": errors,
                "responses": serializable_responses,
            }
            return (
                pil2tensor(all_images),
                "\n".join(all_urls),
                self.safe_json(raw_payload),
            )
        except Exception as exc:
            return blank_image_tensor(), "", self.error_json(exc)

    async def _generate_jl(
        self,
        config,
        channel_id,
        selected_provider,
        prompt,
        model,
        image,
        mask,
        n,
        quality,
        aspect_ratio,
        resolution,
        background,
        output_format,
        moderation,
        max_poll_attempts,
        poll_interval,
        initial_timeout,
        unique_id,
    ):
        """JL: only platform-accepted fields, one submit per image, flat polling."""
        jl_contract(model)
        if mask is not None:
            raise ImageChannelError("JL 渠道不支持 mask 局部重绘，请断开 mask 或切换到 LL。")
        if background != "auto" or moderation != "auto" or output_format != "png":
            raise ImageChannelError("JL 渠道只支持 background=auto、moderation=auto、output_format=png。")
        if quality not in JL_QUALITIES:
            raise ImageChannelError(f"JL 渠道不支持 quality={quality}。")
        resolution = str(resolution or "").strip().upper()
        validate_jl_resolution(model, resolution)
        validate_jl_ratio(model, aspect_ratio)
        files = await asyncio.to_thread(self.build_jl_reference_files, image)
        validate_jl_references(model, [entry[1][1] for entry in files])
        ratio = str(aspect_ratio or "auto").strip()
        if ratio == "auto" and resolution != "1K":
            # The platform rejects an auto ratio above 1K; derive it locally.
            width, height = image_tensor_size(image) if image is not None else (1, 1)
            ratio = nearest_ratio(width, height, jl_contract(model)["ratios"])
        resolved_size = self.resolve_size(ratio, resolution)
        fields = {"prompt": prompt, "model": model, "quality": quality, "response_format": "url"}
        if resolved_size != "auto":
            fields["size"] = resolved_size
        submit_config = config_with_channel(config, channel_id)
        timeout = int(initial_timeout)

        request_count = max(1, int(n))
        all_images, all_urls, responses, errors = [], [], [], []
        for index in range(request_count):
            task_id = ""
            try:
                if files:
                    submit_payload = await submit_once(
                        self, self.append_query_param(self.endpoint(config, "images_edit_path"), "async", "true"),
                        submit_config, timeout=timeout, headers=self.headers(submit_config), data=fields, files=files,
                    )
                else:
                    submit_payload = await submit_once(
                        self, self.append_query_param(self.endpoint(config, "images_generate_path"), "async", "true"),
                        submit_config, timeout=timeout, headers=self.auth_headers(submit_config, json_mode=True), json=fields,
                    )
                task_id = self.extract_task_id(submit_payload)
                if not task_id:
                    raise RuntimeError(f"No task_id in response: {self.public_payload(submit_payload)}")
                self.publish_task_id(unique_id, task_id, label=f"Task ID #{index + 1}")
                image_items, final_payload = await wait_flat_image_task(
                    self, config, task_id, max_poll_attempts=max_poll_attempts, poll_interval=poll_interval,
                )
                images, urls = await asyncio.to_thread(parse_image_response, {"data": image_items}, timeout)
                if not images:
                    raise RuntimeError(f"Task {task_id} succeeded but no images were decoded.")
                all_images.extend(images)
                all_urls.extend(urls)
                responses.append({"task_id": task_id, "urls": list(urls), "image_count": len(images)})
            except ImageSubmitUncertain as exc:
                self.publish_task_id(unique_id, exc.task_id, label=f"Task ID #{index + 1}")
                errors.append({"index": index, "task_id": exc.task_id, "error": str(exc)})
                break
            except Exception as exc:
                errors.append({"index": index, "task_id": task_id, "error": str(exc)})
                break

        raw_payload = {
            "requested_n": request_count,
            "model_provider": provider_display_name(selected_provider),
            "channel": CHANNEL_JL,
            "model": model,
            "aspect_ratio": ratio,
            "resolution": resolution,
            "resolved_size": resolved_size,
            "input_images": len(files),
            "successful_requests": len(responses),
            "failed_requests": len(errors),
            "errors": errors,
            "responses": responses,
        }
        if not all_images:
            return blank_image_tensor(), "", self.error_json("JL 生成失败", **raw_payload)
        return pil2tensor(all_images), "\n".join(all_urls), self.safe_json(raw_payload)

    async def _generate_fixed(
        self,
        config,
        channel_label,
        selected_provider,
        prompt,
        model,
        image,
        mask,
        n,
        quality,
        aspect_ratio,
        resolution,
        background,
        output_format,
        moderation,
        max_poll_attempts,
        poll_interval,
        unique_id,
    ):
        """-hq2/-hq3/-official/-enterprise: the GPT Image 2.5 node's fixed-channel submit, one image per request."""
        # Imported here because nodes_gpt_image_25 imports this module.
        from .nodes_gpt_image_25 import LingliuGptImage25

        if mask is not None:
            raise ImageChannelError(f"{channel_label}不支持 mask 局部重绘，请断开 mask。")
        submitter = LingliuGptImage25()
        submitter.TASK_LABEL = "GPT Image 2"
        request_count = max(1, int(n))
        result = await submitter.submit_fixed_channel(
            config, prompt, model, aspect_ratio, resolution, quality, request_count, output_format,
            background, moderation, image, True, max_poll_attempts, poll_interval, unique_id, channel_label,
        )
        raw_payload = {
            "requested_n": request_count,
            "model_provider": provider_display_name(selected_provider),
            "channel": channel_label,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "resolved_size": result["request"]["size"],
            "endpoint": result["endpoint"],
            "input_images": result["reference_image_count"],
            "task_ids": result["task_ids"],
            "successful_requests": len(result["responses"]),
            "failed_requests": len(result["errors"]),
            "errors": result["errors"],
            "responses": result["responses"],
        }
        if not result["images"]:
            return blank_image_tensor(), "", self.error_json(f"{channel_label}生成失败", **raw_payload)
        return pil2tensor(result["images"]), "\n".join(result["urls"]), self.safe_json(raw_payload)
