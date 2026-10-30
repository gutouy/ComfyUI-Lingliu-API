import asyncio

from comfy.comfy_types import IO
from PIL import Image, ImageOps

from ..lingliu_config import LINGLIU_PROVIDER, build_url, lingliu_channel_key, provider_defaults
from ..utils import (
    LingliuNullVideoAdapter,
    LingliuVideoAdapter,
    image_to_data_url,
    image_to_png_bytes,
    parse_image_response,
    pil2tensor,
)
from .nodes_openai_compatible import LingliuBaseNode


class LingliuGrokBase(LingliuBaseNode):
    CATEGORY = "Lingliu/Grok"
    LINGLIU_BASE_URL = "https://api.lingliuapi.com"
    ROUTE_CHANNEL_ID = 6

    def load_lingliu_route_config(self, api_key_override=""):
        runtime_config = self.load_config(api_key_override=api_key_override)
        config = provider_defaults(LINGLIU_PROVIDER)
        config["api_key"] = lingliu_channel_key(runtime_config.get("api_key", ""), self.ROUTE_CHANNEL_ID)
        config["base_url"] = self.LINGLIU_BASE_URL
        config["timeout"] = int(runtime_config.get("timeout") or config.get("timeout") or 300)
        config["images_generate_path"] = "/v1/images/generations"
        config["videos_generate_path"] = "/v1/videos"
        config["videos_status_path_template"] = "/v1/videos/{task_id}"
        return config

    def endpoint_from_template(self, config, template_key, **values):
        template = str(config.get(template_key) or "").strip()
        return build_url(config.get("base_url"), template.format(**values))

    def first_video_url(self, payload):
        _image_urls, video_urls = self.extract_media_urls(payload)
        return video_urls[0] if video_urls else ""


class LingliuGrokImage(LingliuGrokBase):
    MODELS = ["grok-4.2-image", "grok-4.1-image"]
    MAX_REFERENCE_IMAGES = 4
    OUTPUT_SIZE_CHOICES = ["1024x1024", "1536x1024", "1024x1536"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "size": (cls.OUTPUT_SIZE_CHOICES, {"default": cls.OUTPUT_SIZE_CHOICES[0]}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "fit_mode": (["crop", "pad", "stretch"], {"default": "crop"}),
                "reference_image1": ("IMAGE",),
                "reference_image2": ("IMAGE",),
                "reference_image3": ("IMAGE",),
                "reference_image4": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "task_id", "response")
    FUNCTION = "generate"

    def normalize_size_choice(self, size):
        raw = str(size or "").strip().lower()
        if raw not in self.OUTPUT_SIZE_CHOICES:
            return self.OUTPUT_SIZE_CHOICES[0]
        return raw

    def parse_output_size(self, output_size):
        raw = self.normalize_size_choice(output_size)
        width_text, height_text = raw.split("x", 1)
        width = int(width_text.strip())
        height = int(height_text.strip())
        if width <= 0 or height <= 0:
            raise ValueError("size must be positive.")
        return width, height

    def aspect_ratio_from_size(self, output_size):
        import math

        width, height = self.parse_output_size(output_size)
        divisor = math.gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    def conform_images_to_size(self, images, output_size, fit_mode):
        width, height = self.parse_output_size(output_size)
        mode = str(fit_mode or "crop").strip().lower()
        conformed = []
        for image in images:
            if image.mode != "RGB":
                image = image.convert("RGB")
            if mode == "stretch":
                conformed.append(image.resize((width, height), Image.Resampling.LANCZOS))
            elif mode == "pad":
                contained = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (width, height), "white")
                canvas.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
                conformed.append(canvas)
            else:
                conformed.append(ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5)))
        return conformed

    def collect_reference_image_urls(self, *images):
        references = []
        for image in images:
            if image is None:
                continue
            batch_count = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for index in range(batch_count):
                if len(references) >= self.MAX_REFERENCE_IMAGES:
                    raise ValueError(f"Grok image supports at most {self.MAX_REFERENCE_IMAGES} reference images.")
                single_image = image[index : index + 1] if batch_count > 1 else image
                references.append(image_to_data_url(single_image))
        return references

    def build_payload_attempts(self, *, prompt, model, output_size, reference_images):
        model = str(model or "").strip() or self.MODELS[0]
        output_size = self.normalize_size_choice(output_size)
        base_payload = {
            "model": model,
            "prompt": str(prompt or ""),
            "image": list(reference_images or []),
        }
        if model == "grok-4.1-image":
            payload = dict(base_payload)
            payload["aspect_ratio"] = self.aspect_ratio_from_size(output_size)
            return [payload]

        payload = dict(base_payload)
        payload["size"] = output_size
        return [payload]

    def build_payload(self, *, prompt, model, output_size=None, reference_images=None, size=None):
        return self.build_payload_attempts(
            prompt=prompt,
            model=model,
            output_size=output_size or size,
            reference_images=reference_images,
        )[0]

    async def request_first_image(self, *, config, url, payloads):
        payload = payloads[0]
        response_payload = await self.async_request_json_retrying(
            "POST",
            url,
            config,
            max_retries=1,
            initial_timeout=self.timeout(config),
            headers=self.headers(config, {"Content-Type": "application/json"}),
            json=payload,
        )
        images, urls = await asyncio.to_thread(parse_image_response, response_payload, self.timeout(config))
        if not images:
            raise RuntimeError(f"Grok image response did not contain an image: {response_payload}")
        return images, urls, response_payload, payload, 1

    async def generate(
        self,
        prompt,
        model,
        size="1024x1024",
        api_key="",
        fit_mode="crop",
        reference_image1=None,
        reference_image2=None,
        reference_image3=None,
        reference_image4=None,
        unique_id=None,
        output_size=None,
    ):
        output_size = self.normalize_size_choice(output_size or size)
        try:
            config = self.load_lingliu_route_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                raise RuntimeError("API key is required for lingliu_grok_image.")
            reference_images = self.collect_reference_image_urls(
                reference_image1,
                reference_image2,
                reference_image3,
                reference_image4,
            )
            payloads = self.build_payload_attempts(
                prompt=prompt,
                model=model,
                output_size=output_size,
                reference_images=reference_images,
            )
            url = self.endpoint(config, "images_generate_path")
            images, urls, response_payload, payload, attempt_index = await self.request_first_image(
                config=config,
                url=url,
                payloads=payloads,
            )
            task_id = ""
            images = self.conform_images_to_size(images, output_size, fit_mode)
            task_id = task_id or self.extract_task_id(response_payload)
            if task_id:
                self.publish_task_id(unique_id, task_id, label="Task ID", append=False)
            response_info = {
                "capabilities": {
                    "reference_image_max": self.MAX_REFERENCE_IMAGES,
                    "reference_image_transport": "JSON image[] data URLs",
                    "size_choices": self.OUTPUT_SIZE_CHOICES,
                },
                "request": {
                    "model": payload["model"],
                    "size": output_size,
                    "fit_mode": str(fit_mode or "crop").strip(),
                    "upstream_size": payload.get("size", ""),
                    "upstream_aspect_ratio": payload.get("aspect_ratio", ""),
                    "reference_image_count": len(reference_images),
                    "attempt_index": attempt_index,
                    "attempt_count": len(payloads),
                },
                "response": response_payload,
            }
            return pil2tensor(images), "\n".join(urls), task_id, self.safe_json(response_info)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc


class LingliuGrokVideo(LingliuGrokBase):
    MODELS = ["grok-1.5-video-6s", "grok-1.5-video-10s", "grok-1.5-video-15s"]
    MAX_REFERENCE_IMAGES = 1
    SIZE_CHOICES = ["1280x720", "720x1280", "1024x1024"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.MODELS, {"default": cls.MODELS[0]}),
                "size": (cls.SIZE_CHOICES, {"default": cls.SIZE_CHOICES[0]}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "reference_image": ("IMAGE",),
                "wait_for_completion": ("BOOLEAN", {"default": True}),
                "max_poll_attempts": ("INT", {"default": 180, "min": 1, "max": 720}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 30}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "response", "video_url")
    FUNCTION = "generate"

    def normalize_size_choice(self, size):
        raw = str(size or "").strip().lower()
        if raw not in self.SIZE_CHOICES:
            return self.SIZE_CHOICES[0]
        return raw

    def parse_size(self, size):
        raw = self.normalize_size_choice(size)
        width_text, height_text = raw.split("x", 1)
        return int(width_text), int(height_text)

    def build_multipart_files(self, *, model, prompt, size, reference_image=None):
        size = self.normalize_size_choice(size)
        files = [
            ("model", (None, str(model or "").strip() or self.MODELS[0])),
            ("prompt", (None, str(prompt or ""))),
            ("size", (None, size)),
        ]
        if reference_image is not None:
            batch_count = reference_image.shape[0] if hasattr(reference_image, "shape") and len(reference_image.shape) == 4 else 1
            if batch_count > self.MAX_REFERENCE_IMAGES:
                raise ValueError(f"Grok video supports at most {self.MAX_REFERENCE_IMAGES} reference image.")
            single_image = reference_image[:1] if batch_count > 1 else reference_image
            files.append(("input_reference", ("input_reference.png", image_to_png_bytes(single_image), "image/png")))
        return files

    async def wait_for_video_task(self, config, task_id, *, max_poll_attempts=180, poll_interval=5):
        task_url = self.endpoint_from_template(config, "videos_status_path_template", task_id=task_id)
        last_payload = None
        for _ in range(int(max_poll_attempts)):
            await asyncio.sleep(max(1, int(poll_interval)))
            payload = await self.async_request_json("GET", task_url, config, headers=self.headers(config))
            last_payload = payload
            video_url = self.first_video_url(payload)
            status = self.extract_status(payload)
            if video_url and (not status or self.is_success_status(status) or str(self.extract_progress(payload)) == "100"):
                return payload, video_url
            if self.is_success_status(status) and video_url:
                return payload, video_url
            if self.is_failure_status(status):
                detail = self.extract_failure_detail(payload)
                raise RuntimeError(f"Grok video task failed: {detail or payload}")
        raise TimeoutError(f"Grok video task {task_id} timed out. Last payload: {last_payload}")

    async def generate(
        self,
        prompt,
        model,
        size="1280x720",
        api_key="",
        reference_image=None,
        wait_for_completion=True,
        max_poll_attempts=180,
        poll_interval=5,
        unique_id=None,
    ):
        try:
            size = self.normalize_size_choice(size)
            self.publish_task_id(unique_id, clear=True)
            config = self.load_lingliu_route_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_grok_video."
                return LingliuNullVideoAdapter(error), "", self.error_json(error), ""
            files = self.build_multipart_files(
                model=model,
                prompt=prompt,
                size=size,
                reference_image=reference_image,
            )
            submit_url = self.endpoint(config, "videos_generate_path")
            submit_payload = await self.async_request_json_retrying(
                "POST",
                submit_url,
                config,
                max_retries=3,
                initial_timeout=self.timeout(config),
                headers=self.headers(config),
                files=files,
            )
            task_id = self.extract_task_id(submit_payload)
            if task_id:
                self.publish_task_id(unique_id, task_id)
            video_url = self.first_video_url(submit_payload)
            final_payload = submit_payload
            if wait_for_completion and not video_url:
                if not task_id:
                    raise RuntimeError(f"Grok video submit response did not include task_id: {submit_payload}")
                final_payload, video_url = await self.wait_for_video_task(
                    config,
                    task_id,
                    max_poll_attempts=max_poll_attempts,
                    poll_interval=poll_interval,
                )
            response_info = {
                "capabilities": {
                    "reference_image_max": self.MAX_REFERENCE_IMAGES,
                    "reference_image_transport": "multipart input_reference",
                    "size_choices": self.SIZE_CHOICES,
                    "output_video_sizing": "upstream original size",
                },
                "request": {
                    "model": str(model or "").strip() or self.MODELS[0],
                    "size": size,
                    "reference_image_count": 1 if reference_image is not None else 0,
                },
                "submit": submit_payload,
                "final": final_payload,
            }
            if not video_url:
                return LingliuNullVideoAdapter("No video URL yet."), task_id, self.safe_json(response_info), ""
            response_info["video_url"] = video_url
            response_info["output_size"] = size
            return LingliuVideoAdapter(video_url, mime_type="video/mp4"), task_id, self.safe_json(response_info), video_url
        except Exception as exc:
            return LingliuNullVideoAdapter(str(exc)), "", self.error_json(exc), ""
