import asyncio

from comfy.comfy_types import IO

from ..lingliu_config import LINGLIU_PROVIDER, build_url, lingliu_channel_key, provider_defaults
from ..utils import LingliuNullVideoAdapter, LingliuVideoAdapter, image_to_png_bytes
from .nodes_openai_compatible import LingliuBaseNode


class LingliuOmniVideo(LingliuBaseNode):
    CATEGORY = "Lingliu/Omni"
    LINGLIU_BASE_URL = "https://api.lingliuapi.com"
    OMNI_CHANNEL_ID = 10
    MODELS = ["omni_flash-10s"]
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

    def load_lingliu_omni_config(self, api_key_override=""):
        runtime_config = self.load_config(api_key_override=api_key_override)
        config = provider_defaults(LINGLIU_PROVIDER)
        config["base_url"] = self.LINGLIU_BASE_URL
        config["api_key"] = lingliu_channel_key(runtime_config.get("api_key", ""), self.OMNI_CHANNEL_ID)
        config["timeout"] = int(runtime_config.get("timeout") or config.get("timeout") or 300)
        config["videos_generate_path"] = "/v1/videos"
        config["videos_status_path_template"] = "/v1/videos/{task_id}"
        return config

    def endpoint_from_template(self, config, template_key, **values):
        template = str(config.get(template_key) or "").strip()
        return build_url(config.get("base_url"), template.format(**values))

    def first_video_url(self, payload):
        _image_urls, video_urls = self.extract_media_urls(payload)
        return video_urls[0] if video_urls else ""

    def normalize_size_choice(self, size):
        raw = str(size or "").strip().lower()
        if raw not in self.SIZE_CHOICES:
            return self.SIZE_CHOICES[0]
        return raw

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
                raise ValueError(f"Omni video supports at most {self.MAX_REFERENCE_IMAGES} reference image.")
            single_image = reference_image[:1] if batch_count > 1 else reference_image
            files.append(("input_reference", ("input_reference.png", image_to_png_bytes(single_image), "image/png")))
        return files

    async def wait_for_video_task(self, config, task_id, *, max_poll_attempts=180, poll_interval=5):
        task_url = self.endpoint_from_template(config, "videos_status_path_template", task_id=task_id)
        last_payload = None
        for _ in range(int(max_poll_attempts)):
            await asyncio.sleep(max(0, int(poll_interval)))
            payload = await self.async_request_json("GET", task_url, config, headers=self.headers(config))
            last_payload = payload
            video_url = self.first_video_url(payload)
            status = self.extract_status(payload)
            if video_url and (not status or self.is_success_status(status) or str(self.extract_progress(payload)) == "100"):
                return payload, video_url
            if self.is_failure_status(status):
                detail = self.extract_failure_detail(payload)
                raise RuntimeError(f"Omni video task failed: {detail or payload}")
        raise TimeoutError(f"Omni video task {task_id} timed out. Last payload: {last_payload}")

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
            config = self.load_lingliu_omni_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_omni_video."
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
                max_retries=1,
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
                    raise RuntimeError(f"Omni video submit response did not include task_id: {submit_payload}")
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
