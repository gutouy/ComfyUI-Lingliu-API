import asyncio
import json

import requests
from comfy.comfy_types import IO

from ..lingliu_config import (
    PUBLIC_PROVIDER_DISPLAY_CHOICES,
    LINGLIU_BASE_URL,
    build_url,
    get_config,
    lingliu_channel_key,
    normalize_provider,
    provider_defaults,
    provider_display_name,
)
from ..utils import LingliuNullVideoAdapter, LingliuVideoAdapter, blank_image_tensor, parse_image_response, pil2tensor
from .nodes_openai_compatible import LingliuBaseNode
from .lingliu_image_channels import parse_flat_image_task


def current_task_provider_default():
    return provider_display_name("lingliu")


class LingliuTaskQuery(LingliuBaseNode):
    CATEGORY = "Lingliu/Query"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "task_provider": (PUBLIC_PROVIDER_DISPLAY_CHOICES, {"default": current_task_provider_default()}),
                "task_type": (
                    [
                        "auto",
                        "image_task",
                        "video_task",
                        "seedance_task",
                        "seedance_official_task",
                        "fal_task",
                    ],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", IO.VIDEO, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "video", "image_url", "video_url", "response", "task_status")
    FUNCTION = "query"

    def _not_found(self, exc):
        return isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None and exc.response.status_code == 404

    def _image_task_url(self, config, task_id):
        template = config.get("images_task_status_path_template", "")
        return build_url(config.get("base_url"), template.format(task_id=task_id))

    def _video_task_url(self, config, task_id):
        template = config.get("videos_status_path_template", "")
        return build_url(config.get("base_url"), template.format(task_id=task_id))

    def _seedance_task_url(self, task_id):
        return build_url(LINGLIU_BASE_URL, f"/seedance/v3/contents/generations/tasks/{task_id}")

    def _seedance_official_task_url(self, task_id):
        return build_url(LINGLIU_BASE_URL, f"/v1/video/generations/{task_id}")

    def _status_text(self, status):
        normalized = str(status or "").strip().lower()
        if normalized in ("success", "succeeded", "completed", "done"):
            return "生成成功"
        if normalized in ("failure", "failed", "error", "cancelled", "canceled", "rejected"):
            return "生成失败"
        return "正在生成"

    def _extract_video_url_from_image_task_payload(self, payload):
        outer = payload.get("data", {}) if isinstance(payload, dict) else {}
        action = str(outer.get("action", "")).lower()

        inner = outer.get("data", {}) if isinstance(outer, dict) else {}
        if isinstance(inner, dict):
            content = inner.get("content")
            if isinstance(content, dict):
                video_url = content.get("video_url") or content.get("videoUrl") or ""
                if video_url:
                    return video_url
            video_url = inner.get("video_url") or inner.get("videoUrl") or ""
            if video_url:
                return video_url

        if "video" in action:
            return "__video_task_without_url__"
        return ""

    def config_for_provider(self, task_provider, api_key_override=""):
        task_provider = normalize_provider(task_provider)
        provider_config = provider_defaults(task_provider)
        runtime_config = self.load_config(api_key_override=api_key_override)
        provider_config["api_key"] = runtime_config.get("api_key", "")
        provider_config["timeout"] = runtime_config.get("timeout", provider_config.get("timeout", 300))
        return provider_config

    def normalize_task_type(self, task_type):
        return str(task_type or "").strip()

    def display_task_type(self, task_type):
        return str(task_type or "").strip()

    def split_fal_task_id(self, task_id):
        value = str(task_id or "").strip()
        for separator in ("::", "|"):
            if separator in value:
                endpoint_id, request_id = value.split(separator, 1)
                endpoint_id = endpoint_id.strip().strip("/")
                request_id = request_id.strip()
                if endpoint_id and request_id:
                    return endpoint_id, request_id
        raise RuntimeError("fal_task 查询需要使用 endpoint_id::request_id 格式，例如 openai/gpt-image-2::xxxx。")

    def _unwrap_platform_payload(self, payload):
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload if isinstance(payload, dict) else {}

    def _normalize_platform_status(self, payload):
        payload = self._unwrap_platform_payload(payload)
        raw = payload.get("status", "") or payload.get("state", "") or ""
        normalized = str(raw or "").strip().lower()
        if normalized in ("success", "done", "completed"):
            return "succeeded"
        if normalized in ("fail", "failure", "error"):
            return "failed"
        return normalized

    def _extract_platform_video_url(self, payload):
        payload = self._unwrap_platform_payload(payload)
        for container in (
            payload,
            payload.get("result") if isinstance(payload, dict) else None,
            payload.get("content") if isinstance(payload, dict) else None,
        ):
            if not isinstance(container, dict):
                continue
            for key in ("video_url", "videoUrl", "result_url", "resultUrl", "url", "video"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    async def asyncio_to_thread_parse(self, items, config):
        payload = {"data": items}
        images, urls = await asyncio.to_thread(parse_image_response, payload, self.timeout(config))
        if not images:
            return None, ""
        return pil2tensor(images), ("\n".join(urls) if urls else "")

    async def query_image_task(self, config, task_id):
        payload = await self.async_request_json_retrying(
            "GET",
            self._image_task_url(config, task_id),
            config,
            max_retries=12,
            headers=self.headers(config),
        )
        flat = parse_flat_image_task(payload)
        if flat is not None:
            # Platform-native image task (JL channel): flat envelope.
            flat_state, flat_items, _flat_error = flat
            result = {
                "matched": True,
                "kind": "image_task",
                "status": str(payload.get("status", "")),
                "image": blank_image_tensor(),
                "video": LingliuNullVideoAdapter("No video in image task."),
                "image_url": "",
                "video_url": "",
                "payload": payload,
            }
            if flat_state == "success" and flat_items:
                images, urls = await self.asyncio_to_thread_parse(flat_items, config)
                if images is not None:
                    result["image"] = images
                    result["image_url"] = urls
            return result
        outer = payload.get("data", {}) if isinstance(payload, dict) else {}
        status = outer.get("status", "")
        result = {
            "matched": True,
            "kind": "image_task",
            "status": status,
            "image": blank_image_tensor(),
            "video": LingliuNullVideoAdapter("No video in image task."),
            "image_url": "",
            "video_url": "",
            "payload": payload,
        }
        if status == "SUCCESS":
            video_url = self._extract_video_url_from_image_task_payload(payload)
            if video_url:
                result["kind"] = "seedance_task" if video_url != "__video_task_without_url__" else "video_task"
                result["video_url"] = "" if video_url == "__video_task_without_url__" else video_url
                result["video"] = (
                    LingliuVideoAdapter(video_url)
                    if video_url and video_url != "__video_task_without_url__"
                    else LingliuNullVideoAdapter("Video task succeeded but no video URL was returned.")
                )
                return result

            inner = outer.get("data", {})
            items = inner.get("data", []) if isinstance(inner, dict) else []
            images, urls = await self.asyncio_to_thread_parse(items, config)
            if images is not None:
                result["image"] = images
                result["image_url"] = urls
        return result

    async def query_video_task(self, config, task_id):
        payload = await self.async_request_json_retrying(
            "GET",
            self._video_task_url(config, task_id),
            config,
            max_retries=12,
            headers=self.headers(config),
        )
        status = payload.get("status", "")
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        video_url = data.get("output", "") if isinstance(data, dict) else ""
        return {
            "matched": True,
            "kind": "video_task",
            "status": status,
            "image": blank_image_tensor(),
            "video": LingliuVideoAdapter(video_url) if video_url else LingliuNullVideoAdapter("Video not ready."),
            "image_url": "",
            "video_url": video_url,
            "payload": payload,
        }

    async def query_seedance_task(self, config, task_id):
        config = dict(config or {})
        config["api_key"] = lingliu_channel_key(config.get("api_key", ""), 6)
        payload = await self.async_request_json_retrying(
            "GET",
            self._seedance_task_url(task_id),
            config,
            max_retries=12,
            headers={"Authorization": f"Bearer {(config.get('api_key') or '').strip()}"},
        )
        status = str(payload.get("status", "")).lower()
        if status == "success":
            status = "succeeded"
        elif status in ("fail", "failure"):
            status = "failed"

        video_url = ""
        root_content = payload.get("content")
        if isinstance(root_content, dict):
            video_url = root_content.get("video_url") or root_content.get("videoUrl") or ""

        data = payload.get("data")
        if isinstance(data, dict) and not video_url:
            data_content = data.get("content")
            if isinstance(data_content, dict):
                video_url = data_content.get("video_url") or data_content.get("videoUrl") or ""
            if not video_url:
                video_url = data.get("video_url") or data.get("videoUrl") or ""

        if not video_url:
            results = payload.get("results", [])
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

        return {
            "matched": True,
            "kind": "seedance_task",
            "status": status,
            "image": blank_image_tensor(),
            "video": LingliuVideoAdapter(video_url) if video_url else LingliuNullVideoAdapter("Video not ready."),
            "image_url": "",
            "video_url": video_url,
            "payload": payload,
        }

    async def query_seedance_official_task(self, config, task_id):
        config = dict(config or {})
        config["api_key"] = lingliu_channel_key(config.get("api_key", ""), 5)
        payload = await self.async_request_json_retrying(
            "GET",
            self._seedance_official_task_url(task_id),
            config,
            max_retries=12,
            headers=self.headers(config),
        )
        status = str(payload.get("status", "")).lower()
        if status == "success":
            status = "succeeded"
        elif status in ("fail", "failure"):
            status = "failed"

        image_urls, video_urls = self.extract_media_urls(payload)
        video_url = video_urls[0] if video_urls else ""
        if not video_url:
            data = payload.get("data")
            if isinstance(data, dict):
                video_url = data.get("video_url") or data.get("videoUrl") or data.get("url") or ""
        return {
            "matched": True,
            "kind": "seedance_official_task",
            "status": status,
            "image": blank_image_tensor(),
            "video": LingliuVideoAdapter(video_url) if video_url else LingliuNullVideoAdapter("Video not ready."),
            "image_url": "",
            "video_url": video_url,
            "payload": payload,
        }

    async def query_fal_task(self, config, task_id):
        endpoint_id, request_id = self.split_fal_task_id(task_id)
        payload, image_urls, video_urls = await self.query_fal_result(
            config,
            endpoint_id,
            request_id,
            max_retries=12,
            poll_interval=1,
        )
        image = blank_image_tensor()
        image_url_text = ""
        if image_urls:
            image, image_url_text = await self.asyncio_to_thread_parse([{"url": url} for url in image_urls], config)
            if image is None:
                image = blank_image_tensor()

        status = self.extract_status(payload) or ("success" if (image_urls or video_urls) else "pending")
        return {
            "matched": True,
            "kind": "fal_task" if not video_urls else "video_task",
            "status": status,
            "image": image,
            "video": LingliuVideoAdapter(video_urls[0]) if video_urls else LingliuNullVideoAdapter("Video not ready."),
            "image_url": image_url_text or ("\n".join(image_urls) if image_urls else ""),
            "video_url": video_urls[0] if video_urls else "",
            "payload": payload,
        }

    async def query(self, task_id, task_provider="lingliu", task_type="auto", api_key=""):
        config = self.config_for_provider(task_provider, api_key_override=api_key)
        if not (config.get("api_key") or "").strip():
            error = "API key is required for lingliu_task_query."
            return blank_image_tensor(), LingliuNullVideoAdapter(error), "", "", self.error_json(error), "生成失败"

        task_id = (task_id or "").strip()
        if not task_id:
            error = "task_id is required."
            return blank_image_tensor(), LingliuNullVideoAdapter(error), "", "", self.error_json(error), "生成失败"

        provider_label = provider_display_name(normalize_provider(task_provider))
        task_type = self.normalize_task_type(task_type)
        if task_type == "auto":
            query_plan = [
                ("image_task", self.query_image_task),
                ("video_task", self.query_video_task),
                ("seedance_task", self.query_seedance_task),
                ("seedance_official_task", self.query_seedance_official_task),
            ]
        elif task_type == "image_task":
            query_plan = [("image_task", self.query_image_task)]
        elif task_type == "video_task":
            query_plan = [("video_task", self.query_video_task)]
        elif task_type == "fal_task":
            query_plan = [("fal_task", self.query_fal_task)]
        elif task_type == "seedance_official_task":
            query_plan = [("seedance_official_task", self.query_seedance_official_task)]
        else:
            query_plan = [("seedance_task", self.query_seedance_task)]

        last_error = None
        for kind, query_fn in query_plan:
            try:
                result = await query_fn(config, task_id)
                response = {
                    "query_provider": provider_label,
                    "matched_type": self.display_task_type(result.get("kind", kind)),
                    "task_id": task_id,
                    "status": result["status"],
                    "payload": result["payload"],
                }
                return (
                    result["image"],
                    result["video"],
                    result["image_url"],
                    result["video_url"],
                    self.safe_json(response),
                    self._status_text(result["status"]),
                )
            except Exception as exc:
                if self._not_found(exc):
                    last_error = exc
                    continue
                response = {
                    "query_provider": provider_label,
                    "matched_type": self.display_task_type(kind),
                    "task_id": task_id,
                    "error": str(exc),
                }
                return (
                    blank_image_tensor(),
                    LingliuNullVideoAdapter(str(exc)),
                    "",
                    "",
                    self.safe_json(response),
                    "生成失败",
                )

        if last_error is not None:
            return (
                blank_image_tensor(),
                LingliuNullVideoAdapter("Task not found."),
                "",
                "",
                self.error_json("Task not found in supported task endpoints.", task_id=task_id, query_provider=provider_label),
                "生成失败",
            )

        return (
            blank_image_tensor(),
            LingliuNullVideoAdapter("Task query failed."),
            "",
            "",
            self.error_json("Task query failed.", task_id=task_id, query_provider=provider_label),
            "生成失败",
        )
