import asyncio
import json
import math
from urllib.parse import quote, urlparse

import requests

from ..lingliu_config import (
    PUBLIC_PROVIDER_DISPLAY_CHOICES,
    build_url,
    get_config,
    mask_provider_payload,
    mask_provider_text,
    normalize_provider,
    provider_defaults,
    provider_display_name,
    save_config,
    set_runtime_config,
)
from ..utils import blank_image_tensor, content_to_text, image_to_data_url, image_to_png_bytes, load_image_from_candidate, parse_image_response, pil2tensor


class LingliuBaseNode:
    CATEGORY = "Lingliu/API"

    def load_config(self, api_key_override=""):
        config = get_config()
        if api_key_override.strip():
            config["api_key"] = api_key_override.strip()
        return config

    def load_chat_config(self, api_key_override=""):
        config = self.load_config(api_key_override=api_key_override)
        return config

    def provider_name(self, config):
        return normalize_provider((config or {}).get("api_provider"))

    def provider_display_name(self, provider_or_config):
        if isinstance(provider_or_config, dict):
            provider_or_config = provider_or_config.get("api_provider")
        return provider_display_name(provider_or_config)

    def endpoint_for_path(self, config, path):
        return build_url(config.get("base_url"), path)

    def headers(self, config, extra_headers=None):
        headers = {"Accept": "application/json"}
        api_key = (config.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def endpoint(self, config, key):
        return build_url(config.get("base_url"), config.get(key))

    def fal_endpoint(self, config, endpoint_id):
        endpoint_id = str(endpoint_id or "").strip().strip("/")
        if not endpoint_id:
            raise RuntimeError("fal endpoint_id is required.")
        return build_url(config.get("base_url"), endpoint_id)

    def timeout(self, config):
        try:
            return int(config.get("timeout", 300))
        except Exception:
            return 300

    def _parse_sse_payload(self, text):
        candidates = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                candidates.append(json.loads(payload))
            except Exception:
                continue
        return candidates[-1] if candidates else None

    def _decode_json_response(self, response, url):
        try:
            return response.json()
        except ValueError as exc:
            text = response.text or ""
            sse_payload = self._parse_sse_payload(text)
            if sse_payload is not None:
                return sse_payload
            content_type = response.headers.get("Content-Type", "")
            snippet = " ".join(text.strip().split())
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            raise RuntimeError(
                f"API did not return JSON. URL: {url}. Content-Type: {content_type}. Body preview: {snippet}"
            ) from exc

    def explain_known_api_error(self, payload):
        text = json.dumps(payload, ensure_ascii=False) if isinstance(payload, (dict, list)) else str(payload or "")
        lowered = text.lower()
        if "invalid_api_platform" in lowered and "59" in lowered:
            return (
                "Seedance 官方路由提示：/v1/video/generations 当前会命中服务端缺失的 platform 59 task adaptor。"
                "节点已改用该站点实际可用的 /v1/videos 通用异步任务路由；"
                "如果仍看到此错误，请重启 ComfyUI 确保新节点代码已加载。"
            )
        return ""

    def _raise_for_status_with_details(self, response, url):
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            content_type = response.headers.get("Content-Type", "")
            body_preview = ""
            body_json = None
            try:
                body_json = response.json()
            except Exception:
                text = response.text or ""
                snippet = " ".join(text.strip().split())
                if len(snippet) > 1000:
                    snippet = snippet[:1000] + "..."
                body_preview = snippet

            detail_parts = [mask_provider_text(f"{response.status_code} Client Error: {response.reason} for url: {url}")]
            if body_json is not None:
                detail_parts.append(f"Response JSON: {json.dumps(mask_provider_payload(body_json), ensure_ascii=False)}")
                known_explanation = self.explain_known_api_error(body_json)
                if known_explanation:
                    detail_parts.append(known_explanation)
            elif body_preview:
                detail_parts.append(f"Response Body: {mask_provider_text(body_preview)}")
                known_explanation = self.explain_known_api_error(body_preview)
                if known_explanation:
                    detail_parts.append(known_explanation)
            elif content_type:
                detail_parts.append(f"Content-Type: {content_type}")

            error = RuntimeError(" | ".join(detail_parts))
            error.status_code = response.status_code
            raise error from exc

    def request_json(self, method, url, config, **kwargs):
        response = requests.request(method, url, timeout=self.timeout(config), **kwargs)
        self._raise_for_status_with_details(response, url)
        return self._decode_json_response(response, url)

    async def async_request_json(self, method, url, config, **kwargs):
        return await asyncio.to_thread(self.request_json, method, url, config, **kwargs)

    def request_json_with_timeout(self, method, url, timeout, **kwargs):
        response = requests.request(method, url, timeout=timeout, **kwargs)
        self._raise_for_status_with_details(response, url)
        return self._decode_json_response(response, url)

    async def async_request_json_with_timeout(self, method, url, timeout, **kwargs):
        return await asyncio.to_thread(self.request_json_with_timeout, method, url, timeout, **kwargs)

    def is_transient_request_error(self, exc):
        transient_types = (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
        )
        return isinstance(exc, transient_types)

    def error_json(self, message, **extra):
        payload = {"error": mask_provider_text(message)}
        payload.update(extra)
        return self.safe_json(payload)

    def public_payload(self, payload):
        return mask_provider_payload(payload)

    def compact_large_strings(self, payload):
        if isinstance(payload, dict):
            return {key: self.compact_large_strings(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self.compact_large_strings(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.compact_large_strings(item) for item in payload)
        if isinstance(payload, str):
            text = payload.strip()
            lowered = text.lower()
            if lowered.startswith(("data:image/", "data:video/", "data:application/")):
                return mask_provider_text(text)
            if len(text) > 4000:
                compacted = mask_provider_text(text)
                if len(compacted) <= 4000:
                    return compacted
                return f"{compacted[:1200]}... <truncated, {len(compacted)} chars>"
        return payload

    def safe_json(self, payload):
        return json.dumps(self.compact_large_strings(self.public_payload(payload)), ensure_ascii=False, indent=2)

    def normalize_unique_id(self, unique_id):
        if unique_id is None:
            return ""
        if isinstance(unique_id, (list, tuple)) and unique_id:
            unique_id = unique_id[0]
        unique_id = str(unique_id)
        if "." in unique_id:
            unique_id = unique_id.split(".")[-1]
        if ":" in unique_id:
            unique_id = unique_id.split(":")[0]
        return unique_id

    def publish_task_id(self, unique_id, task_id="", label="Task ID", append=True, clear=False):
        node_id = self.normalize_unique_id(unique_id)
        if not node_id:
            return
        try:
            from server import PromptServer

            if PromptServer.instance is None or PromptServer.instance.client_id is None:
                return
            task_id = str(task_id or "").strip()
            PromptServer.instance.send_sync(
                "lingliu-task-id",
                {
                    "node_id": node_id,
                    "task_id": task_id,
                    "message": f"{label}: {task_id}" if task_id else "",
                    "append": bool(append),
                    "clear": bool(clear),
                },
                PromptServer.instance.client_id,
            )
        except Exception:
            pass

    def publish_live_message(self, unique_id, message="", append=False, clear=False):
        node_id = self.normalize_unique_id(unique_id)
        if not node_id:
            return
        try:
            from server import PromptServer

            if PromptServer.instance is None or PromptServer.instance.client_id is None:
                return
            PromptServer.instance.send_sync(
                "lingliu-task-id",
                {
                    "node_id": node_id,
                    "task_id": "",
                    "message": str(message or ""),
                    "append": bool(append),
                    "clear": bool(clear),
                },
                PromptServer.instance.client_id,
            )
        except Exception:
            pass

    def extract_task_id(self, payload):
        if isinstance(payload, dict):
            for key in ("task_id", "taskId", "request_id", "requestId", "id"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in payload.values():
                nested = self.extract_task_id(value)
                if nested:
                    return nested
        elif isinstance(payload, list):
            for item in payload:
                nested = self.extract_task_id(item)
                if nested:
                    return nested
        return ""

    def extract_status(self, payload):
        if isinstance(payload, dict):
            for key, value in payload.items():
                if str(key).strip().lower() in {"status", "state"} and isinstance(value, (str, int, float)):
                    return str(value)
            for value in payload.values():
                nested = self.extract_status(value)
                if nested:
                    return nested
        elif isinstance(payload, list):
            for item in payload:
                nested = self.extract_status(item)
                if nested:
                    return nested
        return ""

    def extract_progress(self, payload):
        if isinstance(payload, dict):
            for key, value in payload.items():
                normalized_key = str(key).strip().lower()
                if normalized_key in {"progress", "progress_percent", "progresspercentage", "percent", "percentage"}:
                    if isinstance(value, (str, int, float)):
                        text = str(value).strip()
                        if text:
                            return text
                nested = self.extract_progress(value)
                if nested:
                    return nested
        elif isinstance(payload, list):
            for item in payload:
                nested = self.extract_progress(item)
                if nested:
                    return nested
        return ""

    def is_success_status(self, status):
        normalized = str(status or "").strip().lower()
        return normalized in {"success", "succeeded", "completed", "done", "finish", "finished", "active"}

    def is_failure_status(self, status):
        normalized = str(status or "").strip().lower()
        return normalized in {
            "fail",
            "failure",
            "failed",
            "error",
            "cancelled",
            "canceled",
            "deleted",
            "rejected",
            "failed_task",
            "任务失败",
            "失败",
            "已失败",
        }

    def extract_failure_detail(self, payload):
        details = []
        interesting_keys = {
            "failure_reason",
            "fail_reason",
            "error",
            "message",
            "msg",
            "reason",
            "detail",
            "details",
        }

        def add_detail(value):
            if value is None:
                return
            text = str(value).strip()
            if not text or text.lower() in {"success", "succeeded", "ok", "none", "null"}:
                return
            if text not in details:
                details.append(text)

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    normalized_key = str(key).strip().lower()
                    if normalized_key in interesting_keys:
                        add_detail(item)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(payload)
        return " | ".join(details)

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

    def _looks_like_url(self, value):
        value = str(value or "").strip()
        return value.startswith(("http://", "https://", "data:image/", "data:video/"))

    def extract_media_urls(self, payload):
        image_urls = []
        video_urls = []

        def add_url(value, key_hint=""):
            if not isinstance(value, str):
                return
            candidate = value.strip()
            if not self._looks_like_url(candidate):
                return
            lowered = candidate.lower()
            key_hint = str(key_hint or "").lower()
            is_video = (
                "video" in key_hint
                or lowered.endswith(".mp4")
                or lowered.endswith(".mov")
                or lowered.endswith(".webm")
                or lowered.startswith("data:video/")
            )
            is_image = (
                "image" in key_hint
                or "img" in key_hint
                or lowered.endswith(".png")
                or lowered.endswith(".jpg")
                or lowered.endswith(".jpeg")
                or lowered.endswith(".webp")
                or lowered.startswith("data:image/")
            )
            if is_video:
                if candidate not in video_urls:
                    video_urls.append(candidate)
                return
            if is_image or any(token in key_hint for token in ("url", "result", "output")):
                if candidate not in image_urls:
                    image_urls.append(candidate)

        def walk(value, parent_key=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    add_url(item, key)
                    walk(item, key)
            elif isinstance(value, list):
                for item in value:
                    walk(item, parent_key)
            else:
                add_url(value, parent_key)

        walk(payload)
        return image_urls, video_urls

    def extract_fal_request_id(self, payload):
        if isinstance(payload, dict):
            for key in ("request_id", "requestId", "task_id", "taskId", "id"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for key in ("data", "result"):
                nested = self.extract_fal_request_id(payload.get(key))
                if nested:
                    return nested
        return ""

    def extract_fal_url(self, payload, keys):
        wanted = {str(key).strip().lower() for key in keys}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if str(key).strip().lower() in wanted and isinstance(value, str) and value.strip():
                    return value.strip()
            for value in payload.values():
                nested = self.extract_fal_url(value, wanted)
                if nested:
                    return nested
        elif isinstance(payload, list):
            for item in payload:
                nested = self.extract_fal_url(item, wanted)
                if nested:
                    return nested
        return ""

    def normalize_fal_proxy_url(self, config, url):
        url = str(url or "").strip()
        if not url:
            return ""
        for source in ("https://queue.fal.run/", "http://queue.fal.run/"):
            if url.startswith(source):
                return build_url(config.get("base_url"), url[len(source):])
        return url

    def fal_queue_url_candidates(self, config, endpoint_id, request_id, suffix, returned_url=None):
        candidates = []

        def add(candidate):
            candidate = str(candidate or "").strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        if returned_url:
            add(self.normalize_fal_proxy_url(config, returned_url))
            add(returned_url)
        base_url = self.fal_endpoint(config, endpoint_id)
        add(build_url(base_url, f"requests/{request_id}/{suffix}".rstrip("/")))
        return candidates

    def fal_root_url(self, config):
        base_url = str((config or {}).get("base_url") or "").strip().rstrip("/")
        if base_url.endswith("/fal"):
            return base_url[:-4]
        return base_url

    def fal_task_record_url_candidates(self, config, request_id):
        root_url = self.fal_root_url(config)
        request_id = str(request_id or "").strip()
        if not root_url or not request_id:
            return []
        encoded_request_id = quote(request_id, safe="")
        return [
            build_url(root_url, f"/api/task/self?page=1&page_size=10&task_id={encoded_request_id}"),
        ]

    def extract_fal_task_record(self, payload, request_id):
        request_id = str(request_id or "").strip()
        if not request_id:
            return payload if isinstance(payload, dict) else None
        if isinstance(payload, dict):
            if str(payload.get("task_id") or payload.get("request_id") or "").strip() == request_id:
                return payload
            data = payload.get("data")
            if isinstance(data, dict):
                items = data.get("items")
                if isinstance(items, list):
                    for item in items:
                        found = self.extract_fal_task_record(item, request_id)
                        if found:
                            return found
                found = self.extract_fal_task_record(data, request_id)
                if found:
                    return found
            for value in payload.values():
                if value is data:
                    continue
                found = self.extract_fal_task_record(value, request_id)
                if found:
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = self.extract_fal_task_record(item, request_id)
                if found:
                    return found
        return None

    async def query_fal_task_record_once(self, config, request_id):
        urls = self.fal_task_record_url_candidates(config, request_id)
        if not urls:
            return None
        payload = await self.request_first_success_json(
            "GET",
            urls,
            config,
            request_id=request_id,
        )
        return self.extract_fal_task_record(payload, request_id) or payload

    def fal_poll_headers_candidates(self, config):
        return [self.headers(config, {"Content-Type": "application/json"})]

    async def request_first_success_json(self, method, urls, config, request_id=""):
        last_error = None
        failures = []
        for url in urls:
            for headers in self.fal_poll_headers_candidates(config):
                try:
                    return await self.async_request_json(method, url, config, headers=headers)
                except Exception as exc:
                    last_error = exc
                    failures.append({"method": method, "url": url, "error": str(exc)})
                    continue
        if last_error is not None:
            try:
                setattr(last_error, "lingliu_attempts", failures)
            except Exception:
                pass
            raise last_error
        raise RuntimeError("No URL candidates to request.")

    def fal_result_status(self, payload):
        status = self.extract_status(payload)
        normalized = str(status or "").strip().lower()
        if normalized in {"completed", "succeeded", "success", "done", "finish", "finished", "成功", "已完成"}:
            return "succeeded"
        if normalized in {"failed", "failure", "fail", "error", "cancelled", "canceled", "rejected", "失败", "已失败"}:
            return "failed"
        return normalized

    def filter_fal_media_urls(self, image_urls, video_urls):
        def is_queue_url(url):
            lowered = str(url or "").strip().lower()
            if not lowered:
                return True
            if lowered.startswith(("data:image/", "data:video/")):
                return True
            if any(lowered.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm")):
                return False
            if "fal.media/" in lowered or "storage.googleapis.com/" in lowered:
                return False
            return "/requests/" in lowered or lowered.endswith("/status") or lowered.endswith("/cancel")

        filtered_images = [url for url in image_urls if not is_queue_url(url)]
        filtered_videos = [url for url in video_urls if not is_queue_url(url)]
        return filtered_images, filtered_videos

    def extract_fal_output_media_urls(self, payload):
        image_urls = []
        video_urls = []

        def add_url(container, key, target):
            if not isinstance(container, dict):
                return
            value = container.get(key)
            if isinstance(value, dict):
                value = value.get("url")
            if isinstance(value, str) and self._looks_like_url(value):
                filtered_images, filtered_videos = self.filter_fal_media_urls([value], [])
                if target == "video":
                    _ignored_images, filtered_videos = self.filter_fal_media_urls([], [value])
                    for url in filtered_videos:
                        if url not in video_urls:
                            video_urls.append(url)
                else:
                    for url in filtered_images:
                        if url not in image_urls:
                            image_urls.append(url)

        def walk(value, parent_key=""):
            if isinstance(value, dict):
                normalized_parent = str(parent_key or "").strip().lower()
                if normalized_parent in {"input", "properties", "submit", "request", "payload"}:
                    return

                images = value.get("images")
                if isinstance(images, list):
                    for item in images:
                        add_url(item, "url", "image")
                add_url(value, "image", "image")
                add_url(value, "image_url", "image")

                videos = value.get("videos")
                if isinstance(videos, list):
                    for item in videos:
                        add_url(item, "url", "video")
                add_url(value, "video", "video")
                add_url(value, "video_url", "video")

                for key, item in value.items():
                    normalized_key = str(key or "").strip().lower()
                    if normalized_key in {
                        "image_url",
                        "image_urls",
                        "images",
                        "video",
                        "videos",
                        "video_url",
                        "input",
                        "properties",
                        "prompt",
                        "response_url",
                        "status_url",
                        "cancel_url",
                    }:
                        continue
                    walk(item, key)
            elif isinstance(value, list):
                for item in value:
                    walk(item, parent_key)

        walk(payload)
        if not image_urls and not video_urls:
            return self.filter_fal_media_urls(*self.extract_media_urls(payload))
        return image_urls, video_urls

    async def query_fal_result(
        self,
        config,
        endpoint_id,
        request_id,
        *,
        max_retries=360,
        poll_interval=2,
        status_url=None,
        result_url=None,
    ):
        status_urls = self.fal_queue_url_candidates(config, endpoint_id, request_id, "status", status_url)
        result_urls = self.fal_queue_url_candidates(config, endpoint_id, request_id, "", result_url)
        last_payload = None
        last_errors = []
        for _ in range(int(max_retries)):
            await asyncio.sleep(max(1, int(poll_interval)))
            try:
                record_payload = await self.query_fal_task_record_once(config, request_id)
            except Exception as exc:
                if self.is_transient_request_error(exc):
                    last_payload = {"warning": str(exc), "request_id": request_id}
                else:
                    attempts = getattr(exc, "lingliu_attempts", None)
                    if attempts:
                        last_errors.extend(attempts)
                    else:
                        last_errors.append(
                            {
                                "method": "GET",
                                "url": (self.fal_task_record_url_candidates(config, request_id) or [""])[0],
                                "error": str(exc),
                            }
                        )
            else:
                if record_payload is not None:
                    last_payload = record_payload
                    record_status = self.fal_result_status(record_payload)
                    image_urls, video_urls = self.extract_fal_output_media_urls(record_payload)
                    if record_status == "failed":
                        detail = self.extract_failure_detail(record_payload)
                        raise RuntimeError(f"fal task failed: {detail or record_payload}")
                    if image_urls or video_urls:
                        return record_payload, image_urls, video_urls

            status_payload = None
            try:
                status_payload = await self.request_first_success_json(
                    "GET",
                    status_urls,
                    config,
                    request_id=request_id,
                )
            except Exception as exc:
                if self.is_transient_request_error(exc):
                    last_payload = {"warning": str(exc), "request_id": request_id}
                    continue
                attempts = getattr(exc, "lingliu_attempts", None)
                if attempts:
                    last_errors.extend(attempts)
                else:
                    last_errors.append({"method": "GET", "url": status_urls[0] if status_urls else "", "error": str(exc)})

            if status_payload is not None:
                last_payload = status_payload
                status = self.fal_result_status(status_payload)
                image_urls, video_urls = self.extract_fal_output_media_urls(status_payload)
                if status == "succeeded":
                    try:
                        result_payload = await self.request_first_success_json(
                            "GET",
                            result_urls,
                            config,
                            request_id=request_id,
                        )
                    except Exception as exc:
                        attempts = getattr(exc, "lingliu_attempts", None)
                        if attempts:
                            last_errors.extend(attempts)
                        else:
                            last_errors.append(
                                {"method": "GET", "url": result_urls[0] if result_urls else "", "error": str(exc)}
                            )
                    else:
                        final_image_urls, final_video_urls = self.extract_fal_output_media_urls(result_payload)
                        return result_payload, final_image_urls or image_urls, final_video_urls or video_urls
                elif image_urls or video_urls:
                    return status_payload, image_urls, video_urls
                elif status == "failed":
                    detail = self.extract_failure_detail(status_payload)
                    raise RuntimeError(f"fal task failed: {detail or status_payload}")

            if last_errors and last_payload is None:
                error_preview = self.safe_json(last_errors[-8:])
                raise RuntimeError(f"fal query failed. Attempts: {error_preview}")

        error_suffix = f" Last query errors: {self.safe_json(last_errors[-8:])}" if last_errors else ""
        raise TimeoutError(f"fal task {request_id} timed out. Last payload: {last_payload}.{error_suffix}")

    async def run_fal_media_request(
        self,
        config,
        endpoint_id,
        payload,
        *,
        unique_id=None,
        label="Request ID",
        max_retries=5,
        initial_timeout=None,
        max_poll_attempts=360,
        poll_interval=2,
    ):
        submit_url = self.fal_endpoint(config, endpoint_id)
        submit_payload = await self.async_request_json_retrying(
            "POST",
            submit_url,
            config,
            max_retries=max_retries,
            initial_timeout=initial_timeout or self.timeout(config),
            headers=self.headers(config, {"Content-Type": "application/json"}),
            json=payload,
        )
        image_urls, video_urls = self.extract_fal_output_media_urls(submit_payload)
        request_id = self.extract_fal_request_id(submit_payload)
        status_url = self.extract_fal_url(submit_payload, {"status_url", "statusUrl"})
        result_url = self.extract_fal_url(submit_payload, {"response_url", "responseUrl", "result_url", "resultUrl"})
        if image_urls or video_urls:
            return {
                "request_id": request_id,
                "submit": submit_payload,
                "final": submit_payload,
                "image_urls": image_urls,
                "video_urls": video_urls,
                "fal_endpoint_id": endpoint_id,
            }
        if not request_id:
            raise RuntimeError(f"No request_id returned from fal endpoint {endpoint_id}: {submit_payload}")
        display_task_id = f"{endpoint_id}::{request_id}"
        self.publish_task_id(unique_id, display_task_id, label=label, append=False)
        self.publish_live_message(unique_id, f"{label}: {display_task_id}", append=False)
        final_payload, image_urls, video_urls = await self.query_fal_result(
            config,
            endpoint_id,
            request_id,
            max_retries=max_poll_attempts,
            poll_interval=poll_interval,
            status_url=status_url,
            result_url=result_url,
        )
        return {
            "request_id": request_id,
            "submit": submit_payload,
            "final": final_payload,
            "image_urls": image_urls,
            "video_urls": video_urls,
            "fal_endpoint_id": endpoint_id,
        }

    async def gather_json_requests(self, request_specs, return_exceptions=True):
        tasks = [
            self.async_request_json(method, url, config, **kwargs)
            for method, url, config, kwargs in request_specs
        ]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def async_request_json_retrying(
        self,
        method,
        url,
        config,
        *,
        max_retries=5,
        initial_timeout=None,
        retry_status_codes=(429, 500, 502, 503, 504, 520, 521, 522, 523, 524),
        **kwargs,
    ):
        last_error = None
        base_timeout = int(initial_timeout or self.timeout(config))
        for attempt in range(1, max_retries + 1):
            current_timeout = min(int(base_timeout * (1.5 ** (attempt - 1))), 1200)
            try:
                return await self.async_request_json_with_timeout(method, url, current_timeout, **kwargs)
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                last_error = exc
                if status_code not in retry_status_codes or attempt == max_retries:
                    raise
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code is not None:
                    if status_code not in retry_status_codes or attempt == max_retries:
                        raise
                elif not self.is_transient_request_error(exc) or attempt == max_retries:
                    raise
            await asyncio.sleep(min(2 ** (attempt - 1), 60))
        raise last_error or RuntimeError(f"Failed to request {url}")


class LingliuApiSet(LingliuBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        defaults = get_config()
        return {
            "required": {
                "api_provider": (PUBLIC_PROVIDER_DISPLAY_CHOICES, {"default": provider_display_name("lingliu")}),
                "api_key": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": defaults["timeout"], "min": 5, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_settings"
    CATEGORY = "Lingliu/Config"

    def save_settings(
        self,
        api_provider="lingliu",
        api_key="",
        timeout=300,
        **_legacy_inputs,
    ):
        api_provider = normalize_provider(api_provider)
        set_runtime_config(api_key=api_key, timeout=timeout, api_provider=api_provider)
        save_config({"api_provider": api_provider, "timeout": int(timeout)})
        return ("已保存全局API配置",)


class LingliuModels(LingliuBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("models", "raw_json")
    FUNCTION = "list_models"

    async def list_models(self, refresh=True, api_key=""):
        del refresh
        try:
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                return "", self.error_json("API key is required for lingliu_models.")
            url = self.endpoint(config, "models_path")
            payload = await self.async_request_json("GET", url, config, headers=self.headers(config))
            items = payload.get("data", []) if isinstance(payload, dict) else []
            model_ids = []
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    model_ids.append(item["id"])
            return "\n".join(model_ids), self.safe_json(payload)
        except Exception as exc:
            return "", self.error_json(exc)


# Chat models the platform serves for text only: an image input is refused
# upstream (or silently ignored), so it is rejected before any request.
TEXT_ONLY_CHAT_MODELS = frozenset({"gpt-auto", "glm-5.3"})


class LingliuChat(LingliuBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-4.1-mini", "tooltip": "gpt-auto、glm-5.3 只能处理文字，不能接 image。"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "image": ("IMAGE",),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "raw_json")
    FUNCTION = "chat"

    async def chat(
        self,
        prompt,
        model,
        system_prompt="",
        api_key="",
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        seed=-1,
        image=None,
        image_detail="auto",
    ):
        try:
            if image is not None and str(model or "").strip().lower() in TEXT_ONLY_CHAT_MODELS:
                error = f"{str(model).strip()} 只能处理文字，请断开 image 输入后再运行（未提交、未扣费）。"
                return f"Error: {error}", self.error_json(error)
            config = self.load_chat_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_chat."
                return error, self.error_json(error)
            url = self.endpoint(config, "chat_path")

            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})

            if image is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_to_data_url(image),
                                    "detail": image_detail,
                                },
                            },
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
            }
            if int(seed) >= 0:
                payload["seed"] = int(seed)

            response_payload = await self.async_request_json(
                "POST",
                url,
                config,
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )

            choices = response_payload.get("choices", []) if isinstance(response_payload, dict) else []
            if not choices:
                raise RuntimeError(f"No choices returned from chat endpoint: {url}")
            message = choices[0].get("message", {})
            response_text = content_to_text(message.get("content"))
            if not response_text and choices[0].get("text"):
                response_text = str(choices[0]["text"])
            return response_text, self.safe_json(response_payload)
        except Exception as exc:
            return f"Error: {exc}", self.error_json(exc)


class LingliuImageGenerate(LingliuBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "gpt-image-1"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1024x1024"}),
                "quality": ("STRING", {"default": "auto"}),
                "background": ("STRING", {"default": "auto"}),
                "moderation": ("STRING", {"default": "auto"}),
                "output_format": ("STRING", {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "raw_json")
    FUNCTION = "generate"

    def _extract_images_from_payloads(self, payloads, timeout):
        all_images = []
        all_urls = []
        normalized_payloads = []

        for payload in payloads:
            normalized_payloads.append(payload)
            images, urls = parse_image_response(payload, timeout=timeout)
            all_images.extend(images)
            all_urls.extend(urls)

        return all_images, all_urls, normalized_payloads

    async def _wait_for_image_task(self, config, task_id, max_retries=360, poll_interval=2):
        task_url = build_url(
            config.get("base_url"),
            (config.get("images_task_status_path_template") or "").format(task_id=task_id),
        )
        last_payload = None
        consecutive_poll_errors = 0
        for _ in range(int(max_retries)):
            await asyncio.sleep(max(1, int(poll_interval)))
            try:
                payload = await self.async_request_json("GET", task_url, config, headers=self.headers(config))
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

            last_payload = payload
            outer = payload.get("data", {}) if isinstance(payload, dict) else {}
            status = ""
            if isinstance(outer, dict):
                status = str(outer.get("status", "") or outer.get("state", "") or "")
            status = status or self.extract_status(payload)
            if self.is_success_status(status):
                inner = outer.get("data", {}) if isinstance(outer, dict) else {}
                if isinstance(inner, dict):
                    return inner.get("data", []), payload
                if isinstance(inner, list):
                    return inner, payload
                image_urls, _video_urls = self.extract_media_urls(payload)
                if image_urls:
                    return [{"url": url} for url in image_urls], payload
                raise RuntimeError(f"Task {task_id} succeeded but no images were returned. Payload: {payload}")
            if self.is_failure_status(status):
                failure_context = outer if isinstance(outer, dict) and outer else payload
                fail_reason = self.extract_failure_detail(failure_context) or "Unknown error"
                raise RuntimeError(f"Task {task_id} failed: {fail_reason}")

        raise TimeoutError(f"Task {task_id} timed out. Last payload: {last_payload}")

    async def _request_single_generate(self, config, url, base_payload, request_seed, unique_id=None, request_index=0):

        payload = dict(base_payload)
        if int(request_seed) > 0:
            payload["seed"] = int(request_seed)

        response_payload = await self.async_request_json_retrying(
            "POST",
            url,
            config,
            max_retries=5,
            initial_timeout=self.timeout(config),
            headers=self.headers(config, {"Content-Type": "application/json"}),
            json=payload,
        )

        images, urls = await asyncio.to_thread(parse_image_response, response_payload, self.timeout(config))
        if images:
            return {
                "images": images,
                "urls": urls,
                "response": response_payload,
            }

        task_id = response_payload.get("task_id") or response_payload.get("data")
        if isinstance(task_id, str) and task_id.strip():
            self.publish_task_id(unique_id, task_id.strip(), label=f"Task ID #{request_index + 1}")
            image_items, final_payload = await self._wait_for_image_task(config, task_id.strip())
            final_images, final_urls = await asyncio.to_thread(
                parse_image_response, {"data": image_items}, self.timeout(config)
            )
            if final_images:
                return {
                    "images": final_images,
                    "urls": final_urls,
                    "task_id": task_id.strip(),
                    "submit": response_payload,
                    "final": final_payload,
                }

        raise RuntimeError(f"No images returned from generate endpoint: {url}")

    async def generate(
        self,
        prompt,
        model,
        api_key="",
        size="1024x1024",
        quality="auto",
        background="auto",
        moderation="auto",
        output_format="png",
        num_images=1,
        seed=0,
        unique_id=None,
        ):
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_image_generate."
                return blank_image_tensor(), "", self.error_json(error)
            url = self.endpoint(config, "images_generate_path")
            base_payload = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "background": background,
                "moderation": moderation,
                "output_format": output_format,
                "n": 1,
                "num_images": 1,
            }
            if int(seed) > 0:
                base_payload["seed"] = int(seed)

            request_count = max(1, int(num_images))
            request_plan = []
            for index in range(request_count):
                request_seed = int(seed) + index if int(seed) > 0 else 0
                request_plan.append({"index": index, "seed": request_seed})

            response_payloads = await asyncio.gather(
                *[
                    self._request_single_generate(
                        config,
                        url,
                        base_payload,
                        plan["seed"],
                        unique_id=unique_id,
                        request_index=plan["index"],
                    )
                    for plan in request_plan
                ],
                return_exceptions=True,
            )

            successful_payloads = []
            errors = []
            for index, item in enumerate(response_payloads):
                if isinstance(item, Exception):
                    errors.append({"index": index, "error": str(item)})
                else:
                    successful_payloads.append(item)

            if errors:
                recovered = []
                remaining_errors = []
                for error_item in errors:
                    request_index = int(error_item["index"])
                    plan = request_plan[request_index]
                    try:
                        recovered_item = await self._request_single_generate(
                            config,
                            url,
                            base_payload,
                            plan["seed"],
                            unique_id=unique_id,
                            request_index=plan["index"],
                        )
                        recovered.append(recovered_item)
                    except Exception as exc:
                        remaining_errors.append({"index": request_index, "error": str(exc)})
                successful_payloads.extend(recovered)
                errors = remaining_errors

            image_payloads = []
            serializable_payloads = []
            for payload in successful_payloads:
                if "response" in payload:
                    image_payloads.append(payload["response"])
                    serializable_payloads.append(
                        {
                            "urls": list(payload.get("urls", []) or []),
                            "image_count": len(payload.get("images", []) or []),
                            "response": payload.get("response"),
                        }
                    )
                else:
                    final_payload = payload.get("final") or {}
                    image_items = (
                        final_payload.get("data", {}).get("data", {}).get("data", [])
                        if isinstance(final_payload, dict)
                        else []
                    )
                    image_payloads.append({"data": image_items})
                    serializable_payloads.append(
                        {
                            "task_id": payload.get("task_id", ""),
                            "submit": payload.get("submit"),
                            "final": payload.get("final"),
                            "urls": list(payload.get("urls", []) or []),
                            "image_count": len(payload.get("images", []) or []),
                        }
                    )

            images, urls, _normalized_payloads = await asyncio.to_thread(
                self._extract_images_from_payloads, image_payloads, self.timeout(config)
            )
            if not images:
                raise RuntimeError(
                    json.dumps(
                        {
                            "requested_num_images": request_count,
                            "successful_requests": 0,
                            "failed_requests": len(errors),
                            "errors": errors,
                        },
                        ensure_ascii=False,
                    )
                )

            raw_payload = {
                "requested_num_images": request_count,
                "successful_requests": len(successful_payloads),
                "failed_requests": len(errors),
                "errors": errors,
                "responses": serializable_payloads,
            }
            return (
                pil2tensor(images),
                "\n".join(urls),
                self.safe_json(raw_payload),
            )
        except Exception as exc:
            return blank_image_tensor(), "", self.error_json(exc)


class LingliuImageEdit(LingliuBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "gpt-image-1"}),
            },
            "optional": {
                "mask": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1024x1024"}),
                "quality": ("STRING", {"default": "auto"}),
                "background": ("STRING", {"default": "auto"}),
                "output_format": ("STRING", {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "raw_json")
    FUNCTION = "edit"

    async def edit(
        self,
        prompt,
        image,
        model,
        mask=None,
        api_key="",
        size="1024x1024",
        quality="auto",
        background="auto",
        output_format="png",
        seed=0,
        unique_id=None,
    ):
        try:
            self.publish_task_id(unique_id, clear=True)
            config = self.load_config(api_key_override=api_key)
            if not (config.get("api_key") or "").strip():
                error = "API key is required for lingliu_image_edit."
                return blank_image_tensor(), "", self.error_json(error)
            url = self.endpoint(config, "images_edit_path")


            files = [
                ("image", ("image.png", image_to_png_bytes(image), "image/png")),
            ]
            if mask is not None:
                files.append(("mask", ("mask.png", image_to_png_bytes(mask), "image/png")))

            data = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "background": background,
                "output_format": output_format,
                "n": "1",
            }
            if int(seed) > 0:
                data["seed"] = str(seed)
            response_payload = await self.async_request_json(
                "POST",
                url,
                config,
                headers=self.headers(config),
                data=data,
                files=files,
            )
            images, urls = await asyncio.to_thread(parse_image_response, response_payload, self.timeout(config))
            if not images:
                raise RuntimeError(f"No images returned from edit endpoint: {url}")
            return (
                pil2tensor(images),
                "\n".join(urls),
                self.safe_json(response_payload),
            )
        except Exception as exc:
            return blank_image_tensor(), "", self.error_json(exc)
