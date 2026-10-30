"""Video-to-depth node: one input video in, one depth video out.

The platform bills by measured input seconds and reserves the charge before
it creates the upstream task, so the create request is sent exactly once and
every rule the platform checks up front is checked locally first. Strict single
submit, polling and the authenticated download reuse MiniMaxH3Workflow.
"""

import json
import re
from io import BytesIO
from urllib.parse import quote, urlsplit

from comfy.comfy_types import IO

from ..upstream_mask import mask_upstream
from ..lingliu_config import (
    LINGLIU_PROVIDER,
    build_url,
    get_config,
    mask_provider_payload,
    mask_provider_text,
    normalize_provider,
)
from ..utils import LingliuNullVideoAdapter
from .nodes_jl_video import QUERY_HINT, URL_PATTERN, _origin, _upstream_message
from .nodes_minimax_h3 import (
    H3Error,
    H3Result,
    MiniMaxH3Workflow,
    MTH3VideoAdapter,
    _extract_task_id,
    _read_media_input,
    _redact,
    _redact_text,
)
from .nodes_openai_compatible import LingliuBaseNode

NODE_ID = "lingliu_depth_video"
MODEL = "灵流-视频转深度视频"
ALLOWED_PROVIDERS = {LINGLIU_PROVIDER}
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_IDEMPOTENCY_KEY = 256
# The platform stores the whole upload and probes it before answering.
MIN_CREATE_TIMEOUT = 600

PENDING_STATUSES = {"", "queued", "in_progress", "pending", "submitted", "processing", "running"}
UNKNOWN_STATUSES = {"unknown"}
FAILED_STATUSES = {"failed", "failure", "error", "cancelled", "canceled"}
SUCCESS_STATUSES = {"completed", "succeeded", "success"}
VIDEO_RULE = "视频需为 MP4（H.264/H.265）、单个视频轨、每边不超过 4096 像素、时长不超过 60 秒、不超过 100MB"
UNCERTAIN_HINT = "提交结果不确定，请先到任务日志确认，不要重复提交"
UNKNOWN_TIMEOUT_MESSAGE = (
    "轮询超时：平台正在核对上游创建结果，已保留本任务和预扣费，不会自动重复创建；"
    "请稍后用任务 ID 查询，不要重复提交。"
)
TIMEOUT_MESSAGE = f"轮询超时。{QUERY_HINT}。"
STATUS_HINTS = {
    403: "余额或令牌额度不足，或当前计费方式不支持此模型（只支持钱包计费，请把计费方式改为钱包优先或仅钱包）；未扣费",
    409: "防重复提交键已用于另一个视频，请换一个键或留空；未扣费",
    429: "线路同一时间只处理一个任务，当前正忙，请稍后再试；未扣费",
}
CODE_HINTS = {
    "invalid_video": f"{VIDEO_RULE}；未扣费",
    "invalid_request": "请求格式不符合平台要求；未扣费",
    "price_not_configured": "平台尚未配置此模型价格，请联系管理员；未扣费",
}
URL_ERROR = "视频URL 需为 http/https 公网地址，不能带账号密码或 # 片段，端口只能是 80/443"
KEY_PATTERN = re.compile(r"^[\x21-\x7e]+$")


def _hide_upstream_name(text):
    return mask_upstream(str(text or ""))


def _error_code(payload):
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    code = error.get("code") if isinstance(error, dict) else payload.get("code")
    return str(code or "").strip().lower()


def validate_video_url(url):
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise H3Error(URL_ERROR) from exc
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise H3Error(URL_ERROR)
    if parts.username is not None or parts.password is not None or "#" in url:
        raise H3Error(URL_ERROR)
    if port not in (None, 80, 443):
        raise H3Error(URL_ERROR)
    return url


def validate_idempotency_key(key):
    key = str(key or "").strip()
    if key and (len(key) > MAX_IDEMPOTENCY_KEY or not KEY_PATTERN.match(key)):
        raise H3Error(f"防重复提交键只能用英文字母、数字和符号（不含空格），最多 {MAX_IDEMPOTENCY_KEY} 个字符")
    return key


def read_video_bytes(video):
    """Prefer save_to: it honours trims, and the platform bills the uploaded seconds."""
    if video is None or isinstance(video, LingliuNullVideoAdapter):
        raise H3Error("输入视频为空（前面的节点可能失败了）")
    try:
        save_to = getattr(video, "save_to", None)
        if callable(save_to):
            buffer = BytesIO()
            save_to(buffer)
            data = buffer.getvalue()
        else:
            data, _name = _read_media_input(video, ".mp4")
    except H3Error:
        raise
    except Exception as exc:
        raise H3Error(f"读取输入视频失败：{exc}") from exc
    if not data:
        raise H3Error("输入视频为空（前面的节点可能失败了）")
    if len(data) > MAX_VIDEO_BYTES:
        raise H3Error(f"输入视频超过 100MB；{VIDEO_RULE}")
    if len(data) < 16 or data[4:8] != b"ftyp":
        raise H3Error(f"只支持 MP4（H.264/H.265）视频，请先转成 MP4；{VIDEO_RULE}")
    return bytes(data)


def resolve_depth_config(api_key_override=""):
    config = get_config()
    provider = normalize_provider(config.get("api_provider"))
    if provider not in ALLOWED_PROVIDERS:
        raise H3Error("视频转深度视频只支持灵流线路，请在设置里切换到灵流线路", stage="route")
    raw_key = str(api_key_override or config.get("api_key") or "").strip()
    if not raw_key:
        raise H3Error("缺少 API 密钥，请在设置里填写或在节点里输入", stage="auth")
    return {
        # Selects the strict single-submit transport: create once, 2xx only, no redirects.
        "channel": "MT",
        "provider": provider,
        "base_url": str(config.get("base_url") or "").rstrip("/"),
        "timeout": int(config.get("timeout") or 300),
        "raw_key": raw_key,
        "files_key": raw_key,
        "h3_key": raw_key,
    }


class DepthVideoWorkflow(MiniMaxH3Workflow):
    def __init__(self, http=None, sleep=None):
        # The platform probes the upload itself; no local media probe is needed.
        super().__init__(http=http, media_probe=object(), sleep=sleep)
        self._origin = ""
        self._visible_urls = set()
        self._last_status = None

    def _hide_urls(self, value):
        """Keep platform URLs and the user's own URL; hide upstream result URLs."""
        if isinstance(value, dict):
            return {key: self._hide_urls(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._hide_urls(item) for item in value]
        if not isinstance(value, str):
            return value

        def replace(match):
            url = match.group(0)
            if url in self._visible_urls or (self._origin and _origin(url) == self._origin):
                return url
            return "<hidden_upstream_url>"

        return URL_PATTERN.sub(replace, value)

    def _mask_text(self, text, secrets):
        return _hide_upstream_name(mask_provider_text(_redact_text(self._hide_urls(str(text or "")), secrets)))

    def _json(self, payload, secrets):
        masked = mask_provider_payload(_redact(self._hide_urls(payload), secrets))
        return _hide_upstream_name(json.dumps(masked, ensure_ascii=False, indent=2))

    def _notify_status(self, notify, task_id, status):
        if notify and status != self._last_status:
            self._last_status = status
            notify(task_id, "核对中（不要重复提交）" if status in UNKNOWN_STATUSES else status)

    async def execute(self, *, video=None, video_url="", api_key="", wait_for_completion=True,
                      max_poll_attempts=180, poll_interval=10, idempotency_key="", notify=None):
        secrets = {str(api_key or "").strip()}
        envelope = {"status": "error", "model": MODEL}
        task_id = ""
        try:
            max_poll_attempts = int(max_poll_attempts)
            poll_interval = int(poll_interval)
            if not 1 <= max_poll_attempts <= 720 or not 2 <= poll_interval <= 60:
                raise H3Error("最大轮询次数需在 1–720，轮询间隔需在 2–60 秒")
            idempotency_key = validate_idempotency_key(idempotency_key)
            video_url = str(video_url or "").strip()
            has_video = video is not None and not isinstance(video, LingliuNullVideoAdapter)
            if video_url and video is not None:
                raise H3Error("视频和视频URL 只能二选一（未扣费）")
            if not video_url and not has_video:
                if video is not None:
                    raise H3Error("输入视频为空（前面的节点可能失败了）")
                raise H3Error("请连接视频，或填写视频URL（未扣费）")
            if video_url:
                validate_video_url(video_url)
                self._visible_urls.add(video_url)
            config = resolve_depth_config(api_key)
            secrets.update({config["raw_key"], config["h3_key"]})
            self._origin = _origin(config["base_url"])
            create_config = {**config, "timeout": max(int(config["timeout"]), MIN_CREATE_TIMEOUT)}
            headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
            if video_url:
                request = {"json": {"model": MODEL, "video_url": video_url},
                           "extra_headers": {**headers, "Content-Type": "application/json"}}
            else:
                data = read_video_bytes(video)
                envelope["input_bytes"] = len(data)
                request = {"files": {"input_reference": ("input.mp4", data, "video/mp4")},
                           "data": {"model": MODEL}, "extra_headers": headers}
            create_response = await self._request_json(
                "POST", build_url(config["base_url"], "/v1/videos"), create_config,
                stage="create", key=config["h3_key"], retry_network=False, retry_status={429}, **request,
            )
            envelope["create_response"] = create_response
            task_id = _extract_task_id(create_response)
            if not task_id:
                raise H3Error(f"创建响应缺少任务 ID；{UNCERTAIN_HINT}", stage="create", payload=create_response)
            envelope["task_id"] = task_id
            query_url = build_url(config["base_url"], f"/v1/videos/{quote(task_id, safe='')}")
            # A replayed idempotency key can return a task that already finished or failed.
            result = self._outcome(create_response, envelope, task_id, query_url, config, secrets, notify)
            if result:
                return result
            if not wait_for_completion:
                envelope["message"] = "任务已提交，未等待完成。"
                return H3Result(LingliuNullVideoAdapter("任务已提交，未等待完成。"), task_id, self._json(envelope, secrets), "")
            for _attempt in range(max_poll_attempts):
                await self.sleep(poll_interval)
                query_response = await self._request_json(
                    "GET", query_url, config, stage="query", key=config["h3_key"], retry_network=True, retry_status={429},
                )
                envelope["query_response"] = query_response
                result = self._outcome(query_response, envelope, task_id, query_url, config, secrets, notify)
                if result:
                    return result
            message = UNKNOWN_TIMEOUT_MESSAGE if envelope.get("status") in UNKNOWN_STATUSES else TIMEOUT_MESSAGE
            envelope.update({"status": "timeout", "message": message})
            if notify:
                notify(task_id, "timeout")
            return H3Result(LingliuNullVideoAdapter(message), task_id, self._json(envelope, secrets), "")
        except Exception as exc:
            return self._failure(exc, envelope, task_id, secrets, notify)

    def _outcome(self, payload, envelope, task_id, query_url, config, secrets, notify):
        status = self._status(payload)
        envelope["status"] = status or "pending"
        metadata = self._task_payload(payload).get("metadata")
        if isinstance(metadata, dict):
            if isinstance(metadata.get("billing"), dict):
                envelope["billing"] = metadata["billing"]
            notice = str(metadata.get("notice") or "").strip()
            if notice:
                envelope["notice"] = self._mask_text(notice[:500], secrets)
        self._notify_status(notify, task_id, status or "pending")
        if status in FAILED_STATUSES:
            reason = _upstream_message(payload) or status
            message = self._mask_text(f"生成失败：{reason}", secrets)
            envelope["error"] = {"stage": "task", "message": message}
            return H3Result(LingliuNullVideoAdapter(message), task_id, self._json(envelope, secrets), "")
        if status in SUCCESS_STATUSES:
            # Download through the platform's authenticated content endpoint.
            video_url = f"{query_url}/content"
            envelope["video_url"] = video_url
            return H3Result(MTH3VideoAdapter(video_url, config, self.http), task_id, self._json(envelope, secrets), video_url)
        if status not in PENDING_STATUSES and status not in UNKNOWN_STATUSES:
            envelope["note"] = f"未识别的任务状态 {status}，继续查询"
        return None

    def _failure(self, exc, envelope, task_id, secrets, notify):
        stage = getattr(exc, "stage", "internal")
        payload = getattr(exc, "payload", None)
        status_code = getattr(exc, "status_code", None)
        message = str(exc).rstrip("。")
        upstream = _upstream_message(payload)
        if upstream and upstream not in message:
            message = f"{message}：{upstream}"
        if stage == "create" and "不确定" not in message:
            code = int(status_code or 0)
            hint = CODE_HINTS.get(_error_code(payload)) or STATUS_HINTS.get(code)
            if not hint and (code >= 500 or 200 <= code < 300):
                hint = UNCERTAIN_HINT
            if hint:
                message += f"；{hint}"
        if stage == "query" and task_id:
            message += f"；{QUERY_HINT}"
        message = self._mask_text(message, secrets)
        error_payload = {
            **envelope,
            "status": "error",
            "task_id": task_id,
            "error": {"stage": stage, "message": message, "status_code": status_code, "upstream": payload},
        }
        if notify and task_id:
            notify(task_id, "error")
        return H3Result(LingliuNullVideoAdapter(message), task_id, self._json(error_payload, secrets), "")


class LingliuDepthVideo(LingliuBaseNode):
    CATEGORY = "灵流API/工具/视频"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL")
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "视频": (IO.VIDEO, {"tooltip": f"{VIDEO_RULE}。按输入视频实际秒数计费，失败退回。"}),
                "视频URL": ("STRING", {"default": "", "tooltip": "也可填公网 MP4 地址；和视频二选一。"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 180, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60}),
                "防重复提交键": ("STRING", {
                    "default": "",
                    "tooltip": "留空=每次新建任务。填了以后，同一个键+同一个视频会直接返回上次的任务（包括失败的），不再扣费；换视频必须换键。",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _notify(self, unique_id):
        def notify(task_id, status):
            if task_id:
                self.publish_task_id(unique_id, task_id, append=False)
            self.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status}", append=False)

        return notify

    async def convert(self, **kwargs):
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)
        result = await DepthVideoWorkflow().execute(
            video=kwargs.get("视频"),
            video_url=kwargs.get("视频URL", ""),
            api_key=kwargs.get("API密钥", ""),
            wait_for_completion=kwargs.get("等待完成", True),
            max_poll_attempts=kwargs.get("最大轮询次数", 180),
            poll_interval=kwargs.get("轮询间隔", 10),
            idempotency_key=kwargs.get("防重复提交键", ""),
            notify=self._notify(unique_id),
        )
        return result.video, result.task_id, result.response, result.video_url
