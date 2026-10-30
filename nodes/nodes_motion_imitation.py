"""Motion imitation node (V17.7, console 1.6.15): a person image follows a video's motion.

Billing is per measured input second and the charge is reserved before the
upstream task exists, so the create request goes out exactly once (MT strict
transport).  The duration is probed locally first, the same way the platform
measures it (first packet pts to last pts + duration of the one video track),
so the estimate is visible and a long video can be stopped before any charge.
Polling and the authenticated download reuse DepthVideoWorkflow.
"""

import math
from io import BytesIO
from urllib.parse import quote

from comfy.comfy_types import IO

from ..lingliu_config import build_url, get_config, normalize_provider
from ..utils import LingliuNullVideoAdapter, tensor2pil
from .lingliu_reference_upload import log
from .nodes_depth_video import (
    ALLOWED_PROVIDERS,
    TIMEOUT_MESSAGE,
    UNCERTAIN_HINT,
    UNKNOWN_STATUSES,
    UNKNOWN_TIMEOUT_MESSAGE,
    DepthVideoWorkflow,
    LingliuDepthVideo,
    _error_code,
    validate_idempotency_key,
    validate_video_url,
)
from .nodes_jl_video import QUERY_HINT, _origin, _upstream_message
from .nodes_minimax_h3 import H3Error, H3Result, _extract_task_id, _read_media_input

NODE_ID = "lingliu_motion_imitation"
MODEL = "灵流-动作模仿"
PRICE_PER_SECOND = 0.2
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_EDGE = 4096
JPEG_FALLBACK_QUALITY = 95
# The platform stores both uploads and probes the video before answering.
MIN_CREATE_TIMEOUT = 600
# Measured 10-27: about 80 seconds of processing per input second.
WAIT_SECONDS_PER_INPUT_SECOND = 100
MIN_WAIT_MINUTES, MAX_WAIT_MINUTES = 15, 360
VIDEO_RULE = "动作视频需为 MP4（H.264/H.265）、单个视频轨、每边不超过 4096 像素、不超过 100MB，时长不限"
IMAGE_RULE = "人物图需为 PNG/JPEG/WebP、不超过 30MB"
PRICE_TIP = "按动作视频实际秒数计费 0.2✦/秒（再乘分组倍率），提交时预扣，失败全额退；提交后不能取消，处理约为视频时长的 80 倍。"
STATUS_HINTS = {
    403: "余额或令牌额度不足，或当前计费方式不支持此模型（只支持钱包计费，请把计费方式改为钱包优先或仅钱包）；未扣费",
    409: "防重复提交键已用于另一组视频和人物图，请换一个键或留空；未扣费",
    429: "任务排队中，请稍后再试；未扣费",
}
CODE_HINTS = {
    "invalid_video": f"{VIDEO_RULE}；未扣费",
    "invalid_request": f"请求未被接受（{VIDEO_RULE}；{IMAGE_RULE}）；未扣费",
    "price_not_configured": "平台尚未配置此模型价格，请联系管理员；未扣费",
}


def probe_seconds(data):
    """Billable seconds as the platform measures them, or None when the video cannot be probed.

    Raises H3Error for a video the platform would refuse anyway (codec, size, tracks)."""
    try:
        import av
    except Exception:
        return None
    try:
        with av.open(BytesIO(data), mode="r", format="mp4") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise H3Error(f"动作视频必须只有一个视频轨（当前 {len(streams)} 个）；{VIDEO_RULE}；未扣费")
            stream = streams[0]
            codec = str(stream.codec_context.name or "").lower()
            width, height = int(stream.codec_context.width or 0), int(stream.codec_context.height or 0)
            if codec not in ("h264", "hevc"):
                raise H3Error(f"动作视频编码为 {codec or '未知'}，只支持 H.264/H.265；未扣费")
            if not 0 < width <= MAX_EDGE or not 0 < height <= MAX_EDGE:
                raise H3Error(f"动作视频分辨率 {width}×{height} 超过每边 4096 像素；未扣费")
            base = stream.time_base
            start, end = math.inf, -math.inf
            for packet in container.demux(stream):
                if packet.pts is None or not packet.duration:
                    continue
                pts = float(packet.pts * base)
                start = min(start, pts)
                end = max(end, pts + float(packet.duration * base))
    except H3Error:
        raise
    except Exception:
        return None
    if not math.isfinite(start) or end <= start:
        return None
    return round(end - start, 3)


def read_motion_video(video):
    """Prefer save_to: it honours trims, and the platform bills the uploaded seconds."""
    if video is None or isinstance(video, LingliuNullVideoAdapter):
        raise H3Error("动作视频为空（前面的节点可能失败了）")
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
        raise H3Error(f"读取动作视频失败：{exc}") from exc
    if not data:
        raise H3Error("动作视频为空（前面的节点可能失败了）")
    if len(data) > MAX_VIDEO_BYTES:
        raise H3Error(f"动作视频超过 100MB；{VIDEO_RULE}；未扣费")
    if len(data) < 16 or data[4:8] != b"ftyp":
        raise H3Error(f"动作视频只支持 MP4，请先转成 MP4；{VIDEO_RULE}；未扣费")
    return bytes(data)


def encode_person(image):
    """Return (bytes, filename, mime, note).  PNG first; JPEG 95 only when PNG is over 30 MiB."""
    if image is None:
        raise H3Error("请连接人物图，或填写人物图URL（未扣费）")
    frames = tensor2pil(image)
    if not frames:
        raise H3Error("人物图为空（前面的节点可能失败了，未扣费）")
    note = f"人物图输入了 {len(frames)} 张，只用第 1 张。" if len(frames) > 1 else ""
    buffer = BytesIO()
    frames[0].save(buffer, format="PNG")
    if buffer.tell() <= MAX_IMAGE_BYTES:
        return buffer.getvalue(), "person.png", "image/png", note
    buffer = BytesIO()
    frames[0].convert("RGB").save(buffer, format="JPEG", quality=JPEG_FALLBACK_QUALITY)
    if buffer.tell() > MAX_IMAGE_BYTES:
        raise H3Error("人物图太大：转成 JPEG 后仍超过 30MB，请先缩小（未扣费）")
    return buffer.getvalue(), "person.jpg", "image/jpeg", (note + "人物图 PNG 超过 30MB，已改用 JPEG 95。").strip()


def resolve_motion_config(api_key_override=""):
    config = get_config()
    provider = normalize_provider(config.get("api_provider"))
    if provider not in ALLOWED_PROVIDERS:
        raise H3Error("动作模仿只支持灵流线路，请在设置里切换到灵流线路", stage="route")
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


def wait_budget(seconds, max_wait_minutes, poll_interval):
    """(minutes, polls).  0 minutes = auto: about 100 s of waiting per input second, 15–360 minutes."""
    minutes = int(max_wait_minutes)
    if minutes <= 0:
        if seconds:
            minutes = math.ceil(seconds * WAIT_SECONDS_PER_INPUT_SECOND / 60)
            minutes = min(max(minutes, MIN_WAIT_MINUTES), MAX_WAIT_MINUTES)
        else:
            minutes = MAX_WAIT_MINUTES
    return minutes, max(1, math.ceil(minutes * 60 / int(poll_interval)))


def estimate_text(seconds):
    if not seconds:
        return "未能在本地读出动作视频时长，按平台实际计量扣费（0.2✦/秒）"
    return f"输入 {seconds:.1f} 秒，预计约 {seconds * PRICE_PER_SECOND:.2f}✦（按 0.2✦/秒，再乘分组倍率，以平台实际计量为准）"


class MotionImitationWorkflow(DepthVideoWorkflow):
    async def execute(self, *, video=None, video_url="", person=None, person_url="", api_key="",
                      wait_for_completion=True, poll_interval=20, max_wait_minutes=0, max_billed_seconds=60,
                      idempotency_key="", notify=None):
        secrets = {str(api_key or "").strip()}
        envelope = {"status": "error", "model": MODEL}
        task_id = ""
        try:
            poll_interval, max_wait_minutes = int(poll_interval), int(max_wait_minutes)
            max_billed_seconds = int(max_billed_seconds)
            if not 5 <= poll_interval <= 120 or not 0 <= max_wait_minutes <= 720 or not 0 <= max_billed_seconds <= 3600:
                raise H3Error("轮询间隔需在 5–120 秒，最长等待分钟需在 0–720，最多计费秒数需在 0–3600")
            idempotency_key = validate_idempotency_key(idempotency_key)
            video_url, person_url = str(video_url or "").strip(), str(person_url or "").strip()
            if video_url and video is not None:
                raise H3Error("动作视频和视频URL 只能二选一（未扣费）")
            if not video_url and video is None:
                raise H3Error("请连接动作视频，或填写视频URL（未扣费）")
            if person_url and person is not None:
                raise H3Error("人物图和人物图URL 只能二选一（未扣费）")
            if not person_url and person is None:
                raise H3Error("请连接人物图，或填写人物图URL（未扣费）")
            if video_url:
                validate_video_url(video_url)
                self._visible_urls.add(video_url)
            if person_url:
                try:
                    validate_video_url(person_url)
                except H3Error as exc:
                    raise H3Error(str(exc).replace("视频URL", "人物图URL")) from None
                self._visible_urls.add(person_url)
            config = resolve_motion_config(api_key)
            secrets.update({config["raw_key"], config["h3_key"]})
            self._origin = _origin(config["base_url"])
            seconds, data = None, None
            if not video_url:
                data = read_motion_video(video)
                envelope["input_bytes"] = len(data)
                seconds = probe_seconds(data)
            if max_billed_seconds:
                if not seconds:
                    where = "视频URL" if video_url else "这个视频"
                    raise H3Error(f"无法在本地确认{where}的时长；如确认要按实际秒数扣费，请把「最多计费秒数」设为 0（不限）再运行（未扣费）")
                if seconds > max_billed_seconds + 1e-6:
                    raise H3Error(f"动作视频 {seconds:.1f} 秒，超过「最多计费秒数」{max_billed_seconds} 秒"
                                  f"（预计约 {seconds * PRICE_PER_SECOND:.2f}✦）；请剪短或调大上限（未扣费）")
            estimate = estimate_text(seconds)
            envelope.update({"estimated_seconds": seconds, "estimate": estimate})
            log(f"动作模仿：{estimate}")
            minutes, max_polls = wait_budget(seconds, max_wait_minutes, poll_interval)
            envelope["wait"] = {"max_minutes": minutes, "poll_interval": poll_interval}
            create_config = {**config, "timeout": max(int(config["timeout"]), MIN_CREATE_TIMEOUT)}
            headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
            fields, files = {"model": MODEL}, {}
            if video_url:
                fields["video_url"] = video_url
            else:
                files["input_reference"] = ("input.mp4", data, "video/mp4")
            if person_url:
                fields["image_url"] = person_url
            else:
                blob, filename, mime, note = encode_person(person)
                files["image"] = (filename, blob, mime)
                envelope["person_bytes"] = len(blob)
                if note:
                    envelope["note"] = note
            if files:
                request = {"files": files, "data": fields, "extra_headers": headers}
            else:
                request = {"json": fields, "extra_headers": {**headers, "Content-Type": "application/json"}}
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
            for _attempt in range(max_polls):
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

    def _failure(self, exc, envelope, task_id, secrets, notify):
        """Same as depth, with this model's hints."""
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


class LingliuMotionImitation(LingliuDepthVideo):
    FUNCTION = "imitate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "动作视频": (IO.VIDEO, {"tooltip": f"{VIDEO_RULE}。{PRICE_TIP}"}),
                "视频URL": ("STRING", {"default": "", "tooltip": "也可填公网 MP4 地址；和动作视频二选一。用 URL 时本地读不到时长，需把「最多计费秒数」设为 0。"}),
                "人物图": ("IMAGE", {"tooltip": f"{IMAGE_RULE}；多张只用第 1 张。"}),
                "人物图URL": ("STRING", {"default": "", "tooltip": "也可填公网图片地址；和人物图二选一。"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "轮询间隔": ("INT", {"default": 20, "min": 5, "max": 120}),
                "最长等待分钟": ("INT", {"default": 0, "min": 0, "max": 720,
                                    "tooltip": "0 = 自动：按视频时长估算（约每秒 100 秒，15–360 分钟）。"}),
                "最多计费秒数": ("INT", {"default": 60, "min": 0, "max": 3600,
                                    "tooltip": "动作视频超过这个秒数就不提交、不扣费；0 = 不限。"}),
                "防重复提交键": ("STRING", {
                    "default": "",
                    "tooltip": "留空=每次新建任务。填了以后，同一个键+同一组视频和人物图会直接返回上次的任务（包括失败的），不再扣费；换素材必须换键。",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    async def imitate(self, **kwargs):
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)
        result = await MotionImitationWorkflow().execute(
            video=kwargs.get("动作视频"),
            video_url=kwargs.get("视频URL", ""),
            person=kwargs.get("人物图"),
            person_url=kwargs.get("人物图URL", ""),
            api_key=kwargs.get("API密钥", ""),
            wait_for_completion=kwargs.get("等待完成", True),
            poll_interval=kwargs.get("轮询间隔", 20),
            max_wait_minutes=kwargs.get("最长等待分钟", 0),
            max_billed_seconds=kwargs.get("最多计费秒数", 60),
            idempotency_key=kwargs.get("防重复提交键", ""),
            notify=self._notify(unique_id),
        )
        return result.video, result.task_id, result.response, result.video_url
