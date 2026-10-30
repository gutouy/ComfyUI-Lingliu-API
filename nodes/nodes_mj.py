import asyncio
import json

from comfy.comfy_types import IO

from ..lingliu_config import LINGLIU_PROVIDER, build_url, provider_defaults
from ..utils import LingliuNullVideoAdapter, LingliuVideoAdapter, blank_image_tensor, image_to_data_url, load_image_from_candidate, pil2tensor
from .nodes_openai_compatible import LingliuBaseNode


FAST_MODE = "快速"
STANDARD_MODE = "标准"
RELAX_MODE = "慢速"
MODE_CHOICES = [FAST_MODE, STANDARD_MODE, RELAX_MODE]
MODE_PREFIXES = {FAST_MODE: "/fast/mj", STANDARD_MODE: "/mj", RELAX_MODE: "/relax/mj"}


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def _kw(kwargs, *names, default=None):
    aliases = {
        "API密钥": ("api_key",),
        "提示词": ("prompt",),
        "运行模式": ("mode",),
        "任务ID": ("task_id", "taskId"),
        "按钮数据": ("buttons", "buttons_json"),
        "按钮名称": ("button_name",),
        "自定义按钮ID": ("custom_id", "customId"),
        "等待完成": ("wait_for_completion",),
        "最大轮询次数": ("max_poll_attempts",),
        "轮询间隔": ("poll_interval",),
        "下载图片": ("download_image",),
        "图片URL": ("image_url", "imageUrl"),
        "参考图片1": ("image1", "ref_image1", "reference_image1"),
        "参考图片2": ("image2", "ref_image2", "reference_image2"),
        "参考图片3": ("image3", "ref_image3", "reference_image3"),
        "参考图片4": ("image4", "ref_image4", "reference_image4"),
        "参考图片": ("image", "ref_image", "reference_image"),
        "图片": ("image",),
        "源人脸图片": ("source_image", "source_face_image"),
        "目标图片": ("target_image",),
        "遮罩": ("mask",),
        "混图尺寸": ("blend_dimensions", "dimensions"),
        "运动幅度": ("motion",),
        "模型": ("model",),
    }
    for name in names:
        candidates = (name,) + aliases.get(name, ())
        for candidate in candidates:
            if candidate not in kwargs:
                continue
            value = kwargs.get(candidate)
            if value is not None:
                return value
        if name in kwargs:
            value = kwargs.get(name)
            if value is not None:
                return value
    return default


class LingliuMJBase(LingliuBaseNode):
    CATEGORY = "灵流API/MJ/核心"
    LINGLIU_BASE_URL = "https://api.lingliuapi.com"

    def load_mj_config(self, api_key_override=""):
        runtime_config = self.load_config(api_key_override=api_key_override)
        config = provider_defaults(LINGLIU_PROVIDER)
        config["base_url"] = self.LINGLIU_BASE_URL
        config["api_key"] = str(runtime_config.get("api_key") or "").strip()
        config["timeout"] = int(runtime_config.get("timeout") or config.get("timeout") or 300)
        return config

    def normalize_mode(self, mode):
        mode = str(mode or FAST_MODE).strip()
        return mode if mode in MODE_PREFIXES else FAST_MODE

    def mj_path(self, mode, path):
        path = "/" + str(path or "").strip().lstrip("/")
        prefix = MODE_PREFIXES[self.normalize_mode(mode)]
        return prefix + path

    async def request_json(self, method, config, path, **kwargs):
        url = build_url(config.get("base_url"), path)
        return await self.async_request_json_retrying(
            method,
            url,
            config,
            max_retries=1,
            initial_timeout=self.timeout(config),
            headers=self.headers(config, {"Content-Type": "application/json"}),
            **kwargs,
        )

    def collect_base64_array(self, *images):
        result = []
        for image in images:
            if image is None:
                continue
            batch_count = image.shape[0] if hasattr(image, "shape") and len(image.shape) == 4 else 1
            for index in range(batch_count):
                single_image = image[index : index + 1] if batch_count > 1 else image
                result.append(image_to_data_url(single_image))
        return result

    def extract_task_id(self, payload):
        if not isinstance(payload, dict):
            return ""
        for key in ("result", "task_id", "taskId", "id", "mjId"):
            value = payload.get(key)
            if value:
                return str(value)
        return ""

    def extract_buttons(self, payload):
        if not isinstance(payload, dict):
            return []
        buttons = payload.get("buttons") or payload.get("buttonList") or []
        if not buttons and isinstance(payload.get("properties"), dict):
            buttons = payload["properties"].get("buttons") or payload["properties"].get("buttonList") or []
        return buttons if isinstance(buttons, list) else []

    def extract_urls(self, payload):
        if not isinstance(payload, dict):
            return "", ""
        image_url = str(payload.get("imageUrl") or payload.get("image_url") or "")
        video_url = str(payload.get("videoUrl") or payload.get("video_url") or "")
        result = payload.get("result")
        if isinstance(result, str) and result.startswith(("http://", "https://")):
            if result.lower().split("?", 1)[0].endswith((".mp4", ".mov", ".webm", ".m4v")):
                video_url = video_url or result
            else:
                image_url = image_url or result
        elif isinstance(result, list) and result:
            first_result = str(result[0] or "")
            if first_result.startswith(("http://", "https://")):
                image_url = image_url or first_result
        properties = payload.get("properties")
        if isinstance(properties, dict):
            image_url = image_url or str(properties.get("imageUrl") or properties.get("image_url") or "")
            video_url = video_url or str(properties.get("videoUrl") or properties.get("video_url") or "")
        if not video_url and isinstance(payload.get("videoUrls"), list) and payload["videoUrls"]:
            video_url = str(payload["videoUrls"][0])
        return image_url, video_url

    def extract_error(self, payload):
        if not isinstance(payload, dict):
            return ""
        errors = []
        code = payload.get("code")
        description = payload.get("description") or payload.get("message")
        status = str(payload.get("status") or "").upper()
        explicit_error = payload.get("failReason") or payload.get("error")
        if explicit_error:
            errors.append(_json_text(explicit_error) if isinstance(explicit_error, (dict, list)) else str(explicit_error))
        if status == "FAILURE" and description:
            errors.append(str(description))
        if code not in (None, "", 0, 1, "0", "1"):
            message = f"MJ API returned code {code}"
            if description:
                message += f": {description}"
            errors.append(message)
        return "\n".join(dict.fromkeys(error for error in errors if error))

    def upstream_action_error(self, action_name, error):
        error = str(error or "").strip()
        if not error:
            return ""
        if "do_request_failed" in error or "Not Found" in error or "code 5" in error:
            return f"{action_name}失败：上游当前返回 {error}。这通常表示当前上游账号/通道未开通该 MJ 动作，或上游该路由暂不可用。"
        return error

    async def fetch_task_payload(self, config, task_id, mode):
        if not str(task_id or "").strip():
            raise ValueError("任务ID不能为空。")
        path = self.mj_path(mode, f"/task/{str(task_id).strip()}/fetch")
        return await self.request_json("GET", config, path)

    async def wait_for_task(self, config, task_id, mode, max_poll_attempts=120, poll_interval=5):
        payload = {}
        for _ in range(max(1, int(max_poll_attempts or 1))):
            payload = await self.fetch_task_payload(config, task_id, mode)
            status = str(payload.get("status") or "").upper()
            if status in {"SUCCESS", "FAILURE"}:
                return payload
            await asyncio.sleep(max(1, int(poll_interval or 1)))
        return payload

    async def image_tensor_from_url(self, url, timeout):
        if not url:
            return blank_image_tensor()
        image = await asyncio.to_thread(load_image_from_candidate, url, timeout)
        return pil2tensor(image)

    async def result_from_payload(self, payload, config, download_image=True):
        task_id = self.extract_task_id(payload)
        image_url, video_url = self.extract_urls(payload)
        buttons_json = _json_text(self.extract_buttons(payload))
        error = self.extract_error(payload)
        image = blank_image_tensor()
        if image_url and download_image:
            try:
                image = await self.image_tensor_from_url(image_url, self.timeout(config))
            except Exception as exc:
                error = f"{error}\n图片下载失败：{exc}".strip()
        return image, task_id, image_url, buttons_json, _json_text(payload), video_url, error


class LingliuMJImagine(LingliuMJBase):
    CATEGORY = "灵流API/MJ/生成"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "运行模式": (MODE_CHOICES, {"default": FAST_MODE}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
            },
            "optional": {
                "API密钥": ("STRING", {"default": ""}),
                "参考图片1": ("IMAGE",),
                "参考图片2": ("IMAGE",),
                "参考图片3": ("IMAGE",),
                "参考图片4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图片", "任务ID", "图片URL", "按钮数据", "原始响应", "错误信息")
    FUNCTION = "generate"

    def build_payload(self, **kwargs):
        return {
            "base64Array": self.collect_base64_array(
                _kw(kwargs, "参考图片1"),
                _kw(kwargs, "参考图片2"),
                _kw(kwargs, "参考图片3"),
                _kw(kwargs, "参考图片4"),
            ),
            "prompt": str(_kw(kwargs, "提示词", default="") or ""),
        }

    async def generate(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            if not config.get("api_key"):
                raise RuntimeError("请先在灵流API设置中填写 API密钥，或在当前节点填写。")
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            payload = self.build_payload(**kwargs)
            submit = await self.request_json("POST", config, self.mj_path(mode, "/submit/imagine"), json=payload)
            task_id = self.extract_task_id(submit)
            final_payload = submit
            if _kw(kwargs, "等待完成", default=True) and task_id:
                final_payload = await self.wait_for_task(
                    config,
                    task_id,
                    mode,
                    _kw(kwargs, "最大轮询次数", default=120),
                    _kw(kwargs, "轮询间隔", default=5),
                )
            image, task_id2, image_url, buttons, raw, _video_url, error = await self.result_from_payload(final_payload, config)
            return image, task_id2 or task_id, image_url, buttons, raw, error
        except Exception as exc:
            return blank_image_tensor(), "", "", "[]", self.error_json(str(exc)), str(exc)


class LingliuMJFetchTask(LingliuMJBase):
    CATEGORY = "灵流API/MJ/核心"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "任务ID": ("STRING", {"default": ""}),
                "运行模式": (MODE_CHOICES, {"default": FAST_MODE}),
                "下载图片": ("BOOLEAN", {"default": True}),
            },
            "optional": {"API密钥": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", IO.VIDEO, "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图片", "视频", "任务ID", "任务状态", "进度", "图片URL", "视频URL", "按钮数据", "原始响应")
    FUNCTION = "query"

    async def query(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            payload = await self.fetch_task_payload(config, _kw(kwargs, "任务ID", default=""), _kw(kwargs, "运行模式", default=FAST_MODE))
            image, task_id, image_url, buttons, raw, video_url, _error = await self.result_from_payload(
                payload, config, download_image=bool(_kw(kwargs, "下载图片", default=True))
            )
            return (
                image,
                LingliuVideoAdapter(video_url) if video_url else LingliuNullVideoAdapter(""),
                task_id,
                str(payload.get("status") or ""),
                str(payload.get("progress") or ""),
                image_url,
                video_url,
                buttons,
                raw,
            )
        except Exception as exc:
            return blank_image_tensor(), LingliuNullVideoAdapter(str(exc)), "", "ERROR", "", "", "", "[]", self.error_json(str(exc))


class LingliuMJAction(LingliuMJBase):
    CATEGORY = "灵流API/MJ/编辑"
    BUTTON_CHOICES = ["U1", "U2", "U3", "U4", "V1", "V2", "V3", "V4", "Reroll", "Zoom", "Pan", "自定义"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "任务ID": ("STRING", {"default": ""}),
                "按钮数据": ("STRING", {"multiline": True, "default": ""}),
                "按钮名称": (cls.BUTTON_CHOICES, {"default": "U1"}),
                "运行模式": (MODE_CHOICES, {"default": FAST_MODE}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
            },
            "optional": {"API密钥": ("STRING", {"default": ""}), "自定义按钮ID": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图片", "任务ID", "图片URL", "按钮数据", "原始响应", "错误信息")
    FUNCTION = "run"

    def parse_buttons(self, text):
        if not str(text or "").strip():
            raise ValueError("按钮数据不能为空，请先连接灵流MJ文生图或灵流MJ查询任务的按钮数据输出。")
        payload = json.loads(text)
        return payload if isinstance(payload, list) else payload.get("buttons", [])

    def find_custom_id(self, buttons, button_name, custom_id=""):
        if str(custom_id or "").strip():
            return str(custom_id).strip()
        wanted = str(button_name or "").lower()
        for item in buttons:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("emoji") or item.get("customId") or "").lower()
            if wanted and wanted in label and item.get("customId"):
                return str(item["customId"])
        raise ValueError(f"按钮数据中找不到按钮：{button_name}")

    def build_payload(self, **kwargs):
        task_id = str(_kw(kwargs, "任务ID", default="") or "").strip()
        if not task_id:
            raise ValueError("任务ID不能为空。")
        buttons = self.parse_buttons(_kw(kwargs, "按钮数据", default=""))
        return {
            "taskId": task_id,
            "customId": self.find_custom_id(buttons, _kw(kwargs, "按钮名称", default="U1"), _kw(kwargs, "自定义按钮ID", default="")),
        }

    async def run(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            submit = await self.request_json("POST", config, self.mj_path(mode, "/submit/action"), json=self.build_payload(**kwargs))
            task_id = self.extract_task_id(submit)
            final_payload = submit
            if _kw(kwargs, "等待完成", default=True) and task_id:
                final_payload = await self.wait_for_task(config, task_id, mode, _kw(kwargs, "最大轮询次数", default=120), _kw(kwargs, "轮询间隔", default=5))
            image, task_id2, image_url, buttons, raw, _video_url, error = await self.result_from_payload(final_payload, config)
            return image, task_id2 or task_id, image_url, buttons, raw, error
        except Exception as exc:
            return blank_image_tensor(), "", "", "[]", self.error_json(str(exc)), str(exc)


class LingliuMJModalRegionEdit(LingliuMJAction):
    CATEGORY = "灵流API/MJ/编辑"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["提示词"] = ("STRING", {"multiline": True, "default": ""})
        inputs["optional"]["遮罩"] = ("MASK",)
        return inputs

    def build_payload(self, **kwargs):
        payload = super().build_payload(**kwargs)
        payload["prompt"] = str(_kw(kwargs, "提示词", default="") or "")
        return payload

    async def run(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            submit = await self.request_json("POST", config, self.mj_path(mode, "/submit/modal"), json=self.build_payload(**kwargs))
            task_id = self.extract_task_id(submit)
            final_payload = submit
            if _kw(kwargs, "等待完成", default=True) and task_id:
                final_payload = await self.wait_for_task(config, task_id, mode, _kw(kwargs, "最大轮询次数", default=120), _kw(kwargs, "轮询间隔", default=5))
            image, task_id2, image_url, buttons, raw, _video_url, error = await self.result_from_payload(final_payload, config)
            return image, task_id2 or task_id, image_url, buttons, raw, error
        except Exception as exc:
            return blank_image_tensor(), "", "", "[]", self.error_json(str(exc)), str(exc)


class LingliuMJBlend(LingliuMJImagine):
    CATEGORY = "灵流API/MJ/生成"
    FUNCTION = "blend"
    DIMENSION_CHOICES = ["方形", "横向", "纵向"]
    DIMENSION_MAP = {"方形": "SQUARE", "横向": "LANDSCAPE", "纵向": "PORTRAIT", "SQUARE": "SQUARE", "LANDSCAPE": "LANDSCAPE", "PORTRAIT": "PORTRAIT"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "参考图片1": ("IMAGE",),
                "参考图片2": ("IMAGE",),
                "混图尺寸": (cls.DIMENSION_CHOICES, {"default": "方形"}),
                "运行模式": (MODE_CHOICES, {"default": FAST_MODE}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
            },
            "optional": {"API密钥": ("STRING", {"default": ""}), "参考图片3": ("IMAGE",), "参考图片4": ("IMAGE",)},
        }

    def build_payload(self, **kwargs):
        images = self.collect_base64_array(_kw(kwargs, "参考图片1"), _kw(kwargs, "参考图片2"), _kw(kwargs, "参考图片3"), _kw(kwargs, "参考图片4"))
        if len(images) < 2:
            raise ValueError("灵流MJ混图至少需要 2 张参考图片。")
        dimension = str(_kw(kwargs, "混图尺寸", default="方形") or "方形").strip()
        return {"base64Array": images, "dimensions": self.DIMENSION_MAP.get(dimension, "SQUARE")}

    async def blend(self, **kwargs):
        kwargs = dict(kwargs)
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            submit = await self.request_json("POST", config, self.mj_path(mode, "/submit/blend"), json=self.build_payload(**kwargs))
            task_id = self.extract_task_id(submit)
            final_payload = submit
            if _kw(kwargs, "等待完成", default=True) and task_id:
                final_payload = await self.wait_for_task(config, task_id, mode, _kw(kwargs, "最大轮询次数", default=120), _kw(kwargs, "轮询间隔", default=5))
            image, task_id2, image_url, buttons, raw, _video_url, error = await self.result_from_payload(final_payload, config)
            error = self.upstream_action_error("灵流MJ混图", error)
            return image, task_id2 or task_id, image_url, buttons, raw, error
        except Exception as exc:
            return blank_image_tensor(), "", "", "[]", self.error_json(str(exc)), str(exc)


class LingliuMJDescribe(LingliuMJBase):
    CATEGORY = "灵流API/MJ/生成"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "参考图片": ("IMAGE",),
                "运行模式": (MODE_CHOICES, {"default": FAST_MODE}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
            },
            "optional": {"API密钥": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("描述文本", "任务ID", "原始响应", "错误信息")
    FUNCTION = "describe"

    async def describe(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            payload = {"base64": self.collect_base64_array(_kw(kwargs, "参考图片"))[0]}
            result = await self.request_json("POST", config, self.mj_path(mode, "/submit/describe"), json=payload)
            task_id = self.extract_task_id(result)
            final_payload = result
            if _kw(kwargs, "等待完成", default=True) and task_id:
                final_payload = await self.wait_for_task(
                    config,
                    task_id,
                    mode,
                    _kw(kwargs, "最大轮询次数", default=120),
                    _kw(kwargs, "轮询间隔", default=5),
                )
            text = str(
                final_payload.get("promptEn")
                or final_payload.get("prompt")
                or final_payload.get("description")
                or final_payload.get("result")
                or ""
            )
            return text, self.extract_task_id(final_payload) or task_id, _json_text(final_payload), self.extract_error(final_payload)
        except Exception as exc:
            return "", "", self.error_json(str(exc)), str(exc)


class LingliuMJUploadImages(LingliuMJBase):
    CATEGORY = "灵流API/MJ/核心"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"图片": ("IMAGE",), "运行模式": (MODE_CHOICES, {"default": FAST_MODE})}, "optional": {"API密钥": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("图片数据", "原始响应", "错误信息")
    FUNCTION = "upload"

    async def upload(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            payload = {"base64Array": self.collect_base64_array(_kw(kwargs, "图片"))}
            result = await self.request_json("POST", config, self.mj_path(_kw(kwargs, "运行模式", default=FAST_MODE), "/submit/upload-discord-images"), json=payload)
            return _json_text(result.get("result") or result), _json_text(result), self.extract_error(result)
        except Exception as exc:
            return "", self.error_json(str(exc)), str(exc)


class LingliuMJShorten(LingliuMJBase):
    CATEGORY = "灵流API/MJ/高级"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"提示词": ("STRING", {"multiline": True}), "运行模式": (MODE_CHOICES, {"default": FAST_MODE})}, "optional": {"API密钥": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("精简提示词", "原始响应", "错误信息")
    FUNCTION = "shorten"

    async def shorten(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            result = await self.request_json("POST", config, self.mj_path(_kw(kwargs, "运行模式", default=FAST_MODE), "/submit/shorten"), json={"prompt": str(_kw(kwargs, "提示词", default="") or "")})
            return str(result.get("result") or result.get("description") or ""), _json_text(result), self.extract_error(result)
        except Exception as exc:
            return "", self.error_json(str(exc)), str(exc)


class LingliuMJEdits(LingliuMJImagine):
    CATEGORY = "灵流API/MJ/编辑"
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True}),
                "运行模式": (MODE_CHOICES, {"default": FAST_MODE}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
            },
            "optional": {
                "API密钥": ("STRING", {"default": ""}),
                "图片": ("IMAGE",),
                "图片URL": ("STRING", {"default": ""}),
                "遮罩": ("MASK",),
                "参考图片1": ("IMAGE",),
            },
        }

    def mask_to_data_url(self, mask):
        if mask is None:
            return ""
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        return image_to_data_url(mask[0:1])

    def build_payload(self, **kwargs):
        prompt = str(_kw(kwargs, "提示词", default="") or "")
        image_url = str(_kw(kwargs, "图片URL", default="") or "").strip()
        image = _kw(kwargs, "图片", default=None)
        if image is None:
            image = _kw(kwargs, "参考图片1", default=None)
        if image_url:
            image_value = image_url
        elif image is not None:
            image_value = self.collect_base64_array(image)[0]
        else:
            raise ValueError("灵流MJ图像编辑需要提供 图片 或 图片URL。")
        payload = {"prompt": prompt, "image": image_value}
        mask = _kw(kwargs, "遮罩", default=None)
        if mask is not None:
            payload["maskBase64"] = self.mask_to_data_url(mask)
        return payload

    async def edit(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            submit = await self.request_json("POST", config, self.mj_path(mode, "/submit/edits"), json=self.build_payload(**kwargs))
            task_id = self.extract_task_id(submit)
            final_payload = submit
            if _kw(kwargs, "等待完成", default=True) and task_id:
                final_payload = await self.wait_for_task(config, task_id, mode, _kw(kwargs, "最大轮询次数", default=120), _kw(kwargs, "轮询间隔", default=5))
            image, task_id2, image_url, buttons, raw, _video_url, error = await self.result_from_payload(final_payload, config)
            error = self.upstream_action_error("灵流MJ图像编辑", error)
            return image, task_id2 or task_id, image_url, buttons, raw, error
        except Exception as exc:
            return blank_image_tensor(), "", "", "[]", self.error_json(str(exc)), str(exc)


class LingliuMJVideo(LingliuMJBase):
    CATEGORY = "灵流API/MJ/视频"
    MOTION_CHOICES = ["低", "高", "参考图运动"]
    MOTION_MAP = {"低": "low", "高": "high", "参考图运动": "image", "low": "low", "high": "high", "image": "image"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "运动幅度": (cls.MOTION_CHOICES, {"default": "低"}),
                "运行模式": (MODE_CHOICES, {"default": FAST_MODE}),
                "等待完成": ("BOOLEAN", {"default": True}),
            },
            "optional": {"API密钥": ("STRING", {"default": ""}), "图片": ("IMAGE",), "图片URL": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "任务ID", "视频URL", "原始响应", "错误信息")
    FUNCTION = "video"

    def build_payload(self, **kwargs):
        payload = {
            "prompt": str(_kw(kwargs, "提示词", default="") or ""),
            "motion": self.MOTION_MAP.get(str(_kw(kwargs, "运动幅度", default="低") or "低").strip(), "low"),
        }
        image_url = str(_kw(kwargs, "图片URL", default="") or "").strip()
        image = _kw(kwargs, "图片", default=None)
        if image_url:
            payload["image"] = image_url
        elif image is not None:
            payload["image"] = self.collect_base64_array(image)[0]
        return payload

    async def video(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        try:
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            submit = await self.request_json("POST", config, self.mj_path(mode, "/submit/video"), json=self.build_payload(**kwargs))
            task_id = self.extract_task_id(submit)
            final_payload = submit
            if _kw(kwargs, "等待完成", default=True) and task_id:
                final_payload = await self.wait_for_task(config, task_id, mode)
            _image, _task, _image_url, _buttons, raw, video_url, error = await self.result_from_payload(final_payload, config)
            error = self.upstream_action_error("灵流MJ图生视频", error)
            return LingliuVideoAdapter(video_url) if video_url else LingliuNullVideoAdapter(error), task_id, video_url, raw, error
        except Exception as exc:
            return LingliuNullVideoAdapter(str(exc)), "", "", self.error_json(str(exc)), str(exc)


class LingliuMJChatCompat(LingliuMJImagine):
    CATEGORY = "灵流API/MJ/高级"
    MODELS = ["chat_fast_imagine", "chat_fast_video", "chat_relax_imagine", "midjourney"]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["模型"] = (cls.MODELS, {"default": cls.MODELS[0]})
        return inputs


class LingliuMJSwapFace(LingliuMJBase):
    CATEGORY = "灵流API/MJ/高级"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"源人脸图片": ("IMAGE",), "目标图片": ("IMAGE",), "运行模式": (MODE_CHOICES, {"default": FAST_MODE})},
            "optional": {
                "API密钥": ("STRING", {"default": ""}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图片", "图片URL", "原始响应", "错误信息", "任务ID")
    FUNCTION = "swap"

    async def swap(self, **kwargs):
        config = self.load_mj_config(_kw(kwargs, "API密钥", default=""))
        task_id = ""
        try:
            mode = _kw(kwargs, "运行模式", default=FAST_MODE)
            payload = {
                "sourceBase64": self.collect_base64_array(_kw(kwargs, "源人脸图片"))[0],
                "targetBase64": self.collect_base64_array(_kw(kwargs, "目标图片"))[0],
            }
            result = await self.request_json("POST", config, self.mj_path(mode, "/insight-face/swap"), json=payload)
            # The swap submit only returns a task ID; the swapped image arrives via task fetch.
            task_id = self.extract_task_id(result)
            waited = bool(_kw(kwargs, "等待完成", default=True)) and bool(task_id)
            if waited:
                result = await self.wait_for_task(
                    config,
                    task_id,
                    mode,
                    _kw(kwargs, "最大轮询次数", default=120),
                    _kw(kwargs, "轮询间隔", default=5),
                )
            image, task_id2, image_url, _buttons, raw, _video_url, error = await self.result_from_payload(result, config)
            error = self.upstream_action_error("灵流MJ换脸", error)
            if waited and not error:
                status = str(result.get("status") or "").upper()
                if status == "FAILURE":
                    error = "灵流MJ换脸失败：上游未返回原因。"
                elif status != "SUCCESS":
                    error = "等待超时，任务仍在进行，可用任务ID到「灵流MJ查询任务」继续查询。"
                elif not image_url:
                    error = "灵流MJ换脸任务已完成，但没有返回图片。"
            return image, image_url, raw, error, task_id2 or task_id
        except Exception as exc:
            return blank_image_tensor(), "", self.error_json(str(exc)), str(exc), task_id


class LingliuMJAdvancedRead(LingliuMJFetchTask):
    CATEGORY = "灵流API/MJ/高级"
