"""Dedicated Lingliu nodes for the Fal Topaz image and video catalog."""

from __future__ import annotations

import asyncio
import uuid

from comfy.comfy_types import IO

from ..lingliu_config import (
    LINGLIU_BASE_URL,
    LINGLIU_PROVIDER,
    provider_defaults,
)
from ..utils import (
    LingliuNullVideoAdapter,
    LingliuVideoAdapter,
    blank_image_tensor,
    load_image_from_candidate,
    pil2tensor,
)
from .nodes_openai_compatible import LingliuBaseNode
from .nodes_seedance_official import LingliuSeedanceOfficialBase


class _LingliuTopazFalBase(LingliuSeedanceOfficialBase):
    CATEGORY = "Lingliu/Topaz"
    DEFAULT_PROVIDER = LINGLIU_PROVIDER
    FAL_ROOT = "/fal"
    MAX_POLL_ATTEMPTS = 360
    POLL_INTERVAL = 5

    def load_topaz_fal_config(self, api_key_override=""):
        runtime = self.load_config(api_key_override=str(api_key_override or ""))
        config = provider_defaults(LINGLIU_PROVIDER)
        config["base_url"] = LINGLIU_BASE_URL + self.FAL_ROOT
        config["api_key"] = str(runtime.get("api_key") or "").strip()
        config["timeout"] = int(runtime.get("timeout") or config.get("timeout") or 300)
        return config

    @staticmethod
    def _batch_count(value):
        if value is None:
            return 0
        if hasattr(value, "shape") and len(value.shape) == 4:
            return int(value.shape[0])
        return 1

    @classmethod
    def _require_single_image(cls, image):
        if cls._batch_count(image) != 1:
            raise ValueError("图像超分放大节点一次只接受 1 张图像，不接受图像批次。")
        return image

    async def _upload_image(self, image, config, name):
        public_url, _upload_info = await self.r2_upload_local_media(
            "image",
            self._require_single_image(image),
            name=name,
        )
        return public_url

    async def _upload_video(self, video, config, name):
        public_url, _upload_info = await self.r2_upload_local_media("video", video, name=name)
        return public_url

    async def _run_fal_once(self, config, endpoint_id, payload, unique_id, max_poll_attempts, poll_interval, label):
        return await self.run_fal_media_request(
            config,
            endpoint_id,
            payload,
            unique_id=unique_id,
            label=label,
            max_retries=1,
            initial_timeout=self.timeout(config),
            max_poll_attempts=int(max_poll_attempts),
            poll_interval=int(poll_interval),
        )


class LingliuTopazImageUpscale(_LingliuTopazFalBase):
    """Upscale one IMAGE with the configured Fal Topaz Wonder 3.5 route."""

    PUBLIC_MODEL_ID = "fal:topaz/image/wonder-3.5"
    ENDPOINT_ID = "topaz/image/wonder-3.5"
    OUTPUT_FORMATS = {"JPEG": "jpeg", "PNG": "png"}
    SUBJECT_DETECTION = {"全部": "All", "前景": "Foreground", "背景": "Background"}
    ENHANCEMENT_STRENGTH = {"自动": "", "低": "low", "中": "medium", "高": "high"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "放大倍率": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "输出格式": (list(cls.OUTPUT_FORMATS), {"default": "JPEG"}),
                "主体检测": (list(cls.SUBJECT_DETECTION), {"default": "全部"}),
                "人脸增强": ("BOOLEAN", {"default": True}),
                "人脸增强强度": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "人脸增强创造力": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "增强强度": (list(cls.ENHANCEMENT_STRENGTH), {"default": "自动"}),
            },
            "optional": {
                "图像URL": ("STRING", {"default": "", "multiline": False}),
                "裁剪填充": ("BOOLEAN", {"default": False}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 120, "min": 1, "max": 720, "step": 1}),
                "轮询间隔": ("INT", {"default": 2, "min": 1, "max": 60, "step": 1}),
                "上传名称": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图像", "图像URL", "响应", "请求ID")
    FUNCTION = "upscale_image"

    @classmethod
    def build_payload(
        cls,
        *,
        image_url,
        upscale_factor,
        output_format,
        subject_detection,
        face_enhancement,
        face_enhancement_strength,
        face_enhancement_creativity,
        enhancement_strength,
        crop_to_fill,
    ):
        factor = float(upscale_factor)
        if not 1.0 <= factor <= 4.0:
            raise ValueError("放大倍率必须在 1–4 倍之间。")
        if output_format not in cls.OUTPUT_FORMATS:
            raise ValueError("输出格式只能选择 JPEG 或 PNG。")
        if subject_detection not in cls.SUBJECT_DETECTION:
            raise ValueError("主体检测选项无效。")
        strength = cls.ENHANCEMENT_STRENGTH.get(str(enhancement_strength or "自动"), None)
        if strength is None:
            raise ValueError("增强强度选项无效。")
        face_strength = float(face_enhancement_strength)
        face_creativity = float(face_enhancement_creativity)
        if not 0.0 <= face_strength <= 1.0 or not 0.0 <= face_creativity <= 1.0:
            raise ValueError("人脸增强参数必须在 0–1 之间。")
        payload = {
            "image_url": str(image_url or "").strip(),
            "upscale_factor": factor,
            "crop_to_fill": bool(crop_to_fill),
            "output_format": cls.OUTPUT_FORMATS[output_format],
            "subject_detection": cls.SUBJECT_DETECTION[subject_detection],
            "face_enhancement": bool(face_enhancement),
            "face_enhancement_creativity": face_creativity,
            "face_enhancement_strength": face_strength,
        }
        if strength:
            payload["enhancement_strength"] = strength
        return payload

    @staticmethod
    def _extract_image_urls(payload):
        if not isinstance(payload, dict):
            return []
        image = payload.get("image")
        if isinstance(image, dict) and isinstance(image.get("url"), str) and image["url"].strip():
            return [image["url"].strip()]
        return []

    async def upscale_image(
        self,
        图像,
        放大倍率=2.0,
        输出格式="JPEG",
        主体检测="全部",
        人脸增强=True,
        人脸增强强度=0.8,
        人脸增强创造力=0.0,
        增强强度="自动",
        图像URL="",
        裁剪填充=False,
        API密钥="",
        等待完成=True,
        最大轮询次数=120,
        轮询间隔=2,
        上传名称="",
        unique_id=None,
    ):
        task_id = ""
        self.publish_task_id(unique_id, clear=True)
        try:
            self._require_single_image(图像)
            config = self.load_topaz_fal_config(api_key_override=API密钥)
            if not config["api_key"]:
                raise RuntimeError("缺少 API 密钥，请先在 lingliu_api_set 中配置。")
            image_url = str(图像URL or "").strip()
            if not image_url:
                image_url = await self._upload_image(图像, config, str(上传名称 or "").strip() or f"topaz_image_{uuid.uuid4().hex}")
            if not image_url.startswith(("http://", "https://")):
                raise ValueError("图像URL必须是 http:// 或 https:// 公网地址。")
            payload = self.build_payload(
                image_url=image_url,
                upscale_factor=放大倍率,
                output_format=输出格式,
                subject_detection=主体检测,
                face_enhancement=人脸增强,
                face_enhancement_strength=人脸增强强度,
                face_enhancement_creativity=人脸增强创造力,
                enhancement_strength=增强强度,
                crop_to_fill=裁剪填充,
            )
            result = await self._run_fal_once(config, self.ENDPOINT_ID, payload, unique_id, 最大轮询次数, 轮询间隔, "Topaz 图像请求 ID")
            task_id = result.get("request_id", "")
            image_urls = result.get("image_urls") or self._extract_image_urls(result.get("final"))
            if not image_urls:
                raise RuntimeError(f"Topaz 图像任务成功但没有返回 image.url：{result.get('final')}")
            images = [await asyncio.to_thread(load_image_from_candidate, url, self.timeout(config)) for url in image_urls[:1]]
            response = self.safe_json({
                "public_model_id": self.PUBLIC_MODEL_ID,
                "endpoint": f"/fal/{self.ENDPOINT_ID}",
                "request_id": task_id,
                "payload": {**payload, "image_url": "<uploaded-url>"},
                "submit": result.get("submit"),
                "final": result.get("final"),
            })
            return pil2tensor(images), image_urls[0], response, task_id
        except Exception as exc:
            return self.default_image(width=1, height=1), "", self.error_json(exc, task_id=task_id), task_id

    def default_image(self, width=1, height=1):
        return blank_image_tensor(width=width, height=height, color="black")


class LingliuTopazVideoUpscale(_LingliuTopazFalBase):
    """Upscale one VIDEO through the five platform-configured Starlight routes."""

    MODELS = {
        "Starlight Precise 2.6": "fal:topaz/video/starlight-precise-2.6",
        "Starlight HQ": "fal:topaz/video/starlight-hq",
        "Starlight Mini": "fal:topaz/video/starlight-mini",
        "Starlight Sharp": "fal:topaz/video/starlight-sharp",
        "Starlight Fast 2": "fal:topaz/video/starlight-fast-2",
    }
    SOFTNESS_CHOICES = ["使用默认", "1", "2", "3", "4", "5"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频": (IO.VIDEO,),
                "模型": (list(cls.MODELS), {"default": "Starlight Precise 2.6"}),
                "放大倍率": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "目标帧率": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                "H264输出": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "视频URL": ("STRING", {"default": "", "multiline": False}),
                "柔化程度": (cls.SOFTNESS_CHOICES, {"default": "使用默认"}),
                "API密钥": ("STRING", {"default": "", "password": True}),
                "等待完成": ("BOOLEAN", {"default": True}),
                "最大轮询次数": ("INT", {"default": 360, "min": 1, "max": 2000, "step": 1}),
                "轮询间隔": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "上传名称": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "视频URL", "响应", "请求ID")
    FUNCTION = "upscale_video"

    @classmethod
    def build_payload(cls, *, video_url, model, upscale_factor, target_fps, h264_output, softness):
        if model not in cls.MODELS:
            raise ValueError("视频放大模型选项无效。")
        factor = float(upscale_factor)
        if not 1.0 <= factor <= 4.0:
            raise ValueError("放大倍率必须在 1–4 倍之间。")
        target_fps = int(target_fps)
        if target_fps != 0 and not 16 <= target_fps <= 60:
            raise ValueError("目标帧率只能为 0（保持原帧率）或 16–60。")
        if softness not in cls.SOFTNESS_CHOICES:
            raise ValueError("柔化程度选项无效。")
        payload = {
            "video_url": str(video_url or "").strip(),
            "upscale_factor": factor,
            "H264_output": bool(h264_output),
        }
        if target_fps:
            payload["target_fps"] = target_fps
        if softness != "使用默认" and model == "Starlight Precise 2.6":
            payload["softness"] = int(softness)
        return payload

    async def upscale_video(
        self,
        视频,
        模型="Starlight Precise 2.6",
        放大倍率=2.0,
        目标帧率=0,
        H264输出=True,
        视频URL="",
        柔化程度="使用默认",
        API密钥="",
        等待完成=True,
        最大轮询次数=360,
        轮询间隔=5,
        上传名称="",
        unique_id=None,
    ):
        task_id = ""
        self.publish_task_id(unique_id, clear=True)
        try:
            config = self.load_topaz_fal_config(api_key_override=API密钥)
            if not config["api_key"]:
                raise RuntimeError("缺少 API 密钥，请先在 lingliu_api_set 中配置。")
            video_url = str(视频URL or "").strip()
            if not video_url:
                video_url = await self._upload_video(视频, config, str(上传名称 or "").strip() or f"topaz_video_{uuid.uuid4().hex}")
            if not video_url.startswith(("http://", "https://")):
                raise ValueError("视频URL必须是 http:// 或 https:// 公网地址。")
            payload = self.build_payload(
                video_url=video_url,
                model=模型,
                upscale_factor=放大倍率,
                target_fps=目标帧率,
                h264_output=H264输出,
                softness=柔化程度,
            )
            endpoint_id = self.MODELS[模型].split(":", 1)[1]
            result = await self._run_fal_once(config, endpoint_id, payload, unique_id, 最大轮询次数, 轮询间隔, "Topaz 视频请求 ID")
            task_id = result.get("request_id", "")
            video_urls = result.get("video_urls") or []
            if not video_urls:
                raise RuntimeError(f"Topaz 视频任务成功但没有返回 video.url：{result.get('final')}")
            response = self.safe_json({
                "public_model_id": self.MODELS[模型],
                "endpoint": f"/fal/{endpoint_id}",
                "request_id": task_id,
                "payload": {**payload, "video_url": "<uploaded-url>"},
                "submit": result.get("submit"),
                "final": result.get("final"),
            })
            return LingliuVideoAdapter(video_urls[0], mime_type="video/mp4"), video_urls[0], response, task_id
        except Exception as exc:
            return LingliuNullVideoAdapter(str(exc)), "", self.error_json(exc, task_id=task_id), task_id
