"""Stable ComfyUI interface for the WAN low-price media workflow."""

import re

from comfy.comfy_types import IO

from .nodes_openai_compatible import LingliuBaseNode
from .wan3_low_workflow import (
    MEDIA_LIMITS,
    WAN3_LOW_MODELS,
    WAN3_LOW_RESOLUTIONS,
    WAN3_LOW_DURATION_RANGES,
    WAN3_LOW_REFERENCE_REQUIRED,
    WAN3_LOW_RETIRED_MESSAGE,
    WAN3_LOW_RETIRED_MODELS,
    Wan3LowError,
    Wan3LowWorkflow,
)

# Accept old API prompts; the frontend presents only the effective default.
LEGACY_RATIOS = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
LowSeedanceError = Wan3LowError


class LingliuWan3Low(LingliuBaseNode):
    CATEGORY = "Lingliu/Wan"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("视频", "任务ID", "响应", "视频URL")
    FUNCTION = "generate_video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, 模型):
        # Replaces ComfyUI's "Value not in list" for retired models with a clear message.
        if 模型 in WAN3_LOW_RETIRED_MODELS:
            return WAN3_LOW_RETIRED_MESSAGE
        if 模型 not in WAN3_LOW_MODELS:
            return f"模型 {模型} 不在当前 WAN3 低价模型列表中。"
        return True

    @classmethod
    def INPUT_TYPES(cls):
        # Keep the original widget order to preserve V14 widgets_values.
        optional = {
            "参考图像": ("IMAGE", {"tooltip": "兼容旧图片批次；与所有图片接口合计最多 10 张。"}),
            "API密钥": ("STRING", {"default": "", "password": True}),
            "等待完成": ("BOOLEAN", {"default": True}),
            "最大轮询次数": ("INT", {"default": 180, "min": 1, "max": 720, "step": 1}),
            "轮询间隔": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
        }
        for prefix, kind, io_type in (("参考图", "image", "IMAGE"), ("参考视频", "video", IO.VIDEO), ("参考音频", "audio", IO.AUDIO)):
            optional.update({
                f"{prefix}{i}": (io_type, {"tooltip": "连接后自动出现下一个接口；空接口不占素材名额。"})
                for i in range(1, MEDIA_LIMITS[kind] + 1)
            })
        return {
            "required": {
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "模型": (list(WAN3_LOW_MODELS), {"default": WAN3_LOW_MODELS[0]}),
                "生成模式": (["自动", "文生视频", "参考生视频"], {"default": "自动"}),
                "时长": ("INT", {"default": 5, "min": 2, "max": 30, "step": 1}),
                "比例": (["模型默认", *LEGACY_RATIOS], {"default": "模型默认", "tooltip": "当前合同使用模型默认；旧比例仅供兼容读取。"}),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    async def generate_video(self, **kwargs):
        limits = {"参考图": 10, "参考视频": 5, "参考音频": 5}
        for name, value in kwargs.items():
            match = re.fullmatch(r"(参考图|参考视频|参考音频)(\d+)", name)
            if match and value is not None and not 1 <= int(match[2]) <= limits[match[1]]:
                raise ValueError(f"{name} 超出该类素材接口上限。")
        raw_assets = [("参考图像", "image", "reference_image", kwargs.get("参考图像"))]
        for prefix, kind in (("参考图", "image"), ("参考视频", "video"), ("参考音频", "audio")):
            raw_assets.extend(
                (f"{prefix}{i}", kind, f"reference_{kind}", kwargs.get(f"{prefix}{i}"))
                for i in range(1, MEDIA_LIMITS[kind] + 1)
            )
        unique_id = kwargs.get("unique_id")
        self.publish_task_id(unique_id, clear=True)

        def notify(task_id, status):
            if task_id:
                self.publish_task_id(unique_id, task_id, append=False)
            self.publish_live_message(unique_id, f"Task ID: {task_id}\n状态: {status}", append=False)

        result = await Wan3LowWorkflow().execute(
            prompt=kwargs.get("提示词", ""),
            model=kwargs.get("模型", WAN3_LOW_MODELS[0]),
            mode=kwargs.get("生成模式", "自动"),
            duration=kwargs.get("时长", 5),
            ratio=kwargs.get("比例", "模型默认"),
            raw_assets=raw_assets,
            api_key=kwargs.get("API密钥", ""),
            wait_for_completion=kwargs.get("等待完成", True),
            max_poll_attempts=kwargs.get("最大轮询次数", 180),
            poll_interval=kwargs.get("轮询间隔", 10),
            notify=notify,
        )
        return result.video, result.task_id, result.response, result.video_url


__all__ = ["LingliuWan3Low", "Wan3LowWorkflow", "WAN3_LOW_MODELS", "WAN3_LOW_RESOLUTIONS", "WAN3_LOW_DURATION_RANGES"]
