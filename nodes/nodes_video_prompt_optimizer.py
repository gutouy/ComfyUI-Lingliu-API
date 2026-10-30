import re

from ..lingliu_config import build_url, mask_provider_text
from ..utils import image_to_png_bytes
from .nodes_openai_compatible import LingliuBaseNode
from .video_prompt_profiles import (
    ALL_GENERATION_MODES,
    STYLE_PRESETS,
    TARGET_VIDEO_MODELS,
    PromptProfileRegistry,
)
DEFAULT_OPTIMIZER_MODEL = "gemini-3.5-flash"

_OUTER_CODE_FENCE = re.compile(
    r"\A```(?:text|markdown|plaintext)?\s*\n?(.*?)\n?```\s*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)
_PROMPT_ENVELOPE = re.compile(
    r"\A<OPTIMIZED_PROMPT>[ \t]*\r?\n?(.*?)\r?\n?[ \t]*</OPTIMIZED_PROMPT>\s*\Z",
    flags=re.DOTALL,
)
_FORBIDDEN_BODY_PREFIX = re.compile(
    r"\A(?:#{1,6}[ \t]*)?(?:"
    r"优化(?:后的)?(?:结果|提示词|提示词正文)|"
    r"优化后的[ \t]*(?:Seedance[ \t]*2\.5|MiniMax[ \t]*H3)?[ \t]*提示词|"
    r"(?:Seedance[ \t]*2\.5|MiniMax[ \t]*H3)[ \t]*(?:提示词[ \t]*)?优化(?:提示词|结果)?|"
    r"以下是(?:优化后的)?(?:结果|提示词|提示词正文)|"
    r"这是(?:我)?为你优化的(?:结果|版本|提示词)|"
    r"说明[ \t]*[:：]|解释[ \t]*[:：]"
    r")",
    flags=re.IGNORECASE,
)
_FORBIDDEN_META_LINE = re.compile(
    r"(?mi)^[ \t]*(?:"
    r"#{1,6}[ \t]*自动补充了什么[ \t]*|"
    r"(?:补充说明|优化说明|修改说明|说明|解释|备注|分析过程|任务类型|"
    r"检查清单|免责声明|建议|前言)[ \t]*[:：][^\r\n]*"
    r")[ \t]*\r?$"
)
_FORBIDDEN_META_TAG = re.compile(
    r"</?(?:analysis|reasoning|explanation|notes?|checklist|metadata|meta|task_type)\b[^>]*>",
    flags=re.IGNORECASE,
)
_PROMPT_OPEN_TAG = "<OPTIMIZED_PROMPT>"
_PROMPT_CLOSE_TAG = "</OPTIMIZED_PROMPT>"
_TEXT_PART_TYPES = {"text", "output_text"}


def _response_content_to_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RuntimeError("文本模型响应内容不是可用的文本。")

    text_parts = []
    for part in content:
        if isinstance(part, str):
            text = part
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            if "type" in part:
                raw_part_type = part["type"]
                if not isinstance(raw_part_type, str):
                    raise RuntimeError("文本模型响应包含不支持的非文本内容。")
                if raw_part_type.strip().lower() not in _TEXT_PART_TYPES:
                    raise RuntimeError("文本模型响应包含不支持的非文本内容。")
            text = part["text"]
        else:
            raise RuntimeError("文本模型响应包含不支持的非文本内容。")
        if text.strip():
            text_parts.append(text)
    if not text_parts:
        raise RuntimeError("文本模型响应中没有提示词正文。")
    return "\n".join(text_parts)


def normalize_optimized_prompt(content):
    text = _response_content_to_text(content).lstrip("\ufeff").strip()
    fence_match = _OUTER_CODE_FENCE.fullmatch(text)
    if fence_match:
        text = fence_match.group(1).strip()
    if text.count(_PROMPT_OPEN_TAG) != 1 or text.count(_PROMPT_CLOSE_TAG) != 1:
        raise RuntimeError("模型响应缺少唯一的提示词正文标签。")
    envelope_match = _PROMPT_ENVELOPE.fullmatch(text)
    if not envelope_match:
        raise RuntimeError("模型响应不符合提示词正文格式。")
    body = envelope_match.group(1).strip()
    if (
        not body
        or _FORBIDDEN_BODY_PREFIX.match(body)
        or _FORBIDDEN_META_LINE.search(body)
        or _FORBIDDEN_META_TAG.search(body)
        or "```" in body
    ):
        raise RuntimeError("模型没有返回优化后的提示词正文。")
    return body


PROFILE_REGISTRY = PromptProfileRegistry()


def _reference_image_tensors(image_batch):
    if image_batch is None:
        return []
    batch_size = image_batch.shape[0] if hasattr(image_batch, "shape") and len(image_batch.shape) == 4 else 1
    return [image_batch[index : index + 1] if batch_size > 1 else image_batch for index in range(batch_size)]


class LingliuVideoPromptOptimizer(LingliuBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "待优化提示词": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "输入需要优化的视频提示词",
                    },
                ),
                "目标视频模型": (list(TARGET_VIDEO_MODELS), {"default": "Seedance 2.5"}),
                "生成模式": (list(ALL_GENERATION_MODES), {"default": "文生视频"}),
                "风格预设": (list(STYLE_PRESETS), {"default": "跟随原提示词"}),
                "目标时长": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1}),
                "优化所用文本模型": ("STRING", {"default": DEFAULT_OPTIMIZER_MODEL}),
            },
            "optional": {
                "参考图像": ("IMAGE",),
                "参考素材说明": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("优化后的提示词",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "Lingliu/Prompting"

    @staticmethod
    def _extract_uploaded_url(payload):
        if isinstance(payload, dict):
            for key in ("url", "download_url", "file_url"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                    return value.strip()
            data = payload.get("data")
            if isinstance(data, dict):
                return LingliuVideoPromptOptimizer._extract_uploaded_url(data)
        raise RuntimeError("参考图上传响应没有可用的 HTTP(S) URL。")

    async def _upload_reference_images(self, config, image_batch, image_tensors):
        if not image_tensors:
            return []
        endpoint = build_url(config.get("base_url"), "/v1/files")
        uploaded_urls = []
        for index, image_tensor in enumerate(image_tensors, 1):
            response_payload = await self.async_request_json(
                "POST",
                endpoint,
                config,
                headers=self.headers(config),
                data={"purpose": "assistants"},
                files=[
                    (
                        "file",
                        (
                            f"prompt_reference_{index}.png",
                            image_to_png_bytes(image_tensor),
                            "image/png",
                        ),
                    )
                ],
            )
            uploaded_urls.append(self._extract_uploaded_url(response_payload))
        return uploaded_urls

    async def optimize_prompt(
        self,
        待优化提示词,
        目标视频模型,
        生成模式,
        风格预设,
        目标时长,
        优化所用文本模型,
        参考图像=None,
        参考素材说明="",
    ):
        user_prompt = str(待优化提示词 or "").strip()
        if not user_prompt:
            raise ValueError("待优化提示词不能为空。")
        optimizer_model = str(优化所用文本模型 or "").strip()
        if not optimizer_model:
            raise ValueError("优化所用文本模型不能为空。")

        reference_images = _reference_image_tensors(参考图像)
        profile = PROFILE_REGISTRY.resolve(
            target_model=目标视频模型,
            mode=生成模式,
            style=风格预设,
            duration=目标时长,
            reference_count=len(reference_images),
            reference_summary=参考素材说明,
        )

        config = self.load_config()
        if not str(config.get("api_key") or "").strip():
            raise RuntimeError("请先使用 lingliu_api_set 设置全局 API key。")
        if not str(config.get("chat_path") or "").strip():
            provider = self.provider_display_name(config)
            raise RuntimeError(f"当前线路 {provider} 不支持文本模型调用。")

        request_text = user_prompt
        if profile.user_instruction:
            request_text = f"{profile.user_instruction}\n\n待优化提示词：\n{user_prompt}"

        uploaded_urls = await self._upload_reference_images(config, 参考图像, reference_images)

        user_content = request_text
        if reference_images:
            user_content = [{"type": "text", "text": request_text}]
            for image_url in uploaded_urls:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )

        payload = {
            "model": optimizer_model,
            "messages": [
                {"role": "system", "content": profile.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 8192,
            "stream": False,
        }

        try:
            response_payload = await self.async_request_json(
                "POST",
                self.endpoint(config, "chat_path"),
                config,
                headers=self.headers(config, {"Content-Type": "application/json"}),
                json=payload,
            )
            choices = response_payload.get("choices", []) if isinstance(response_payload, dict) else []
            if not choices or not isinstance(choices[0], dict):
                raise RuntimeError("文本模型响应中没有 choices。")
            message = choices[0].get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not content and choices[0].get("text"):
                content = choices[0]["text"]
            body = normalize_optimized_prompt(content)
            return (profile.validate_output(body),)
        except Exception as exc:
            if isinstance(exc, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"视频提示词优化失败：{mask_provider_text(exc)}") from exc
