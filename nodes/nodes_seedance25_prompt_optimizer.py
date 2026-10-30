import re
from pathlib import Path

from ..lingliu_config import mask_provider_text
from ..utils import image_to_data_url, tensor2pil
from .nodes_openai_compatible import LingliuBaseNode


DEFAULT_SEEDANCE25_PROMPT_OPTIMIZER_MODEL = "gemini-2.5-pro"

PROMPT_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "seedance25_prompt_optimizer_source.md"
)

STRICT_OUTPUT_FORMAT = """<OUTPUT_FORMAT>
响应必须且只能使用下面这一对机器标签封装优化后的正文：
<OPTIMIZED_PROMPT>
优化后的完整 Seedance 2.5 提示词正文
</OPTIMIZED_PROMPT>

第一个字符必须是起始标签“<OPTIMIZED_PROMPT>”，最后一个字符必须是结束标签“</OPTIMIZED_PROMPT>”；标签只能各出现一次，标签之外不得有任何字符。

标签内部只能放优化后的提示词正文，不得放标题、机器标签、Markdown 代码块、前言、解释、分析过程、任务类型、检查清单、免责声明、追问、建议或“自动补充了什么”。

即使用户要求解释、要求附加说明或输入信息不完整，也必须基于现有信息做最低风险优化并只返回提示词正文；无法可靠补充的细节保持不指定，不另行提醒或追问。

本节优先于本系统指令中任何允许“在最终提示词之外提醒用户”“提示用户”“追加说明”或“提出问题”的条款。
</OUTPUT_FORMAT>"""


def _extract_structured_instruction(markdown):
    blocks = re.findall(r"```text\s*\n(.*?)\n```", str(markdown or ""), flags=re.DOTALL)
    candidates = [
        block
        for block in blocks
        if "<ROLE>" in block
        and "<PRIMARY_OBJECTIVE>" in block
        and "<OUTPUT_FORMAT>" in block
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Seedance 2.5 提示词优化器内置指令必须包含且仅包含一个主结构化指令块。"
        )

    system_prompt, replacements = re.subn(
        r"<OUTPUT_FORMAT>.*?</OUTPUT_FORMAT>",
        STRICT_OUTPUT_FORMAT,
        candidates[0],
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("Seedance 2.5 提示词优化器无法替换输出格式约束。")
    return system_prompt.strip()


SEEDANCE25_PROMPT_OPTIMIZER_SYSTEM_PROMPT = _extract_structured_instruction(
    PROMPT_SOURCE_PATH.read_text(encoding="utf-8")
)


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
    r"优化后的[ \t]*Seedance[ \t]*2\.5[ \t]*提示词|"
    r"Seedance[ \t]*2\.5[ \t]*(?:提示词[ \t]*)?优化(?:提示词|结果)?|"
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
                part_type = raw_part_type.strip().lower()
                if part_type not in _TEXT_PART_TYPES:
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
    text = envelope_match.group(1).strip()
    if (
        not text
        or _FORBIDDEN_BODY_PREFIX.match(text)
        or _FORBIDDEN_META_LINE.search(text)
        or _FORBIDDEN_META_TAG.search(text)
        or "```" in text
    ):
        raise RuntimeError("模型没有返回优化后的提示词正文。")
    return text


class LingliuSeedance25PromptOptimizer(LingliuBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "待优化提示词": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "输入需要优化的 Seedance 2.5 视频提示词",
                    },
                ),
                "模型": (
                    "STRING",
                    {"default": DEFAULT_SEEDANCE25_PROMPT_OPTIMIZER_MODEL},
                ),
            },
            "optional": {
                "参考图像": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("优化后的提示词",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "Lingliu/Prompting"

    async def optimize_prompt(self, 待优化提示词, 模型, 参考图像=None):
        user_prompt = str(待优化提示词 or "").strip()
        if not user_prompt:
            raise ValueError("待优化提示词不能为空。")
        model = str(模型 or "").strip()
        if not model:
            raise ValueError("模型不能为空。")

        config = self.load_config()
        if not str(config.get("api_key") or "").strip():
            raise RuntimeError("请先使用 lingliu_api_set 设置全局 API key。")
        if not str(config.get("chat_path") or "").strip():
            provider = self.provider_display_name(config)
            raise RuntimeError(f"当前线路 {provider} 不支持文本模型调用。")

        user_content = user_prompt
        if 参考图像 is not None:
            user_content = [{"type": "text", "text": user_prompt}]
            for image in tensor2pil(参考图像):
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    }
                )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": SEEDANCE25_PROMPT_OPTIMIZER_SYSTEM_PROMPT,
                },
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
            return (normalize_optimized_prompt(content),)
        except Exception as exc:
            if isinstance(exc, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(
                f"Seedance 2.5 提示词优化失败：{mask_provider_text(exc)}"
            ) from exc
