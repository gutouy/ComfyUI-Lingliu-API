import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


TARGET_SEEDANCE25 = "Seedance 2.5"
TARGET_MINIMAX_H3 = "MiniMax H3"
TARGET_VIDEO_MODELS = (TARGET_SEEDANCE25, TARGET_MINIMAX_H3)

SEEDANCE25_MODES = ("文生视频", "首帧图生视频", "首尾帧生视频", "参考生视频")
MINIMAX_H3_MODES = ("文生视频", "首帧图生视频", "首尾帧生视频", "尾帧图生视频", "全参考生视频")
ALL_GENERATION_MODES = tuple(dict.fromkeys((*SEEDANCE25_MODES, *MINIMAX_H3_MODES)))

STYLE_PRESETS = (
    "跟随原提示词",
    "极简产品广告",
    "3D 动画短片",
    "纸艺定格科普",
    "品牌宣传",
    "音乐 MV 字幕",
    "双人游戏开场",
    "纸拼贴讲解",
    "手绘实拍融合",
)

PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"
SEEDANCE_SOURCE_PATH = PROMPTS_ROOT / "seedance25_prompt_optimizer_source.md"
PROFILE_ASSET_ROOT = PROMPTS_ROOT / "video_prompt_optimizer"
H3_ASSET_ROOT = PROFILE_ASSET_ROOT / "minimax_h3"
STYLE_PROFILE_PATH = PROFILE_ASSET_ROOT / "style_profiles.json"
PROVENANCE_PATH = PROFILE_ASSET_ROOT / "provenance.json"

STRICT_OUTPUT_FORMAT = """<OUTPUT_FORMAT>
响应必须且只能使用下面这一对机器标签封装优化后的正文：
<OPTIMIZED_PROMPT>
优化后的完整视频提示词正文
</OPTIMIZED_PROMPT>

第一个字符必须是起始标签“<OPTIMIZED_PROMPT>”，最后一个字符必须是结束标签“</OPTIMIZED_PROMPT>”；标签只能各出现一次，标签之外不得有任何字符。

标签内部只能放优化后的提示词正文，不得放标题、机器标签、Markdown 代码块、前言、解释、分析过程、任务类型、检查清单、免责声明、追问、建议或“自动补充了什么”。

即使用户要求解释、附加说明或输入信息不完整，也必须基于现有信息做最低风险优化并只返回提示词正文；无法可靠补充的细节保持不指定，不另行提醒或追问。
</OUTPUT_FORMAT>"""

SEEDANCE_STRICT_OUTPUT_FORMAT = STRICT_OUTPUT_FORMAT.replace(
    "优化后的完整视频提示词正文",
    "优化后的完整 Seedance 2.5 提示词正文",
).replace(
    "即使用户要求解释、附加说明或输入信息不完整",
    "即使用户要求解释、要求附加说明或输入信息不完整",
).replace(
    "不另行提醒或追问。\n</OUTPUT_FORMAT>",
    "不另行提醒或追问。\n\n本节优先于本系统指令中任何允许“在最终提示词之外提醒用户”“提示用户”“追加说明”或“提出问题”的条款。\n</OUTPUT_FORMAT>",
)

H3_BASE_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
H3_REFERENCE_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

_H3_INTERNAL_MODE = {
    "文生视频": "T2VA",
    "首帧图生视频": "I2VA",
    "首尾帧生视频": "FL2VA",
    "尾帧图生视频": "L2VA",
    "全参考生视频": "Ref2VA",
}


def _extract_seedance_system_prompt(markdown):
    blocks = re.findall(r"```text\s*\n(.*?)\n```", str(markdown or ""), flags=re.DOTALL)
    candidates = [
        block
        for block in blocks
        if "<ROLE>" in block and "<PRIMARY_OBJECTIVE>" in block and "<OUTPUT_FORMAT>" in block
    ]
    if len(candidates) != 1:
        raise RuntimeError("Seedance 2.5 内置指令必须包含且仅包含一个主结构化指令块。")
    system_prompt, replacements = re.subn(
        r"<OUTPUT_FORMAT>.*?</OUTPUT_FORMAT>",
        SEEDANCE_STRICT_OUTPUT_FORMAT,
        candidates[0],
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("Seedance 2.5 内置指令无法替换输出格式约束。")
    return system_prompt.strip()


@lru_cache(maxsize=1)
def _load_seedance_system_prompt():
    return _extract_seedance_system_prompt(SEEDANCE_SOURCE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_style_profiles():
    payload = json.loads(STYLE_PROFILE_PATH.read_text(encoding="utf-8"))
    expected = set(STYLE_PRESETS[1:])
    if set(payload) != expected:
        raise RuntimeError("视频提示词风格 Profile 清单与节点选项不一致。")
    return payload


@lru_cache(maxsize=1)
def _load_h3_assets():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    for entry in provenance.get("vendored_files", []):
        path = PROFILE_ASSET_ROOT / entry["path"]
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest().upper()
        if actual != str(entry["sha256"]).upper():
            raise RuntimeError(f"MiniMax H3 官方文件校验失败：{entry['path']}。")
    return {
        "skill": (H3_ASSET_ROOT / "SKILL.md").read_text(encoding="utf-8"),
        "base": (H3_ASSET_ROOT / "base-en.txt").read_text(encoding="utf-8"),
        "reference": (H3_ASSET_ROOT / "ref-en.txt").read_text(encoding="utf-8"),
    }


def _style_instruction(style):
    if style == "跟随原提示词":
        return ""
    profile = _load_style_profiles()[style]
    return (
        f"风格预设：{style}\n"
        f"视觉约束：{profile['visual']}\n"
        f"镜头与节拍：{profile['camera']}\n"
        f"声音方向：{profile['sound']}\n"
        f"禁止项：{profile['forbidden']}"
    )


def _h3_system_prompt(mode):
    assets = _load_h3_assets()
    fields = H3_REFERENCE_FIELDS if mode == "全参考生视频" else H3_BASE_FIELDS
    schema = "\n\n".join(f"{field}:" for field in fields)
    guide = assets["reference"] if mode == "全参考生视频" else assets["base"]
    return (
        "You are a prompt optimizer for MiniMax H3. Follow the selected official mode and preserve "
        "the user's core intent, quantities, identities, dialogue, lyrics, visible text, references, "
        "timing, sound requirements, and prohibitions. Write the official rewrite sections in English; "
        "preserve dialogue, lyrics, and visible scene text in their original language.\n\n"
        f"Selected internal official mode: {_H3_INTERNAL_MODE[mode]}\n\n"
        "Required output field skeleton (keep this exact order):\n"
        f"{schema}\n\n"
        "Official portable skill:\n"
        f"{assets['skill']}\n\n"
        "Official selected guide:\n"
        f"{guide}\n\n"
        f"{STRICT_OUTPUT_FORMAT}"
    )


def _mode_instruction(target, mode, duration, labels, summary):
    parts = [f"目标视频模型：{target}", f"生成模式：{mode}", f"目标时长：{duration} 秒"]
    if target == TARGET_MINIMAX_H3:
        seconds = f"{duration:.2f}"
        if labels:
            parts.append("参考图片标签按输入顺序为：" + "、".join(labels) + "。")
        if mode == "首帧图生视频":
            parts.append(
                "最终输出第一行必须严格写为：For the target video, at 0.00 seconds into the target "
                "video, <Picture 1> (from [Shot 1]) is fully referenced."
            )
        elif mode == "首尾帧生视频":
            parts.append(
                "最终输出第一行必须使用以下格式，并将 Shot N 中的 N 替换为实际尾镜编号："
                "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
                "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns "
                f"with the {seconds}-second mark of the target video."
            )
        elif mode == "尾帧图生视频":
            parts.append(
                "最终输出第一行必须使用以下格式，并将 [Shot N] 中的 N 替换为实际尾镜编号："
                "How the reference pictures align with the target video — <Picture 1> (from [Shot N]) "
                f"aligns with the {seconds}-second mark of the target video."
            )
    elif labels:
        parts.append("参考图片按输入顺序标记为：" + "、".join(labels) + "。")
        if mode == "首帧图生视频":
            parts.append("@图片1 是实际首帧。")
        elif mode == "首尾帧生视频":
            parts.append("@图片1 是实际首帧，@图片2 是实际尾帧。")
    if summary:
        parts.append("参考素材职责说明：" + summary)
    return "\n".join(parts)


@dataclass(frozen=True)
class ResolvedPromptProfile:
    target_model: str
    mode: str
    style: str
    duration: int
    reference_count: int
    reference_summary: str
    reference_labels: tuple
    system_prompt: str = ""
    user_instruction: str = ""
    required_output_fields: tuple = ()

    def validate_output(self, text):
        body = str(text or "")
        first_line = body.splitlines()[0].strip() if body.splitlines() else ""
        if self.target_model == TARGET_MINIMAX_H3:
            if self.mode == "文生视频":
                valid_first_line = first_line.startswith("integrated_multimodal_description:")
            elif self.mode == "全参考生视频":
                valid_first_line = first_line == "subject_definitions:"
            elif self.mode == "首帧图生视频":
                valid_first_line = first_line == (
                    "For the target video, at 0.00 seconds into the target video, "
                    "<Picture 1> (from [Shot 1]) is fully referenced."
                )
            elif self.mode == "首尾帧生视频":
                valid_first_line = bool(
                    re.fullmatch(
                        r"How the reference pictures align with the target video — Picture 1 "
                        r"\(from Shot 1\) aligns with the 0\.00-second mark of the target video; "
                        r"Picture 2 \(from Shot \d+\) aligns with the "
                        + re.escape(f"{self.duration:.2f}")
                        + r"-second mark of the target video\.",
                        first_line,
                    )
                )
            else:
                valid_first_line = bool(
                    re.fullmatch(
                        r"How the reference pictures align with the target video — <Picture 1> "
                        r"\(from \[Shot \d+\]\) aligns with the "
                        + re.escape(f"{self.duration:.2f}")
                        + r"-second mark of the target video\.",
                        first_line,
                    )
                )
            if not valid_first_line:
                raise RuntimeError("MiniMax H3 优化结果缺少或错误的官方首行对齐契约。")

        positions = []
        for field in self.required_output_fields:
            match = re.search(rf"(?m)^{re.escape(field)}\s*:", body)
            if not match:
                raise RuntimeError(f"优化结果缺少 MiniMax H3 必需字段：{field}。")
            positions.append(match.start())
        if positions != sorted(positions):
            raise RuntimeError("优化结果中的 MiniMax H3 字段顺序不正确。")
        return body


class PromptProfileRegistry:
    _MODES = {
        TARGET_SEEDANCE25: SEEDANCE25_MODES,
        TARGET_MINIMAX_H3: MINIMAX_H3_MODES,
    }
    _DURATION_BOUNDS = {
        TARGET_SEEDANCE25: (4, 30),
        TARGET_MINIMAX_H3: (4, 15),
    }

    def mode_options(self, target_model):
        target = str(target_model or "").strip()
        if target not in self._MODES:
            raise ValueError(f"不支持的目标视频模型：{target or '空值'}。")
        return tuple(self._MODES[target])

    def duration_bounds(self, target_model):
        target = str(target_model or "").strip()
        if target not in self._DURATION_BOUNDS:
            raise ValueError(f"不支持的目标视频模型：{target or '空值'}。")
        return self._DURATION_BOUNDS[target]

    @staticmethod
    def _validate_reference_count(target, mode, count, summary):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("参考图像数量无效。")

        if mode == "文生视频":
            if count or summary:
                raise ValueError("文生视频不能连接参考图像或填写参考素材说明。")
            return
        if mode in {"首帧图生视频", "尾帧图生视频"}:
            if count != 1:
                raise ValueError(f"{mode}必须连接 1 张参考图像。")
            return
        if mode == "首尾帧生视频":
            if count != 2:
                raise ValueError("首尾帧生视频必须连接 2 张参考图像，顺序为首帧、尾帧。")
            return

        maximum = 30 if target == TARGET_SEEDANCE25 else 9
        if count > maximum:
            raise ValueError(f"{mode}最多连接 {maximum} 张参考图像。")
        if count == 0 and not summary:
            raise ValueError(f"{mode}必须连接参考图像或填写参考素材说明。")

    def resolve(
        self,
        *,
        target_model,
        mode,
        style,
        duration,
        reference_count,
        reference_summary,
    ):
        target = str(target_model or "").strip()
        selected_mode = str(mode or "").strip()
        selected_style = str(style or "").strip()
        summary = str(reference_summary or "").strip()

        modes = self.mode_options(target)
        if selected_mode not in modes:
            raise ValueError(f"{target} 不支持生成模式“{selected_mode or '空值'}”。")
        if selected_style not in STYLE_PRESETS:
            raise ValueError(f"不支持的风格预设：{selected_style or '空值'}。")

        minimum, maximum = self.duration_bounds(target)
        if isinstance(duration, bool) or not isinstance(duration, int) or not (minimum <= duration <= maximum):
            raise ValueError(f"{target} 的目标时长必须为 {minimum}–{maximum} 秒整数。")

        self._validate_reference_count(target, selected_mode, reference_count, summary)
        if target == TARGET_MINIMAX_H3:
            labels = tuple(f"<Picture {index}>" for index in range(1, reference_count + 1))
            system_prompt = _h3_system_prompt(selected_mode)
            required_fields = H3_REFERENCE_FIELDS if selected_mode == "全参考生视频" else H3_BASE_FIELDS
        else:
            labels = tuple(f"@图片{index}" for index in range(1, reference_count + 1))
            system_prompt = _load_seedance_system_prompt()
            required_fields = ()

        instruction_parts = []
        if target == TARGET_MINIMAX_H3 or selected_mode != "文生视频" or duration != 5 or summary or labels:
            instruction_parts.append(_mode_instruction(target, selected_mode, duration, labels, summary))
        style_instruction = _style_instruction(selected_style)
        if style_instruction:
            instruction_parts.append(style_instruction)

        return ResolvedPromptProfile(
            target_model=target,
            mode=selected_mode,
            style=selected_style,
            duration=duration,
            reference_count=reference_count,
            reference_summary=summary,
            reference_labels=labels,
            system_prompt=system_prompt,
            user_instruction="\n\n".join(instruction_parts),
            required_output_fields=required_fields,
        )
