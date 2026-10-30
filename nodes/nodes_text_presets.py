from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    import folder_paths
except Exception:
    folder_paths = None


DEFAULT_CATEGORY = "默认分类"
PRESET_PLACEHOLDER = "暂无预设"
PRESET_API_PATH = "/lingliu/text_presets"


def _preset_dir() -> Path:
    if folder_paths is not None:
        try:
            user_dir = Path(folder_paths.get_user_directory())
            return user_dir / "default" / "comfyui_lingliu_api"
        except Exception:
            pass
    return Path(__file__).resolve().parents[1] / "user_data"


def _preset_path() -> Path:
    return _preset_dir() / "text_presets.json"


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_category(value: str) -> str:
    return str(value or "").strip() or DEFAULT_CATEGORY


def _normalize_presets(data: Any) -> list[dict[str, str]]:
    if not isinstance(data, dict):
        return []
    raw_items = data.get("presets")
    if not isinstance(raw_items, list):
        return []

    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        category = _normalize_category(str(item.get("category") or ""))
        key = (category, title)
        if not title or key in seen:
            continue
        items.append(
            {
                "category": category,
                "title": title,
                "content": str(item.get("content") or ""),
                "updated_at": str(item.get("updated_at") or ""),
            }
        )
        seen.add(key)
    return items


def load_text_presets() -> list[dict[str, str]]:
    path = _preset_path()
    if not path.exists():
        return []
    try:
        return _normalize_presets(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return []


def list_text_preset_categories() -> list[str]:
    categories: list[str] = []
    for item in load_text_presets():
        category = _normalize_category(item.get("category", ""))
        if category not in categories:
            categories.append(category)
    return categories


def list_text_preset_titles(category: str | None = None) -> list[str]:
    selected_category = _normalize_category(category or DEFAULT_CATEGORY)
    return [
        item["title"]
        for item in load_text_presets()
        if _normalize_category(item.get("category", "")) == selected_category
    ]


def list_all_text_preset_titles() -> list[str]:
    titles: list[str] = []
    for item in load_text_presets():
        title = str(item.get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
    return titles


def _read_raw_presets_for_save(path: Path) -> tuple[list[dict[str, str]], bool]:
    if not path.exists():
        return [], False
    try:
        return _normalize_presets(json.loads(path.read_text(encoding="utf-8"))), False
    except Exception:
        return [], True


def save_text_preset(category: str, title: str, content: str) -> tuple[bool, str]:
    category = _normalize_category(category)
    title = str(title or "").strip()
    if not title:
        return False, "保存失败：预设标题不能为空。"

    path = _preset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    presets, was_invalid = _read_raw_presets_for_save(path)

    if was_invalid and path.exists():
        backup = path.with_suffix(f".invalid-{int(time.time())}.bak")
        path.replace(backup)

    updated = False
    now = _now_text()
    for item in presets:
        if _normalize_category(item.get("category", "")) == category and item["title"] == title:
            item["content"] = str(content or "")
            item["updated_at"] = now
            updated = True
            break
    if not updated:
        presets.append(
            {
                "category": category,
                "title": title,
                "content": str(content or ""),
                "updated_at": now,
            }
        )

    payload = {"presets": presets}
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    action = "已覆盖" if updated else "已新增"
    return True, f"{action}文本预设：[{category}] {title}"


def get_text_preset_content(category: str, title: str) -> str | None:
    category = _normalize_category(category)
    title = str(title or "").strip()
    for item in load_text_presets():
        if _normalize_category(item.get("category", "")) == category and item["title"] == title:
            return item["content"]
    return None


def text_preset_api_payload() -> dict[str, Any]:
    presets = load_text_presets()
    categories = list_text_preset_categories()
    return {
        "count": len(presets),
        "default_category": DEFAULT_CATEGORY,
        "placeholder": PRESET_PLACEHOLDER,
        "categories": categories if categories else [DEFAULT_CATEGORY],
        "presets": [
            {
                "category": item["category"],
                "title": item["title"],
                "updated_at": item.get("updated_at", ""),
                "content_preview": item.get("content", "")[:120],
            }
            for item in presets
        ],
    }


class LingliuTextPresetSetter:
    @classmethod
    def INPUT_TYPES(cls):
        categories = list_text_preset_categories() or [DEFAULT_CATEGORY]
        return {
            "required": {
                "preset_category": (categories, {"default": categories[0]}),
                "preset_title": ("STRING", {"default": ""}),
                "preset_content": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "custom_category": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "text")
    FUNCTION = "save_preset"
    CATEGORY = "Lingliu/Text"

    def save_preset(
        self,
        preset_category: str,
        preset_title: str,
        preset_content: str,
        custom_category: str = "",
    ):
        category = str(custom_category or "").strip() or preset_category
        ok, message = save_text_preset(category, preset_title, preset_content)
        if not ok:
            return (message, "")
        return (message, str(preset_content or ""))


class LingliuTextPresetSelector:
    @classmethod
    def INPUT_TYPES(cls):
        categories = list_text_preset_categories() or [DEFAULT_CATEGORY]
        titles = list_all_text_preset_titles()
        preset_values = titles if titles else [PRESET_PLACEHOLDER]
        return {
            "required": {
                "preset_category": (categories, {"default": categories[0]}),
                "preset": (preset_values, {"default": preset_values[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "status")
    FUNCTION = "select_preset"
    CATEGORY = "Lingliu/Text"

    def select_preset(self, preset_category: str, preset: str):
        category = _normalize_category(preset_category)
        preset = str(preset or "").strip()
        if not preset or preset == PRESET_PLACEHOLDER:
            return ("", f"分类“{category}”下暂无可用文本预设。请先运行“灵流文本预设设置”节点保存预设。")

        content = get_text_preset_content(category, preset)
        if content is None:
            return ("", f"未找到文本预设：[{category}] {preset}。请点击选择器节点上的“刷新预设”。")
        return (content, f"已选择文本预设：[{category}] {preset}")


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get(PRESET_API_PATH)
    async def get_lingliu_text_presets(request):
        del request
        return web.json_response(text_preset_api_payload())
except Exception as exc:
    print(f"[comfyui_lingliu_api] text preset API route was not registered: {exc}")
