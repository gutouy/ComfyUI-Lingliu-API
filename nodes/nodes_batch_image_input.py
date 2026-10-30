import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths


SIZE_POLICIES = ["自动适配", "严格一致", "中心裁剪"]
PADDING_COLORS = ["白色", "黑色", "透明转白"]
SORT_MODES = ["上传顺序", "文件名升序"]


@dataclass
class ImageRef:
    name: str
    subfolder: str = ""
    file_type: str = "input"
    original_name: str = ""

    @property
    def display_name(self):
        return self.original_name or self.name

    def annotated_name(self):
        relative_name = self.name
        if self.subfolder:
            relative_name = os.path.join(self.subfolder, self.name)
        tag = self.file_type if self.file_type in {"input", "output", "temp"} else "input"
        return f"{relative_name} [{tag}]"


class LingliuBatchImageInput:
    CATEGORY = "灵流API/工具/图像"
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("图像批次", "图片数量", "文件名列表", "批次信息")
    FUNCTION = "load_images"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片列表": ("STRING", {"default": "[]", "multiline": True}),
                "尺寸策略": (SIZE_POLICIES, {"default": "自动适配"}),
                "目标宽度": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "目标高度": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "补边颜色": (PADDING_COLORS, {"default": "白色"}),
                "排序方式": (SORT_MODES, {"default": "上传顺序"}),
            }
        }

    def parse_image_refs(self, 图片列表):
        text = str(图片列表 or "").strip()
        if not text:
            raise RuntimeError("请先点击加载图像或拖入图片。")
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"图片列表不是有效 JSON：{exc}") from exc

        if isinstance(payload, dict):
            payload = payload.get("images") or payload.get("files") or [payload]
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("请先点击加载图像或拖入图片。")

        refs = []
        for item in payload:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    refs.append(ImageRef(name=name, original_name=os.path.basename(name)))
                continue
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("filename") or "").strip()
            if not name:
                continue
            refs.append(
                ImageRef(
                    name=name,
                    subfolder=str(item.get("subfolder") or "").strip(),
                    file_type=str(item.get("type") or item.get("file_type") or "input").strip(),
                    original_name=str(item.get("original_name") or item.get("originalName") or name).strip(),
                )
            )

        if not refs:
            raise RuntimeError("请先点击加载图像或拖入图片。")
        return refs

    def sort_refs(self, refs, 排序方式):
        if 排序方式 == "文件名升序":
            return sorted(refs, key=lambda ref: ref.display_name.lower())
        return refs

    def resolve_image_path(self, ref):
        return folder_paths.get_annotated_filepath(ref.annotated_name())

    def open_image(self, ref):
        path = self.resolve_image_path(ref)
        if not os.path.exists(path):
            raise RuntimeError(f"找不到图片文件：{ref.display_name}")
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image)
            return self.composite_to_rgb(image)
        except Exception as exc:
            raise RuntimeError(f"图片读取失败：{ref.display_name}：{exc}") from exc

    def composite_to_rgb(self, image):
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            return background
        if image.mode in ("LA", "PA"):
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            return background
        return image.convert("RGB")

    def choose_target_size(self, first_image, 目标宽度, 目标高度):
        width = int(目标宽度 or 0)
        height = int(目标高度 or 0)
        first_width, first_height = first_image.size
        if width <= 0:
            width = first_width
        if height <= 0:
            height = first_height
        if width <= 0 or height <= 0:
            raise RuntimeError("目标宽度和目标高度必须大于 0，或至少提供一张有效图片。")
        return width, height

    def padding_color(self, 补边颜色):
        return "black" if 补边颜色 == "黑色" else "white"

    def fit_auto(self, image, target_size, 补边颜色):
        target_width, target_height = target_size
        scale = min(target_width / image.width, target_height / image.height)
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", target_size, self.padding_color(补边颜色))
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def crop_center(self, image, target_size):
        target_width, target_height = target_size
        scale = max(target_width / image.width, target_height / image.height)
        new_width = max(1, int(round(image.width * scale)))
        new_height = max(1, int(round(image.height * scale)))
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left = max(0, (new_width - target_width) // 2)
        top = max(0, (new_height - target_height) // 2)
        return resized.crop((left, top, left + target_width, top + target_height))

    def process_image(self, image, target_size, 尺寸策略, 补边颜色, ref):
        if 尺寸策略 == "严格一致":
            if image.size != target_size:
                raise RuntimeError(
                    f"图片尺寸不一致：{ref.display_name} 原始尺寸 {image.size[0]}x{image.size[1]}，"
                    f"目标尺寸 {target_size[0]}x{target_size[1]}。请改用自动适配或统一图片尺寸。"
                )
            return image
        if 尺寸策略 == "中心裁剪":
            return self.crop_center(image, target_size)
        return self.fit_auto(image, target_size, 补边颜色)

    def pil_to_tensor(self, image):
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array)[None,]

    def load_images(self, 图片列表, 尺寸策略="自动适配", 目标宽度=0, 目标高度=0, 补边颜色="白色", 排序方式="上传顺序"):
        refs = self.sort_refs(self.parse_image_refs(图片列表), 排序方式)
        images = [(ref, self.open_image(ref)) for ref in refs]
        target_size = self.choose_target_size(images[0][1], 目标宽度, 目标高度)

        tensors = []
        image_info = []
        for ref, image in images:
            processed = self.process_image(image, target_size, 尺寸策略, 补边颜色, ref)
            tensors.append(self.pil_to_tensor(processed))
            image_info.append(
                {
                    "name": ref.display_name,
                    "stored_name": ref.name,
                    "type": ref.file_type,
                    "subfolder": ref.subfolder,
                    "original_size": [image.width, image.height],
                    "output_size": [processed.width, processed.height],
                }
            )

        batch = torch.cat(tensors, dim=0)
        filenames = "\n".join(ref.display_name for ref, _image in images)
        info = {
            "count": len(images),
            "size_policy": 尺寸策略,
            "sort_mode": 排序方式,
            "padding_color": 补边颜色,
            "output_size": [target_size[0], target_size[1]],
            "images": image_info,
        }
        return batch, len(images), filenames, json.dumps(info, ensure_ascii=False, indent=2)
