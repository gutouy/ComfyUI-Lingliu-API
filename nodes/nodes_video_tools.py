import json
import os
import re
import shutil
import subprocess
import time
from fractions import Fraction

from comfy.comfy_types import IO

from ..utils import LingliuNullVideoAdapter, LingliuVideoAdapter, lingliu_temp_dir


class LingliuVideoConcat:
    CATEGORY = "灵流API/工具/视频"
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING")
    RETURN_NAMES = ("拼接视频", "视频路径", "拼接信息", "错误信息")
    FUNCTION = "concat"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频1": (IO.VIDEO,),
                "视频2": (IO.VIDEO,),
                "输出文件名前缀": ("STRING", {"default": "lingliu_concat"}),
            }
        }

    def concat(self, 视频1, 视频2, 输出文件名前缀="lingliu_concat"):
        input_paths = []
        normalized_paths = []
        concat_list_path = ""
        task_stamp = f"{int(time.time() * 1000)}_{os.getpid()}"
        try:
            ffmpeg = self._require_binary("ffmpeg")
            ffprobe = self._require_binary("ffprobe")
            if 视频1 is None:
                raise RuntimeError("视频1不能为空。")
            if 视频2 is None:
                raise RuntimeError("视频2不能为空。")

            temp_dir = lingliu_temp_dir()
            input1_path = os.path.join(temp_dir, f"concat_input1_{task_stamp}.mp4")
            input2_path = os.path.join(temp_dir, f"concat_input2_{task_stamp}.mp4")
            self._save_video_input(视频1, input1_path, "视频1")
            self._save_video_input(视频2, input2_path, "视频2")
            input_paths.extend([input1_path, input2_path])

            info1 = self._probe_video(ffprobe, input1_path)
            info2 = self._probe_video(ffprobe, input2_path)
            target_width = info1["width"]
            target_height = info1["height"]
            target_fps = info1["fps"] or "30"
            include_audio = bool(info1["has_audio"] or info2["has_audio"])

            normalized1 = os.path.join(temp_dir, f"concat_norm1_{task_stamp}.mp4")
            normalized2 = os.path.join(temp_dir, f"concat_norm2_{task_stamp}.mp4")
            self._normalize_video(
                ffmpeg,
                input1_path,
                normalized1,
                info1,
                target_width,
                target_height,
                target_fps,
                include_audio,
            )
            self._normalize_video(
                ffmpeg,
                input2_path,
                normalized2,
                info2,
                target_width,
                target_height,
                target_fps,
                include_audio,
            )
            normalized_paths.extend([normalized1, normalized2])

            prefix = self._safe_prefix(输出文件名前缀)
            output_path = os.path.join(temp_dir, f"{prefix}_{task_stamp}.mp4")
            concat_list_path = os.path.join(temp_dir, f"concat_list_{task_stamp}.txt")
            self._write_concat_list(concat_list_path, normalized_paths)
            self._run_command(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_list_path,
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    output_path,
                ],
                "拼接视频失败",
            )

            info = {
                "target_width": target_width,
                "target_height": target_height,
                "target_fps": target_fps,
                "has_audio": include_audio,
                "output_path": output_path,
                "inputs": [info1, info2],
            }
            return LingliuVideoAdapter(output_path, mime_type="video/mp4"), output_path, self._json(info), ""
        except Exception as exc:
            message = str(exc)
            return LingliuNullVideoAdapter(message), "", self._json({"error": message}), message
        finally:
            for path in input_paths + normalized_paths + ([concat_list_path] if concat_list_path else []):
                self._remove_file(path)

    def _require_binary(self, name):
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f"未找到 {name}，请先安装 ffmpeg 并确保它在 PATH 中。")
        return path

    def _save_video_input(self, video, output_path, label):
        try:
            if not hasattr(video, "save_to"):
                raise RuntimeError(f"{label}不是有效的 VIDEO 输入。")
            video.save_to(output_path)
        except Exception as exc:
            raise RuntimeError(f"{label}保存为临时视频失败：{exc}") from exc
        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
            raise RuntimeError(f"{label}保存后为空文件。")

    def _probe_video(self, ffprobe, path):
        result = self._run_command(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                "format=duration:stream=codec_type,width,height,avg_frame_rate,r_frame_rate",
                path,
            ],
            "读取视频信息失败",
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
        if not video_stream:
            raise RuntimeError("输入文件不包含视频轨。")
        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("无法读取视频分辨率。")
        duration = float((payload.get("format") or {}).get("duration") or 0)
        if duration <= 0:
            duration = 1.0
        return {
            "path": path,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": self._normalize_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
            "has_audio": any(item.get("codec_type") == "audio" for item in streams),
        }

    def _normalize_video(self, ffmpeg, input_path, output_path, info, width, height, fps, include_audio):
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={fps},format=yuv420p"
        )
        command = [ffmpeg, "-y", "-i", input_path]
        if include_audio and not info["has_audio"]:
            command += [
                "-f",
                "lavfi",
                "-t",
                f"{info['duration']:.6f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
            ]

        command += ["-map", "0:v:0", "-vf", video_filter, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        if include_audio:
            command += ["-map", "0:a:0" if info["has_audio"] else "1:a:0", "-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            command += ["-an"]
        command += ["-movflags", "+faststart", output_path]
        self._run_command(command, "规整视频失败")

    def _write_concat_list(self, path, video_paths):
        with open(path, "w", encoding="utf-8") as handle:
            for item in video_paths:
                safe = os.path.abspath(item).replace("\\", "/").replace("'", r"'\''")
                handle.write(f"file '{safe}'\n")

    def _normalize_fps(self, value):
        raw = str(value or "").strip()
        if not raw or raw == "0/0":
            return "30"
        try:
            fps = Fraction(raw)
            if fps <= 0:
                return "30"
            if fps.denominator == 1:
                return str(fps.numerator)
            return f"{fps.numerator}/{fps.denominator}"
        except Exception:
            return "30"

    def _safe_prefix(self, value):
        raw = str(value or "").strip() or "lingliu_concat"
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("._-")
        return safe or "lingliu_concat"

    def _run_command(self, command, label):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            if len(stderr) > 1200:
                stderr = stderr[-1200:]
            raise RuntimeError(f"{label}：退出码 {result.returncode}。{stderr}")
        return result

    def _json(self, payload):
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _remove_file(self, path):
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
