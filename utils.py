import base64
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36 LingliuComfyUI/1.0"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Encoding": "identity",
    "Connection": "close",
}


def lingliu_temp_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def pil2tensor(image):
    if isinstance(image, list):
        if not image:
            return torch.empty(0)
        return torch.cat([pil2tensor(item) for item in image], dim=0)

    if image.mode != "RGB":
        image = image.convert("RGB")

    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        images = []
        for idx in range(batch_count):
            images.extend(tensor2pil(image[idx]))
        return images

    array = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(array)]


def image_to_png_bytes(image):
    if hasattr(image, "shape"):
        image = tensor2pil(image)[0]
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image):
    png_bytes = image_to_png_bytes(image)
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_image_response(payload, timeout=300):
    images = []
    urls = []
    items = []

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            items = payload["data"]
        elif "b64_json" in payload or "url" in payload:
            items = [payload]

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            raw = base64.b64decode(item["b64_json"])
            images.append(Image.open(BytesIO(raw)).convert("RGB"))
        elif item.get("url"):
            url = item["url"]
            urls.append(url)
            images.append(load_image_from_candidate(url, timeout=timeout))

    return images, urls


def content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(str(text_value))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "")


def blank_image_tensor(width=1024, height=1024, color="white"):
    image = Image.new("RGB", (width, height), color=color)
    return pil2tensor(image)


def guess_mime_type(path_or_url, default="application/octet-stream"):
    mime_type, _ = mimetypes.guess_type(str(path_or_url))
    return mime_type or default


def video_save_options(output, format="auto", codec="auto", bit_depth=None, crf=None, color_space=None, preset=None):
    """Return native save options, or None when the provider bytes can be written as-is.

    ComfyUI SaveVideo passes format/codec/metadata (v0.19), adds crf (v0.30+),
    preset for previews (v0.35+) and VideoContainer/VideoCodec enums (v0.41).
    """
    format_name = str(getattr(format, "value", format) or "auto").lower()
    codec_name = str(getattr(codec, "value", codec) or "auto").lower()
    extras = {
        key: value
        for key, value in (("bit_depth", bit_depth), ("crf", crf), ("color_space", color_space), ("preset", preset))
        if value is not None
    }
    # preset only tunes an encode; it never asks for one.
    if not (set(extras) - {"preset"}):
        # Provider results are MP4 files.
        if format_name in ("auto", "mp4") and codec_name == "auto":
            return None
        # In-memory callers (Concatenate Video, API uploads) read the buffer
        # back with PyAV, so a container-only request must not re-encode the
        # provider's H.264 stream.
        if hasattr(output, "write") and format_name in ("auto", "mp4", "mkv") and codec_name in ("auto", "h264"):
            return None
    return dict(extras, format=format, codec=codec)


def save_video_natively(raw_save, output, options):
    """Fetch the provider bytes, then let ComfyUI convert them as requested."""
    from comfy_api.latest._input_impl.video_types import VideoFromFile

    buffer = BytesIO()
    raw_save(buffer)
    buffer.seek(0)
    # Workflow metadata may contain node API credentials; never embed it.
    VideoFromFile(buffer).save_to(output, metadata=None, **options)
    return True


class LingliuVideoAdapter:
    def __init__(self, video_path_or_url, mime_type=None):
        self.video_path = None
        self.video_url = None
        if isinstance(video_path_or_url, str) and video_path_or_url.startswith(("http://", "https://")):
            self.video_url = video_path_or_url
        else:
            self.video_path = video_path_or_url
        self.mime_type = mime_type or guess_mime_type(video_path_or_url, default="video/mp4")

    def get_dimensions(self):
        # Save Video only needs a valid width/height for output path generation.
        # For remote URLs we can't cheaply inspect the stream here, so return a safe default.
        if self.video_path and os.path.exists(self.video_path):
            try:
                import cv2

                capture = cv2.VideoCapture(self.video_path)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                capture.release()
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                pass
        return 1280, 720

    def _write_bytes(self, output, data):
        if hasattr(output, "write"):
            output.write(data)
            if hasattr(output, "seek"):
                output.seek(0)
            return True
        with open(output, "wb") as handle:
            handle.write(data)
        return True

    def save_to(self, output, format="auto", codec="auto", metadata=None,
                bit_depth=None, crf=None, color_space=None, preset=None, **kwargs):
        del metadata, kwargs
        options = video_save_options(output, format, codec, bit_depth, crf, color_space, preset)
        if options is not None:
            return save_video_natively(self.save_to, output, options)
        if self.video_url:
            response = requests.get(self.video_url, stream=True, timeout=300)
            response.raise_for_status()
            if hasattr(output, "write"):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)
                if hasattr(output, "seek"):
                    output.seek(0)
                return True
            with open(output, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
            return True

        if hasattr(output, "write"):
            with open(self.video_path, "rb") as source:
                shutil.copyfileobj(source, output)
            if hasattr(output, "seek"):
                output.seek(0)
            return True

        shutil.copyfile(self.video_path, output)
        return True


class LingliuNullVideoAdapter:
    def __init__(self, message=""):
        self.message = message
        self.video_url = ""
        self.video_path = ""
        self.mime_type = "video/mp4"

    def get_dimensions(self):
        return 1280, 720

    def save_to(self, output, format="auto", codec="auto", metadata=None, **kwargs):
        del format, codec, metadata, kwargs
        if hasattr(output, "write"):
            output.write(b"")
            if hasattr(output, "seek"):
                output.seek(0)
            return True
        with open(output, "wb") as handle:
            handle.write(b"")
        return True


def _video_bytes_from_video_input(video):
    if video is None:
        return None, None

    if hasattr(video, "video_url") and getattr(video, "video_url"):
        url = getattr(video, "video_url")
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        return response.content, guess_mime_type(url, default="video/mp4")

    if hasattr(video, "video_path") and getattr(video, "video_path"):
        path = getattr(video, "video_path")
        with open(path, "rb") as handle:
            return handle.read(), guess_mime_type(path, default="video/mp4")

    if isinstance(video, str):
        if video.startswith(("http://", "https://")):
            response = requests.get(video, timeout=300)
            response.raise_for_status()
            return response.content, guess_mime_type(video, default="video/mp4")
        if os.path.exists(video):
            with open(video, "rb") as handle:
                return handle.read(), guess_mime_type(video, default="video/mp4")

    if hasattr(video, "save_to"):
        buffer = BytesIO()
        try:
            video.save_to(buffer)
            return buffer.getvalue(), "video/mp4"
        except Exception:
            pass

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=lingliu_temp_dir())
        tmp_path = tmp_file.name
        tmp_file.close()
        try:
            video.save_to(tmp_path)
            with open(tmp_path, "rb") as handle:
                return handle.read(), "video/mp4"
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    raise ValueError("Unsupported VIDEO input; unable to extract video bytes.")


def video_to_data_url(video):
    video_bytes, mime_type = _video_bytes_from_video_input(video)
    if not video_bytes:
        return None
    encoded = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def video_to_remote_url(video):
    if video is None:
        return ""
    if hasattr(video, "video_url") and getattr(video, "video_url"):
        return getattr(video, "video_url")
    if isinstance(video, str) and video.startswith(("http://", "https://")):
        return video
    return ""


def extract_image_candidates_from_text(text):
    if not text:
        return []

    candidates = []

    base64_prefix = "data:image/"
    search_start = 0
    while True:
        start = text.find(base64_prefix, search_start)
        if start < 0:
            break
        end = text.find(")", start)
        whitespace_end = len(text)
        for separator in ["\n", "\r", " ", "\"", "'"]:
            pos = text.find(separator, start)
            if pos != -1:
                whitespace_end = min(whitespace_end, pos)
        if end == -1:
            end = whitespace_end
        else:
            end = min(end, whitespace_end)
        candidates.append(text[start:end])
        search_start = end

    import re

    markdown_matches = re.findall(r"!\[.*?\]\((.*?)\)", text)
    direct_matches = re.findall(r"https?://\S+\.(?:jpg|jpeg|png|gif|webp)", text)
    all_url_matches = re.findall(r"https?://\S+", text)

    for value in markdown_matches + direct_matches + all_url_matches:
        if value not in candidates:
            candidates.append(value)

    return candidates


def load_image_from_candidate(candidate, timeout=300):
    if candidate.startswith("data:image/"):
        _, encoded = candidate.split(",", 1)
        raw = base64.b64decode(encoded)
        return Image.open(BytesIO(raw)).convert("RGB")

    content = download_image_content(candidate, timeout=timeout)
    try:
        return Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:
        preview = content[:120].decode("utf-8", errors="replace").replace("\n", " ")
        raise RuntimeError(f"Downloaded content is not a valid image: {exc}. Preview: {preview}") from exc


def _download_timeout(timeout, read_cap=90):
    try:
        read_timeout = int(timeout)
    except Exception:
        read_timeout = read_cap
    read_timeout = min(max(read_timeout, 30), read_cap)
    return (10, read_timeout)


def _curl_download_image_content(url, timeout=300):
    curl_path = shutil.which("curl")
    if not curl_path:
        raise RuntimeError("curl executable was not found.")

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".img", dir=lingliu_temp_dir())
    output_path = handle.name
    handle.close()
    try:
        try:
            max_time = int(timeout)
        except Exception:
            max_time = 180
        max_time = min(max(max_time, 60), 240)
        command = [
            curl_path,
            "-L",
            "--ipv4",
            "--http1.1",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "15",
            "--max-time",
            str(max_time),
            "--retry",
            "2",
            "--retry-delay",
            "2",
            "-H",
            f"User-Agent: {DOWNLOAD_HEADERS['User-Agent']}",
            "-H",
            f"Accept: {DOWNLOAD_HEADERS['Accept']}",
            "-H",
            "Accept-Encoding: identity",
            "-o",
            output_path,
            url,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=max_time + 20)
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"curl download failed with code {completed.returncode}: {message}")
        with open(output_path, "rb") as source:
            content = source.read()
        if not content:
            raise RuntimeError("curl downloaded empty image content.")
        return content
    finally:
        try:
            os.remove(output_path)
        except Exception:
            pass


def download_image_content(url, timeout=300, max_retries=2):
    lowered_url = str(url or "").lower()
    if "fal.media/" in lowered_url or "v3b.fal.media/" in lowered_url:
        try:
            return _curl_download_image_content(url, timeout=timeout)
        except Exception:
            pass

    last_error = None
    timeout_value = _download_timeout(timeout)
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=timeout_value,
                headers=DOWNLOAD_HEADERS,
            )
            response.raise_for_status()
            expected_length = response.headers.get("Content-Length")
            try:
                expected_length = int(expected_length) if expected_length else None
            except Exception:
                expected_length = None

            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)

            content = b"".join(chunks)
            if expected_length is not None and total < expected_length:
                raise requests.exceptions.ChunkedEncodingError(
                    f"Incomplete image download: {total} bytes read, {expected_length - total} more expected"
                )
            if not content:
                raise requests.exceptions.ContentDecodingError("Downloaded image content is empty.")
            return content
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, int(max_retries)):
                break
            time.sleep(min(2 * attempt, 6))
    try:
        return _curl_download_image_content(url, timeout=timeout)
    except Exception as curl_exc:
        raise RuntimeError(
            f"Failed to download image with Python requests and curl fallback. "
            f"requests_error={last_error}; curl_error={curl_exc}"
        ) from curl_exc


def download_images_in_parallel(urls, timeout=300, max_workers=5):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(download_image_content, url, timeout): (idx, url)
            for idx, url in urls
        }
        for future in as_completed(future_map):
            idx, url = future_map[future]
            results[idx] = (url, future.result())
    return results
