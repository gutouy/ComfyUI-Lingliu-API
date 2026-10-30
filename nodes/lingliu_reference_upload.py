"""Direct-to-R2 reference upload for Lingliu image submits.

The platform used to receive reference images inside the multipart submit and
forward the whole body abroad before it could return a task ID; on busy
afternoons that took minutes.  Here each reference is PUT straight to R2 with a
console-signed URL and the submit only carries public URLs.  Any failure
returns None so the caller keeps its original multipart submit.

Soul (V17.6) has no multipart submit, so it uses upload_references_strict,
which raises instead, and paces presign under the console's per-token limit
(60/minute, a reused image counts too) so 100 training images do not hit 429.
"""
import asyncio
import hashlib
import threading
import time

import requests

from ..lingliu_config import build_url

PRESIGN_PATH = "/v1/files/reference-presign"
MAX_REFERENCE_BYTES = 64 * 1024 * 1024
PUT_ATTEMPTS = 3
PUT_CONNECT_TIMEOUT = 30
PUT_CONCURRENCY = 3
UNAVAILABLE_SECONDS = 600
PROXY_HINT = "如开启了代理或 VPN，请把 r2.cloudflarestorage.com 设为直连。"
PRESIGN_PER_MINUTE = 50
PRESIGN_WINDOW_SECONDS = 60
RATE_LIMIT_WAIT_SECONDS = 15
RATE_LIMIT_ATTEMPTS = 6

# base_url -> monotonic time until which presign is skipped (503/404: not deployed yet).
_unavailable_until = {}
# (base_url, api_key) -> PresignGate; the console limit is per token.
_gates = {}
_gates_lock = threading.Lock()
_sleep = time.sleep


class ReferenceUploadError(RuntimeError):
    pass


class ReferenceRateLimited(ReferenceUploadError):
    pass


class PresignGate:
    """Sliding window: at most `limit` presign calls per `window` seconds."""

    def __init__(self, limit=PRESIGN_PER_MINUTE, window=PRESIGN_WINDOW_SECONDS, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.calls = []
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = self.clock()
                self.calls = [t for t in self.calls if now - t < self.window]
                if len(self.calls) < self.limit:
                    self.calls.append(now)
                    return
                delay = self.window - (now - self.calls[0])
            _sleep(max(delay, 0.05))


def presign_gate(base_url, api_key):
    with _gates_lock:
        return _gates.setdefault((str(base_url or "").rstrip("/"), str(api_key or "")), PresignGate())


def log(message):
    print(f"[comfyui_lingliu_api] {message}", flush=True)


def _mb(size):
    return f"{size / 1048576:.1f}MB"


def presign_reference(base_url, api_key, data, mime="image/png"):
    body = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mime": mime}
    try:
        response = requests.post(
            build_url(base_url, PRESIGN_PATH),
            json=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=(10, 30),
        )
    except requests.RequestException as exc:
        raise ReferenceUploadError(f"申请上传地址失败（{type(exc).__name__}）") from exc
    if response.status_code in (404, 503):
        _unavailable_until[base_url] = time.monotonic() + UNAVAILABLE_SECONDS
    if response.status_code == 429:
        raise ReferenceRateLimited("申请上传地址失败（HTTP 429）")
    if response.status_code != 200:
        raise ReferenceUploadError(f"申请上传地址失败（HTTP {response.status_code}）")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ReferenceUploadError("申请上传地址返回格式不正确") from exc
    url = str(payload.get("url") or "")
    if not url.startswith("https://"):
        raise ReferenceUploadError("申请上传地址未返回公网 URL")
    if not payload.get("reused") and not str(payload.get("put_url") or "").startswith("https://"):
        raise ReferenceUploadError("申请上传地址未返回 put_url")
    return payload


def put_reference(ticket, data):
    """PUT the exact bytes with the signed headers; retry only transport failures and 5xx."""
    headers = {str(k): str(v) for k, v in (ticket.get("headers") or {}).items()}
    read_timeout = 120 + len(data) // 50000
    last = ""
    for attempt in range(1, PUT_ATTEMPTS + 1):
        try:
            response = requests.put(ticket["put_url"], data=data, headers=headers,
                                    timeout=(PUT_CONNECT_TIMEOUT, read_timeout))
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = type(exc).__name__
            continue
        if response.status_code in (200, 201):
            return attempt
        last = f"HTTP {response.status_code}"
        if response.status_code < 500:
            break
    raise ReferenceUploadError(f"直传失败（{last}）")


def _paced_presign(base_url, api_key, data, mime, pace):
    """Without pace a 429 fails at once; with pace wait for the window and retry (presign is free)."""
    gate = presign_gate(base_url, api_key) if pace else None
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        if gate:
            gate.wait()
        try:
            return presign_reference(base_url, api_key, data, mime)
        except ReferenceRateLimited:
            if not gate or attempt == RATE_LIMIT_ATTEMPTS:
                raise
            log(f"上传地址申请过于频繁，等待 {RATE_LIMIT_WAIT_SECONDS}s 后继续")
            _sleep(RATE_LIMIT_WAIT_SECONDS)


async def _upload_all(base_url, api_key, blobs, mime, label, pace):
    started = time.perf_counter()
    gate = asyncio.Semaphore(PUT_CONCURRENCY)
    by_digest = {}
    done = [0]
    total = len({hashlib.sha256(data).hexdigest() for data in blobs})

    async def one(index, data):
        async with gate:
            t0 = time.perf_counter()
            ticket = await asyncio.to_thread(_paced_presign, base_url, api_key, data, mime, pace)
            if ticket.get("reused"):
                done[0] += 1
                progress = f"（已传 {done[0]}/{total}）" if pace else ""
                log(f"{label} {index}: {_mb(len(data))} 已存在，直接复用{progress}")
                return ticket["url"]
            attempts = await asyncio.to_thread(put_reference, ticket, data)
            elapsed = max(time.perf_counter() - t0, 0.001)
            retry = f"，重试 {attempts - 1} 次" if attempts > 1 else ""
            done[0] += 1
            progress = f"，已传 {done[0]}/{total}" if pace else ""
            log(f"{label} {index}: {_mb(len(data))} 直传 {elapsed:.1f}s（{len(data) / 1048576 / elapsed:.2f}MB/s{retry}{progress}）")
            return ticket["url"]

    try:
        for index, data in enumerate(blobs, start=1):
            digest = hashlib.sha256(data).hexdigest()
            if digest not in by_digest:
                by_digest[digest] = asyncio.ensure_future(one(index, data))
        urls = await asyncio.gather(*(by_digest[hashlib.sha256(data).hexdigest()] for data in blobs))
    except BaseException:
        for task in by_digest.values():
            task.cancel()
        raise
    log(f"{label}共 {len(blobs)} 张直传完成，用时 {time.perf_counter() - started:.1f}s")
    return list(urls)


async def upload_references(base_url, api_key, blobs, mime="image/png"):
    """Return one public URL per blob, or None when any reference cannot go direct."""
    if not blobs:
        return None
    if time.monotonic() < _unavailable_until.get(base_url, 0):
        return None
    if any(not data or len(data) > MAX_REFERENCE_BYTES for data in blobs):
        log("参考图超过 64MB，改用原方式提交。")
        return None
    try:
        return await _upload_all(base_url, api_key, blobs, mime, "参考图", pace=False)
    except ReferenceUploadError as exc:
        hint = PROXY_HINT if "直传失败" in str(exc) else ""
        log(f"参考图{exc}，改用原方式提交。{hint}")
        return None


async def upload_references_strict(base_url, api_key, blobs, mime="image/jpeg", label="参考图"):
    """Return one public URL per blob or raise ReferenceUploadError; there is no fallback."""
    if not blobs:
        raise ReferenceUploadError("没有可上传的图片")
    for index, data in enumerate(blobs, start=1):
        if not data or len(data) > MAX_REFERENCE_BYTES:
            raise ReferenceUploadError(f"第 {index} 张{label}为空或超过 64MB")
    if time.monotonic() < _unavailable_until.get(base_url, 0):
        raise ReferenceUploadError("当前线路暂不支持图片直传，请稍后再试")
    try:
        return await _upload_all(base_url, api_key, blobs, mime, label, pace=True)
    except ReferenceUploadError as exc:
        if "直传失败" in str(exc):
            raise ReferenceUploadError(f"{exc}。{PROXY_HINT}") from exc
        raise
