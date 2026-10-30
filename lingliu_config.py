import json
import os
import re
from urllib.parse import urljoin

from .upstream_mask import mask_upstream

LINGLIU_PROVIDER = "lingliu"
LINGLIU_BASE_URL = "https://api.lingliuapi.com"

PROVIDER_DISPLAY_LABELS = {
    LINGLIU_PROVIDER: "灵流",
}
PROVIDER_ALIAS_TO_VALUE = {
    LINGLIU_PROVIDER: LINGLIU_PROVIDER,
    "灵流": LINGLIU_PROVIDER,
    "灵流线路": LINGLIU_PROVIDER,
}
PUBLIC_PROVIDER_CHOICES = [LINGLIU_PROVIDER]
PUBLIC_PROVIDER_DISPLAY_CHOICES = [PROVIDER_DISPLAY_LABELS[LINGLIU_PROVIDER]]
PROVIDER_CHOICES = [LINGLIU_PROVIDER]

PROVIDER_DEFAULTS = {
    LINGLIU_PROVIDER: {
        "base_url": LINGLIU_BASE_URL,
        "chat_path": "/v1/chat/completions",
        "images_generate_path": "/v1/images/generations",
        "images_edit_path": "/v1/images/edits",
        "models_path": "/v1/models",
        "images_task_status_path_template": "/v1/images/tasks/{task_id}",
        "videos_generate_path": "/v2/videos/generations",
        "videos_status_path_template": "/v2/videos/generations/{task_id}",
    },
}

CONFIG_FILENAME = "lingliu_api.json"
PRIVATE_R2_CONFIG_FILENAME = "cloudflare_r2.json"

DEFAULT_TIMEOUT = 300
R2_CONFIG_KEYS = (
    "r2_account_id",
    "r2_access_key_id",
    "r2_secret_access_key",
    "r2_bucket_name",
    "r2_public_base_url",
    "r2_object_prefix",
)

RUNTIME_SETTINGS = {
    "api_key": "",
    "timeout": None,
    "api_provider": "",
    "r2_account_id": "",
    "r2_access_key_id": "",
    "r2_secret_access_key": "",
    "r2_bucket_name": "",
    "r2_public_base_url": "",
    "r2_object_prefix": "",
}


def normalize_provider(provider):
    # The public edition has a single route; any stored value maps to it.
    return LINGLIU_PROVIDER


def provider_display_name(provider):
    return PROVIDER_DISPLAY_LABELS[LINGLIU_PROVIDER]


DATA_URI_PATTERN = re.compile(
    r"data:(?P<mime>(?:image|video|audio|application)/[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=_-]{128,})",
    re.IGNORECASE,
)


def redact_large_data_uris(value):
    text = str(value)

    def replace(match):
        mime = match.group("mime")
        data_length = len(match.group("data") or "")
        return f"data:{mime};base64,base64_data<{data_length} chars>"

    return DATA_URI_PATTERN.sub(replace, text)


def mask_provider_text(value):
    text = redact_large_data_uris(value)
    for source in ("https://api.lingliuapi.com", "http://api.lingliuapi.com", "api.lingliuapi.com"):
        text = text.replace(source, PROVIDER_DISPLAY_LABELS[LINGLIU_PROVIDER])
    return mask_upstream(text)


def mask_provider_payload(value):
    if isinstance(value, dict):
        return {mask_provider_text(key): mask_provider_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_provider_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_provider_payload(item) for item in value)
    if isinstance(value, str):
        return mask_provider_text(value)
    return value


def provider_defaults(provider=None):
    defaults = dict(PROVIDER_DEFAULTS[LINGLIU_PROVIDER])
    defaults["api_provider"] = LINGLIU_PROVIDER
    defaults["api_key"] = ""
    defaults["timeout"] = DEFAULT_TIMEOUT
    return defaults


def _config_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), CONFIG_FILENAME)


def r2_private_config_path():
    appdata_dir = os.environ.get("APPDATA", "").strip()
    if appdata_dir:
        return os.path.join(appdata_dir, "LingliuAPI", PRIVATE_R2_CONFIG_FILENAME)
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, ".lingliu_api", PRIVATE_R2_CONFIG_FILENAME)


def _r2_private_config_paths():
    paths = []
    env_path = os.environ.get("LINGLIU_R2_CONFIG", "").strip()
    if env_path:
        paths.append(env_path)
    paths.append(r2_private_config_path())
    package_local_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "lingliu_r2_private.json",
    )
    paths.append(package_local_path)

    unique_paths = []
    seen = set()
    for path in paths:
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized not in seen:
            unique_paths.append(path)
            seen.add(normalized)
    return unique_paths


def _read_config_file():
    path = _config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _read_r2_private_config():
    for path in _r2_private_config_paths():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                continue
            return {
                key: str(loaded.get(key) or "").strip()
                for key in R2_CONFIG_KEYS
                if loaded.get(key) is not None
            }
        except Exception:
            continue
    return {}


def _write_config_file(payload):
    with open(_config_path(), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def get_config():
    stored_timeout = DEFAULT_TIMEOUT
    loaded = _read_config_file()
    private_r2 = _read_r2_private_config()
    if loaded.get("timeout") is not None:
        try:
            stored_timeout = int(loaded["timeout"])
        except Exception:
            stored_timeout = DEFAULT_TIMEOUT

    config = provider_defaults(LINGLIU_PROVIDER)
    config["api_key"] = RUNTIME_SETTINGS.get("api_key", "") or ""
    runtime_timeout = RUNTIME_SETTINGS.get("timeout")
    config["timeout"] = int(runtime_timeout if runtime_timeout is not None else stored_timeout)
    for key in R2_CONFIG_KEYS:
        config[key] = RUNTIME_SETTINGS.get(key, "") or private_r2.get(key, "") or loaded.get(key, "") or ""
    return config


def save_config(config):
    timeout = int((config or {}).get("timeout", DEFAULT_TIMEOUT))
    merged = provider_defaults(LINGLIU_PROVIDER)
    merged["timeout"] = timeout
    stored = _read_config_file()
    stored.update({"api_provider": LINGLIU_PROVIDER, "timeout": timeout})
    _write_config_file(stored)
    return merged


def save_r2_config(config):
    stored = _read_config_file()
    for key in R2_CONFIG_KEYS:
        if key in (config or {}):
            stored[key] = str((config or {}).get(key) or "").strip()
            RUNTIME_SETTINGS[key] = stored[key]
    _write_config_file(stored)
    return get_config()


def set_runtime_config(
    api_key=None,
    timeout=None,
    api_provider=None,
    r2_account_id=None,
    r2_access_key_id=None,
    r2_secret_access_key=None,
    r2_bucket_name=None,
    r2_public_base_url=None,
    r2_object_prefix=None,
):
    if api_key is not None:
        RUNTIME_SETTINGS["api_key"] = api_key.strip()
    if timeout is not None:
        RUNTIME_SETTINGS["timeout"] = int(timeout)
    if api_provider is not None:
        RUNTIME_SETTINGS["api_provider"] = normalize_provider(api_provider)
    if r2_account_id is not None:
        RUNTIME_SETTINGS["r2_account_id"] = r2_account_id.strip()
    if r2_access_key_id is not None:
        RUNTIME_SETTINGS["r2_access_key_id"] = r2_access_key_id.strip()
    if r2_secret_access_key is not None:
        RUNTIME_SETTINGS["r2_secret_access_key"] = r2_secret_access_key.strip()
    if r2_bucket_name is not None:
        RUNTIME_SETTINGS["r2_bucket_name"] = r2_bucket_name.strip()
    if r2_public_base_url is not None:
        RUNTIME_SETTINGS["r2_public_base_url"] = r2_public_base_url.strip()
    if r2_object_prefix is not None:
        RUNTIME_SETTINGS["r2_object_prefix"] = r2_object_prefix.strip()
    return get_config()


def build_url(base_url, path):
    base_url = (base_url or "").strip()
    path = (path or "").strip()
    if not path:
        return base_url.rstrip("/")
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not base_url:
        base_url = LINGLIU_BASE_URL
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def lingliu_channel_key(api_key, channel_id):
    key = str(api_key or "").strip()
    channel = str(channel_id or "").strip()
    if not key or not channel or not channel.isdigit():
        return key
    if not key.lower().startswith("sk-"):
        return key
    parts = key[3:].split("-")
    if len(parts) > 1 and parts[-1].isdigit():
        return key
    return f"{key}-{channel}"
