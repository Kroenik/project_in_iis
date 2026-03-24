import re
from scoring import _parse_hour_range
from typing import Any
from aux_tools import PROFILE_ALIASES
from aux import get_embedding
from openai import OpenAI
import os


def parse_availability_update(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in re.split(r"[;,]", value):
        if ":" not in part:
            continue
        day, timerange = part.split(":", 1)
        day_name = day.strip()
        time_value = timerange.strip()
        if not day_name or not time_value:
            continue
        if _parse_hour_range(time_value) is None:
            continue
        parsed[day_name] = time_value
    return parsed


def detect_profile_column(row: dict[str, Any], canonical_field: str) -> str:
    aliases = PROFILE_ALIASES.get(canonical_field, [canonical_field])
    for alias in aliases:
        if alias in row:
            return alias
    return aliases[0]


def _build_openai_client() -> OpenAI | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def maybe_create_embedding(text: str) -> list[float] | None:
    if not text.strip():
        return None
    client = _build_openai_client()
    if client is None:
        return None
    try:
        return get_embedding(client, text.strip())
    except Exception:
        return None
