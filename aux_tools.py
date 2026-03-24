from typing import Any
import json
import re

PROFILE_ALIASES: dict[str, list[str]] = {
    "name": ["name", "name_text", "full_name", "full_name_text"],
    "contact": ["contact", "contact_text", "email", "email_text"],
    "city": ["city", "city_text"],
    "zip_code": ["zip_code", "zip_int", "zip"],
    "radius": ["radius", "radius_int"],
    "availability": ["availability", "availability_json"],
    "skills": ["skills", "skills_json"],
    "languages": ["languages", "languages_text"],
    "h_week": ["h_week", "h_week_int", "hours_week", "hours_week_int"],
    "start_date": ["start_date"],
    "end_date": ["end_date"],
    "recurring": ["recurring", "recurring_bool"],
    "preference": ["preference", "preference_text", "preference_summary"],
    "preference_embedding": ["preference_embedding"],
}

UPDATE_FIELD_ALIASES: dict[str, list[str]] = {
    "name": ["name", "full name"],
    "contact": ["contact", "email", "phone"],
    "city": ["city", "location", "place"],
    "zip_code": ["zip", "zip code", "postal", "postal code"],
    "radius": ["radius", "distance"],
    "availability": ["availability", "time", "schedule", "days"],
    "skills": ["skills", "skill"],
    "languages": ["languages", "language"],
    "h_week": ["hours", "hours per week", "weekly hours"],
    "start_date": ["start date", "begin date", "from date"],
    "end_date": ["end date", "until date"],
    "recurring": ["recurring", "regular", "one-time", "one time"],
    "preference": ["preference", "interest", "goal", "cause"],
}


OPPORTUNITY_ALIASES: dict[str, list[str]] = {
    "organization": ["org", "org_text", "organization", "organization_text"],
    "title": ["title", "title_text", "opp_title", "opp_title_text"],
    "summary": [
        "summary",
        "summary_text",
        "short_summary",
        "short_summary_text",
    ],
    "tasks": [
        "tasks",
        "tasks_text",
        "responsibilities",
        "responsibilities_text",
    ],
    "required_skills": ["required_skills", "required_skills_text"],
    "optional_skills": ["optional_skills", "optional_skills_text"],
    "languages": [
        "language_requirements",
        "language_requirements_text",
        "languages",
    ],
    "amount_volunteers": ["amount_volunteers", "amount_volunteers_int"],
    "schedule": ["schedule", "schedule_json"],
    "hours_week": ["hours_week", "hours_week_int"],
    "recurring": ["recurring", "recurring_bool"],
    "zip_code": ["zip_code", "zip_int", "zip"],
    "city": ["city", "city_text"],
    "email": ["email", "email_text", "contact", "contact_text"],
    "embedding": ["embedding"],
}


def _coalesce(row: dict[str, Any], aliases: list[str]) -> Any:
    for key in aliases:
        if key in row and row[key] is not None:
            return row[key]
    return None


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "ja", "1", "recurring", "regular"}:
        return True
    if text in {"false", "no", "n", "nein", "0", "one-time", "one time"}:
        return False
    return None


def safe_profile_lookup(
    client: Any, user_id: str | int
) -> dict[str, Any] | None:
    user_candidates: list[str | int] = [user_id]
    if isinstance(user_id, str) and user_id.isdigit():
        user_candidates.append(int(user_id))

    for candidate in user_candidates:
        try:
            result = (
                client.table("volunteer_profiles")
                .select("*")
                .eq("user_id", candidate)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]
        except Exception:
            continue
    return None


def to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                data = json.loads(stripped)
                if isinstance(data, list):
                    return [
                        str(item).strip() for item in data if str(item).strip()
                    ]
            except json.JSONDecodeError:
                pass
        return [
            part.strip()
            for part in re.split(r"[,;/\n]", stripped)
            if part.strip()
        ]
    return [str(value).strip()]


def to_dict(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass
    return {}


def normalize_profile(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _coalesce(row, PROFILE_ALIASES["name"]),
        "contact": _coalesce(row, PROFILE_ALIASES["contact"]),
        "city": _coalesce(row, PROFILE_ALIASES["city"]),
        "zip_code": to_int(_coalesce(row, PROFILE_ALIASES["zip_code"])),
        "radius": to_int(_coalesce(row, PROFILE_ALIASES["radius"])),
        "availability": to_dict(
            _coalesce(row, PROFILE_ALIASES["availability"])
        ),
        "skills": to_list(_coalesce(row, PROFILE_ALIASES["skills"])),
        "languages": to_list(_coalesce(row, PROFILE_ALIASES["languages"])),
        "h_week": to_int(_coalesce(row, PROFILE_ALIASES["h_week"])),
        "start_date": _coalesce(row, PROFILE_ALIASES["start_date"]),
        "end_date": _coalesce(row, PROFILE_ALIASES["end_date"]),
        "recurring": to_bool(_coalesce(row, PROFILE_ALIASES["recurring"])),
        "preference": _coalesce(row, PROFILE_ALIASES["preference"]),
        "preference_embedding": _coalesce(
            row, PROFILE_ALIASES["preference_embedding"]
        ),
    }


def normalize_opportunity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "organization": _coalesce(row, OPPORTUNITY_ALIASES["organization"]),
        "title": _coalesce(row, OPPORTUNITY_ALIASES["title"]),
        "summary": _coalesce(row, OPPORTUNITY_ALIASES["summary"]),
        "tasks": to_list(_coalesce(row, OPPORTUNITY_ALIASES["tasks"])),
        "required_skills": to_list(
            _coalesce(row, OPPORTUNITY_ALIASES["required_skills"])
        ),
        "optional_skills": to_list(
            _coalesce(row, OPPORTUNITY_ALIASES["optional_skills"])
        ),
        "languages": to_list(_coalesce(row, OPPORTUNITY_ALIASES["languages"])),
        "amount_volunteers": to_int(
            _coalesce(row, OPPORTUNITY_ALIASES["amount_volunteers"])
        ),
        "schedule": to_dict(_coalesce(row, OPPORTUNITY_ALIASES["schedule"])),
        "hours_week": to_int(
            _coalesce(row, OPPORTUNITY_ALIASES["hours_week"])
        ),
        "recurring": to_bool(_coalesce(row, OPPORTUNITY_ALIASES["recurring"])),
        "zip_code": to_int(_coalesce(row, OPPORTUNITY_ALIASES["zip_code"])),
        "city": _coalesce(row, OPPORTUNITY_ALIASES["city"]),
        "email": _coalesce(row, OPPORTUNITY_ALIASES["email"]),
        "embedding": _coalesce(row, OPPORTUNITY_ALIASES["embedding"]),
    }


def safe_get_all_opportunities(client: Any) -> list[dict[str, Any]]:
    try:
        result = client.table("opportunities").select("*").execute()
        return result.data or []
    except Exception:
        return []


def normalize_update_field(field: str) -> str | None:
    lower = field.strip().lower()
    for canonical, aliases in UPDATE_FIELD_ALIASES.items():
        if lower == canonical or lower in aliases:
            return canonical
    return None
