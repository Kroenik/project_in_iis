import re
from typing import Any


DAY_MAP = {
    "monday": "monday",
    "montag": "monday",
    "tuesday": "tuesday",
    "dienstag": "tuesday",
    "wednesday": "wednesday",
    "mittwoch": "wednesday",
    "thursday": "thursday",
    "donnerstag": "thursday",
    "friday": "friday",
    "freitag": "friday",
    "saturday": "saturday",
    "samstag": "saturday",
    "sunday": "sunday",
    "sonntag": "sunday",
}


def _normalize_day(day: str) -> str:
    return DAY_MAP.get(day.strip().lower(), day.strip().lower())


def _parse_hour_range(raw: str) -> tuple[int, int] | None:
    match = re.match(
        r"^\s*(\d{1,2})(?::\d{2})?\s*-\s*(\d{1,2})(?::\d{2})?\s*$", raw
    )
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    if start > 23 or end > 24 or start >= end:
        return None
    return (start, end)


def _hour_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZäöüÄÖÜß]{3,}", text.lower()))


def _zip_score(user_zip: int | None, opp_zip: int | None) -> float:
    """Calculate zip score based on the zip code. When all match highest score
    when third digit matches second highest, when first matches third highest
    as this is the federal state. Else none

    Args:
        user_zip (int | None): _description_
        opp_zip (int | None): _description_

    Returns:
        float: _description_
    """
    if user_zip is None or opp_zip is None:
        return 0.0
    if user_zip == opp_zip:
        return 1.0
    if str(user_zip)[2:3] == str(opp_zip)[2:3]:
        return 0.5
    if str(user_zip)[:1] == str(opp_zip)[:1]:
        return 0.2

    return 0.0


def _availability_score(
    user_availability: dict[str, str], opportunity_schedule: dict[str, str]
) -> float:
    if not user_availability or not opportunity_schedule:
        return 0.0
    overlap_total = 0
    comparisons = 0
    normalized_user = {
        _normalize_day(day): timerange
        for day, timerange in user_availability.items()
    }
    normalized_opp = {
        _normalize_day(day): timerange
        for day, timerange in opportunity_schedule.items()
    }

    for day, user_range in normalized_user.items():
        if day not in normalized_opp:
            continue
        parsed_user = _parse_hour_range(str(user_range))
        parsed_opp = _parse_hour_range(str(normalized_opp[day]))
        if parsed_user is None or parsed_opp is None:
            continue
        comparisons += 1
        overlap_total += _hour_overlap(parsed_user, parsed_opp)

    if comparisons == 0:
        return 0.0
    return min(1.0, overlap_total / (comparisons * 3))


def _keyword_score(
    profile: dict[str, Any], opportunity: dict[str, Any]
) -> float:
    user_text = " ".join(
        [
            str(profile.get("preference") or ""),
            " ".join(profile.get("skills") or []),
            " ".join(profile.get("languages") or []),
        ]
    )
    opp_text = " ".join(
        [
            str(opportunity.get("title") or ""),
            str(opportunity.get("summary") or ""),
            " ".join(opportunity.get("tasks") or []),
            " ".join(opportunity.get("required_skills") or []),
            " ".join(opportunity.get("optional_skills") or []),
        ]
    )
    user_tokens = _tokenize(user_text)
    opp_tokens = _tokenize(opp_text)
    if not user_tokens or not opp_tokens:
        return 0.0
    overlap = len(user_tokens.intersection(opp_tokens))
    return overlap / max(len(user_tokens), 1)


def _skills_score(
    profile: dict[str, Any], opportunity: dict[str, Any]
) -> float:
    user_skills = _tokenize(" ".join(profile.get("skills") or []))
    required = _tokenize(" ".join(opportunity.get("required_skills") or []))
    optional = _tokenize(" ".join(opportunity.get("optional_skills") or []))
    if not user_skills:
        return 0.0
    required_overlap = len(user_skills.intersection(required))
    optional_overlap = len(user_skills.intersection(optional))
    required_score = (
        required_overlap / max(len(required), 1) if required else 0.0
    )
    optional_score = (
        optional_overlap / max(len(optional), 1) if optional else 0.0
    )
    return min(1.0, (required_score * 0.8) + (optional_score * 0.2))


def _recurring_score(
    profile: dict[str, Any], opportunity: dict[str, Any]
) -> float:
    user_pref = profile.get("recurring")
    opp_rec = opportunity.get("recurring")
    if user_pref is None or opp_rec is None:
        return 0.0
    return 1.0 if bool(user_pref) is bool(opp_rec) else -0.2


def score_opportunity(
    profile: dict[str, Any], opportunity: dict[str, Any]
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    skill_score = _skills_score(profile, opportunity)
    keyword_score = _keyword_score(profile, opportunity)
    availability_score = _availability_score(
        profile.get("availability") or {}, opportunity.get("schedule") or {}
    )
    location_score = _zip_score(
        profile.get("zip_code"),
        opportunity.get("zip_code"),
    )
    recurring_score = _recurring_score(profile, opportunity)

    score += skill_score * 0.35
    score += keyword_score * 0.25
    score += availability_score * 0.25
    score += location_score * 0.10
    score += recurring_score * 0.05

    if skill_score > 0.2:
        reasons.append("Your skills align with this opportunity.")
    if availability_score > 0.2:
        reasons.append("The schedule overlaps with your availability.")
    if location_score > 0.4:
        reasons.append("The location is close to your preferred area.")
    if recurring_score > 0:
        reasons.append(
            "This matches your preference for recurring/one-time work."
        )
    if keyword_score > 0.15:
        reasons.append("The topic matches your stated interests.")

    return score, reasons
