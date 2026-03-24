from typing import Any
from langchain.tools import tool, ToolRuntime
import os
from openai import OpenAI
from aux import get_embedding, get_profile, get_opportunities
from custom_classes import Context
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain_openai import ChatOpenAI
from datetime import date
from pydantic import BaseModel, Field
import re
from aux_tools import (
    safe_profile_lookup,
    normalize_profile,
    safe_get_all_opportunities,
    normalize_opportunity,
)


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


@tool
def get_volunteer_information(runtime: ToolRuntime[Context]):
    """Retrieve the currently logged-in volunteer prfile. Use once at start"""
    profile_row = safe_profile_lookup(
        runtime.context.supabase, runtime.context.user_id
    )
    if profile_row is None:
        return (
            "No volunteer profile exists yet for this account. "
            "Please guide the user through profile setup"
        )

    profile = normalize_profile(profile_row)
    profile.pop("preference_embedding", None)

    return {
        key: value
        for key, value in profile.items()
        if value not in (None, "", [], {})
    }


@tool
def get_opportunities_for_volunteer(runtime: ToolRuntime[Context]):
    """Return best-matching opportunities based on profile, skills, location,
    and time."""
    print(f"Getting matches for volunteer ID: {runtime.context.user_id}")
    profile_row = safe_profile_lookup(
        client=runtime.context.supabase, user_id=runtime.context.user_id
    )

    if profile_row is None:
        return (
            "No profile found yet, so matching cannot run. "
            "Please create the profile first."
        )

    profile = normalize_profile(profile_row)
    opportunities = safe_get_all_opportunities(runtime.context.supabase)
    if not opportunities:
        return "No volunteering opportunities are available at the moment."

    ranked: list[tuple[float, dict[str, Any], list[str]]] = []
    for row in opportunities:
        normalized = normalize_opportunity(row)
        score, reasons = score_opportunity(profile, normalized)
        if score <= 0:
            continue
        ranked.append((score, normalized, reasons))

    if not ranked:
        return (
            "I could not find a strong match yet. Ask the user for more details "
            "about interests, preferred schedule, and location radius."
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_matches = ranked[:5]
    response: list[dict[str, Any]] = []
    for score, opp, reasons in top_matches:
        response.append(
            {
                "id": opp.get("id"),
                "organization": opp.get("organization"),
                "title": opp.get("title"),
                "summary": opp.get("summary"),
                "schedule": opp.get("schedule"),
                "hours_week": opp.get("hours_week"),
                "recurring": opp.get("recurring"),
                "zip_code": opp.get("zip_code"),
                "contact_email": opp.get("email"),
                "match_score": round(score, 3),
                "why_good_fit": reasons
                or ["This opportunity is generally aligned with the profile."],
            }
        )
    return response


@tool
def get_opportunity_details(
    runtime: ToolRuntime[Context], opportunity_id: int
):
    """Retrieve the details of a specific opportunity"""
    print(f"Getting details for opportunity ID: {opportunity_id}")
    supabase_client = runtime.context.supabase
    result = (
        supabase_client.table("opportunities")
        .select("*")
        .eq("id", opportunity_id)
        .execute()
    )
    return result.data


@tool
def update_volunteer_profile(
    runtime: ToolRuntime[Context], user_input: str
) -> str:
    """
    Updates a field in the user's profile based on a natural language description.
    Use this when the user wants to update any personal information.
    Only use this when a user already has a profile.
    """
    print(f"Updating volunteer profile for user ID: {runtime.context.user_id}")
    column_names = [
        "city",
        "zip_code",
        "radius",
        "availability",
        "skills",
        "languages",
        "h_week",
        "start_date",
        "end_date",
        "recurring",
        "preference",
    ]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, timeout=10)
    prompt = f"""You are helping update a user profile in a database.

            Available columns in the user_profiles table:
            {", ".join(column_names)}

            The user said: "{user_input}"

            Your job:
            1. Identify which column the user wants to update
            (match by meaning, not exact wording)
            2. Extract the value they want to set

            Respond in this exact JSON format with no extra text:
            {{"column": "<exact_column_name>", "value": "<extracted_value>"}}

            If you cannot confidently identify the column, respond:
            {{"column": null, "value": null, "reason": "<why you couldn't match>"}}"""
    response = llm.invoke(prompt)
    parsed = json.loads(response.content)

    if not parsed.get("column"):
        return f"""I could not determine which column to update: {
            parsed.get("reason")
        }"""

    column = parsed.get("column")
    value = parsed.get("value")

    if column not in column_names:
        return f"Identified column {column} is not valid. Please try again."

    supabase_client = runtime.context.supabase
    # arrays
    if column in ["skills", "languages"]:
        current_col_vals = (
            supabase_client.table("volunteer_profiles")
            .select(f"{column}")
            .eq("user_id", runtime.context.user_id)
            .execute()
        )

        if current_col_vals.data == []:
            return f"""The user does not yet have a {column} in their profile.
            Please ask them to create one."""
        else:
            updated_col_vals = current_col_vals.data[0][column] + [value]

        result = (
            supabase_client.table("volunteer_profiles")
            .update({column: updated_col_vals})
            .eq("user_id", runtime.context.user_id)
            .execute()
        )
        return f"Appended {value} to {column}"
    else:
        result = (
            supabase_client.table("volunteer_profiles")
            .update({column: value})
            .eq("user_id", runtime.context.user_id)
            .execute()
        )
        return f"Updated {column} to {value}"


class VolunteerProfile(BaseModel):
    # Class describing a volunteer profile
    city: str
    name: str
    zip_code: int
    radius: int = Field(description="Radius in kilometers")
    availability: dict[str, str] = Field(
        description=(
            """Availability per weekday as a dict mapping day names to time
             ranges. """
            """Day names must be full English weekday names (Monday, Tuesday,
             etc.). """
            "Time ranges must be in 'HH-HH' format using 24h time. "
            'Example: {"Monday": "10-13", "Tuesday": "9-17"}'
            '{"Wednesday": "17-20", "Thursday": "14-16", "Friday": "9-13}'
        )
    )
    skills: list[str] = Field(
        description="List of skills the user has, e.g. ['Cooking', 'Cleaning']"
    )
    languages: list[str] = Field(
        description="""List of languages the user can speak,
        e.g. ['English', 'German']"""
    )
    h_week: int = Field(
        description="""Number of hours the user is available/wants
         to commit per week"""
    )
    start_date: date = Field(description="Start date in YYYY-MM-DD format")
    end_date: date = Field(description="End date in YYYY-MM-DD format")
    recurring: bool = Field(
        description="""Whether the user wants to commit to a recurring schedule
         or just one-time events"""
    )
    preference: str = Field(
        description="""Use this field as a kind of search term and summary of
         what the user is looking for at the moment. It will be used to create
         an embedding for the user's profile."""
    )
    preference_embedding: list[float] | None = Field(default=None)


@tool
def create_volunteer_profile(
    runtime: ToolRuntime[Context], profile: VolunteerProfile
) -> str:
    """
    Creates a new volunteer profile in the database.
    Use this when the user wants to create a new profile.
    """
    print(f"Creating volunteer profile for user ID: {runtime.context.user_id}")

    openai = OpenAI(api_key=os.environ.get("OPENAI_TOKEN"))
    # TODO: profile preference soll ein vom model erstellte summary sein
    supabase_client = runtime.context.supabase

    profile.preference_embedding = get_embedding(openai, profile.preference)
    profile = json.loads(profile.model_dump_json())
    profile["user_id"] = runtime.context.user_id
    result = (
        supabase_client.table("volunteer_profiles").insert(profile).execute()
    )
    return f"""Created volunteer profile for user ID: {runtime.context.user_id}
             with the following data: {result.data[0]}"""


def _zip_score(
    user_zip: int | None, opp_zip: int | None, radius: int | None
) -> float:
    """Calculate zip score based on the zip code. When all match highest score
    when third digit matches second highest, when first matches third highest
    as this is the federal state. Else none

    Args:
        user_zip (int | None): _description_
        opp_zip (int | None): _description_
        radius (int | None): _description_

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
        profile.get("radius"),
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
