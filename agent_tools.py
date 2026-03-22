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


def _coalesce(row: dict[str, Any], aliases: list[str]) -> Any:
    for key in aliases:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _to_bool(value: Any) -> bool | None:
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


def _safe_profile_lookup(
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


def _to_list(value: Any) -> list[str]:
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


def _to_dict(value: Any) -> dict[str, str]:
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


def _normalize_profile(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _coalesce(row, PROFILE_ALIASES["name"]),
        "contact": _coalesce(row, PROFILE_ALIASES["contact"]),
        "city": _coalesce(row, PROFILE_ALIASES["city"]),
        "zip_code": _to_int(_coalesce(row, PROFILE_ALIASES["zip_code"])),
        "radius": _to_int(_coalesce(row, PROFILE_ALIASES["radius"])),
        "availability": _to_dict(
            _coalesce(row, PROFILE_ALIASES["availability"])
        ),
        "skills": _to_list(_coalesce(row, PROFILE_ALIASES["skills"])),
        "languages": _to_list(_coalesce(row, PROFILE_ALIASES["languages"])),
        "h_week": _to_int(_coalesce(row, PROFILE_ALIASES["h_week"])),
        "start_date": _coalesce(row, PROFILE_ALIASES["start_date"]),
        "end_date": _coalesce(row, PROFILE_ALIASES["end_date"]),
        "recurring": _to_bool(_coalesce(row, PROFILE_ALIASES["recurring"])),
        "preference": _coalesce(row, PROFILE_ALIASES["preference"]),
        "preference_embedding": _coalesce(
            row, PROFILE_ALIASES["preference_embedding"]
        ),
    }


@tool
def get_volunteer_information(runtime: ToolRuntime[Context]):
    """Retrieve the currently logged-in volunteer prfile. Use once at start"""
    profile_row = _safe_profile_lookup(
        runtime.context.supabase, runtime.context.user_id
    )
    if profile_row is None:
        return (
            "No volunteer profile exists yet for this account. "
            "Please guide the user through profile setup"
        )

    profile = _normalize_profile(profile_row)
    profile.pop("preference_embedding", None)

    return {
        key: value
        for key, value in profile.items()
        if value not in (None, "", [], {})
    }


# @tool
# def update_volunteer_information(runtime: ToolRuntime[Context]):
@tool
def get_opportunities_for_volunteer(runtime: ToolRuntime[Context]):
    """Retrieve the best matches for the currently logged-in volunteer, based
    on their profile and preferences
    Present the matches more thoroughly and not just a summary"""
    print(f"Getting matches for volunteer ID: {runtime.context.user_id}")
    supabase_client = runtime.context.supabase
    profile = get_profile(
        client=supabase_client, profile_id=runtime.context.user_id
    )
    opportunities = get_opportunities()
    matches = []

    for opportunity in opportunities:
        if opportunity["embedding"] is not None:
            similarity = cosine_similarity(
                [json.loads(profile["preference_embedding"])],
                [json.loads(opportunity["embedding"])],
            )
            print(similarity)
            if similarity > 0.1:
                matches.append(opportunity)
    return matches[:3]


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
            "Availability per weekday as a dict mapping day names to time ranges. "
            "Day names must be full English weekday names (Monday, Tuesday, etc.). "
            "Time ranges must be in 'HH-HH' format using 24h time. "
            'Example: {"Monday": "10-13", "Tuesday": "9-17"}'
            '{"Wednesday": "17-20", "Thursday": "14-16", "Friday": "9-13}'
        )
    )
    skills: list[str] = Field(
        description="List of skills the user has, e.g. ['Cooking', 'Cleaning']"
    )
    languages: list[str] = Field(
        description="List of languages the user can speak, e.g. ['English', 'German']"
    )
    h_week: int = Field(
        description="Number of hours the user is available7wants to commit per week"
    )
    start_date: date = Field(description="Start date in YYYY-MM-DD format")
    end_date: date = Field(description="End date in YYYY-MM-DD format")
    recurring: bool = Field(
        description="Whether the user wants to commit to a recurring schedule or just one-time events"
    )
    preference: str = Field(
        description="""Use this field as a kind of search term and summary of 
        what the user is looking for at the moment. It will be used to create an embedding for the user's profile."""
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
    return f"Created volunteer profile for user ID: {runtime.context.user_id} with the following data: {result.data[0]}"
