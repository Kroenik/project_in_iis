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
from scoring import score_opportunity


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
) -> dict[str, Any] | str:
    """Get details for one opportunity including contact info and next steps"""
    print(f"Getting details for opportunity ID: {opportunity_id}")
    try:
        result = (
            runtime.context.supabase.table("opportunities")
            .select("*")
            .eq("id", opportunity_id)
            .limit(1)
            .execute()
        )
    except Exception:
        return "I could not load that opportunity right now. Please try again."

    if not result.data:
        return "That opportunity could not be found."

    opportunity = normalize_opportunity(result.data[0])
    next_step = (
        f"Contact {opportunity.get('organization') or 'the organization'}"
        f" via {opportunity.get('email') or 'the listed contact'} and mention"
        f" opportunity ID {opportunity_id}."
    )
    opportunity["next_steps"] = next_step
    return opportunity


@tool
def update_volunteer_profile(
    runtime: ToolRuntime[Context], user_input: str
) -> str:
    """
    Updates a field in the user's profile based on a natural
    language description.
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
            {
        "column": null,
        "value": null,
        "reason": "<why you couldn't match>"
            }
            """
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

        _ = (
            supabase_client.table("volunteer_profiles")
            .update({column: updated_col_vals})
            .eq("user_id", runtime.context.user_id)
            .execute()
        )
        return f"Appended {value} to {column}"
    else:
        _ = (
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
