from typing import Any
from langchain.tools import tool, ToolRuntime
import os
from openai import OpenAI
from custom_classes import Context
import json
from datetime import date
from pydantic import BaseModel, Field
from aux_tools import (
    safe_profile_lookup,
    normalize_profile,
    safe_get_all_opportunities,
    normalize_opportunity,
    normalize_update_field,
    UPDATE_FIELD_ALIASES,
    to_int,
    to_bool,
    to_list,
    to_dict,
)
from profile_update_aux import (
    detect_profile_column,
    parse_availability_update,
    maybe_create_embedding,
)
from scoring import score_opportunity


class VolunteerProfile(BaseModel):
    # Class describing a volunteer profile
    city: str = Field(description="City of the user")
    name: str = Field(description="Name of the user")
    zip_code: int = Field(description="Zip code of the user")
    availability: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Availability by weekday and time range, e.g. {'Monday': '10-13'}"
        ),
    )
    skills: list[str] = Field(
        default_factory=list,
        description=(
            "List of skills the user has, e.g. ['Cooking', 'Cleaning']"
        ),
    )
    languages: list[str] = Field(
        default_factory=list,
        description=(
            "List of languages the user can speak, e.g. ['English', 'German']"
        ),
    )
    h_week: int = Field(
        description=(
            "Number of hours the user is available/wants to commit per week"
        )
    )
    start_date: date = Field(description="Start date in YYYY-MM-DD format")
    end_date: date = Field(description="End date in YYYY-MM-DD format")
    recurring: bool = Field(
        description="Whether recurring opportunities are preferred"
    )
    preference: str = Field(
        description="""Use this field as a kind of search term and summary of
         what the user is looking for at the moment. It will be used to create
         an embedding for the user's profile."""
    )
    preference_embedding: list[float] | None = Field(default=None)
    contact: str = Field(description="Contact detail (email or phone)")


class OpportunityInput(BaseModel):
    organization: str = Field(description="Organization name")
    title: str = Field(description="Opportunity title")
    summary: str = Field(description="Short summary")
    tasks: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    optional_skills: str | None = Field(
        default=None, description="Optional skills as free text"
    )
    languages: list[str] = Field(default_factory=list)
    amount_volunteers: int | None = Field(default=None)
    start_date: date | None = Field(default=None)
    end_date: date | None = Field(default=None)
    schedule: dict[str, str] = Field(default_factory=dict)
    hours_week: int | None = Field(default=None)

    recurring: bool | None = Field(default=None)
    zip_code: int | None = Field(default=None)
    city: str | None = Field(default=None)
    email: str = Field(description="Contact email")


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
            "about interests, preferred schedule, and location."
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
    runtime: ToolRuntime[Context],
    field: str,
    value: str,
    operation: str,
) -> str:
    """Update one profile field. Use operation='add'
    for skills/languages additions."""
    canonical_field = normalize_update_field(field)
    if canonical_field is None:
        valid = ", ".join(sorted(UPDATE_FIELD_ALIASES))
        return f"I could not map that field. Use one of: {valid}."

    profile_row = safe_profile_lookup(
        runtime.context.supabase, runtime.context.user_id
    )
    if profile_row is None:
        return "No profile found to update. Please create a profile first."

    db_column = detect_profile_column(profile_row, canonical_field)
    update_value: Any = value.strip()

    if canonical_field in {"zip_code", "h_week"}:
        parsed_int = to_int(update_value)
        if parsed_int is None:
            return f"Please provide a valid number for {canonical_field}."
        update_value = parsed_int
    elif canonical_field == "recurring":
        parsed_bool = to_bool(update_value)
        if parsed_bool is None:
            return "Please provide true/false for recurring preference."
        update_value = parsed_bool
    elif canonical_field == "availability":
        parsed_availability = parse_availability_update(update_value)
        if not parsed_availability:
            return (
                "Please use format like 'Monday:10-13, Wednesday:14-18' "
                "for availability updates."
            )
        current = to_dict(profile_row.get(db_column))
        if operation.lower() == "add":
            current.update(parsed_availability)
            update_value = current
        else:
            update_value = parsed_availability
    elif canonical_field in {"skills", "languages"}:
        incoming = to_list(update_value)
        if not incoming:
            return f"Please provide at least one value for {canonical_field}."
        current = to_list(profile_row.get(db_column))
        if operation.lower() == "add":
            merged = current[:]
            for item in incoming:
                if item not in merged:
                    merged.append(item)
            update_value = merged
        else:
            update_value = incoming

    payload: dict[str, Any] = {db_column: update_value}

    if canonical_field == "preference":
        embedding = maybe_create_embedding(str(update_value))
        if embedding is not None:
            embedding_column = detect_profile_column(
                profile_row, "preference_embedding"
            )
            payload[embedding_column] = embedding

    try:
        runtime.context.supabase.table("volunteer_profiles").update(
            payload
        ).eq("user_id", runtime.context.user_id).execute()
    except Exception:
        return (
            "I could not save that profile update right now. Please try again."
        )

    return f"Updated {canonical_field} successfully."


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


@tool
def create_volunteer_profile(
    runtime: ToolRuntime[Context], profile: VolunteerProfile
) -> str:
    """Create a new volunteer profile for the logged-in useer"""
    existing = safe_profile_lookup(
        runtime.context.supabase, runtime.context.user_id
    )
    if existing is not None:
        return "A profile already exists for this account. Use update instead."

    embedding = maybe_create_embedding(profile.preference)
    payload = _compact(
        {
            "user_id": runtime.context.user_id,
            "name": profile.name,
            "contact": profile.contact,
            "city": profile.city,
            "zip_code": profile.zip_code,
            "availability": profile.availability,
            "skills": profile.skills,
            "languages": profile.languages,
            "h_week": profile.h_week,
            "start_date": profile.start_date.isoformat()
            if profile.start_date
            else None,
            "end_date": profile.end_date.isoformat()
            if profile.end_date
            else None,
            "recurring": profile.recurring,
            "preference": profile.preference,
            "preference_embedding": embedding,
        }
    )
    try:
        runtime.context.supabase.table("volunteer_profiles").insert(
            payload
        ).execute()
        return "Profile created successfully."
    except Exception:
        return (
            "I could not create the profile due to a database schema mismatch."
        )
