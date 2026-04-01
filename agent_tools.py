from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.utils.function_calling import ToolDescription
from typing import Any
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
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
from aux import get_secret


class VolunteerProfile(BaseModel):
    # Class describing a volunteer profile
    city: str | None = Field(default=None, description="City of the user")
    name: str | None = Field(default=None, description="Name of the user")
    zip_code: int | str | None = Field(
        default=None, description="Zip code of the user"
    )
    availability: dict[str, str] | str | None = Field(
        default_factory=dict,
        description=(
            "Availability by weekday and time range, e.g. {'Monday': '10-13'}"
        ),
    )
    skills: list[str] | str | None = Field(
        default_factory=list,
        description=(
            "List of skills the user has, e.g. ['Cooking', 'Cleaning']"
        ),
    )
    languages: list[str] | str | None = Field(
        default_factory=list,
        description=(
            "List of languages the user can speak, e.g. ['English', 'German']"
        ),
    )
    h_week: int | str | None = Field(
        default=None,
        description=(
            "Number of hours the user is available/wants to commit per week"
        ),
    )
    start_date: date | str | None = Field(
        default=None, description="Start date in YYYY-MM-DD format"
    )
    end_date: date | str | None = Field(
        default=None, description="End date in YYYY-MM-DD format"
    )
    recurring: bool | str | None = Field(
        default=None,
        description="Whether recurring opportunities are preferred",
    )
    preference: str | None = Field(
        default=None,
        description="""Use this field as a kind of search term and summary of
         what the user is looking for at the moment. It will be used to create
         an embedding for the user's profile.""",
    )
    preference_embedding: list[float] | None = Field(default=None)
    contact: str | None = Field(
        default=None, description="Contact detail (email or phone)"
    )


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
    print("Calling get_volunteer_information")
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
    print(f"Updating volunteer profile: {field} {value} {operation}")
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


REQUIRED_PROFILE_FIELDS: tuple[str, ...] = (
    "name",
    "contact",
    "city",
    "zip_code",
    "h_week",
    "start_date",
    "end_date",
    "recurring",
    "preference",
    "skills",
    "languages",
    "availability",
)


def _is_missing_profile_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _parse_date_value(value: date | str | None) -> tuple[date | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, date):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False
    try:
        return date.fromisoformat(text), False
    except ValueError:
        return None, True


def _normalize_profile_for_creation(
    profile: VolunteerProfile,
) -> tuple[dict[str, Any], list[str], list[str]]:
    missing_fields: list[str] = []
    validation_errors: list[str] = []
    normalized: dict[str, Any] = {}

    for text_field in ("name", "contact", "city", "preference"):
        raw = getattr(profile, text_field, None)
        value = str(raw).strip() if raw is not None else None
        normalized[text_field] = value if value else None
        if _is_missing_profile_value(normalized[text_field]):
            missing_fields.append(text_field)

    zip_code = to_int(profile.zip_code)
    normalized["zip_code"] = zip_code
    if zip_code is None:
        if _is_missing_profile_value(profile.zip_code):
            missing_fields.append("zip_code")
        else:
            validation_errors.append("zip_code must be a valid number.")
    elif zip_code <= 0:
        validation_errors.append("zip_code must be a positive number.")

    h_week = to_int(profile.h_week)
    normalized["h_week"] = h_week
    if h_week is None:
        if _is_missing_profile_value(profile.h_week):
            missing_fields.append("h_week")
        else:
            validation_errors.append("h_week must be a valid number.")
    elif h_week <= 0:
        validation_errors.append("h_week must be a positive number.")

    recurring = to_bool(profile.recurring)
    normalized["recurring"] = recurring
    if recurring is None:
        if _is_missing_profile_value(profile.recurring):
            missing_fields.append("recurring")
        else:
            validation_errors.append(
                "recurring must be true/false (or yes/no)."
            )

    start_date, start_invalid = _parse_date_value(profile.start_date)
    end_date, end_invalid = _parse_date_value(profile.end_date)
    normalized["start_date"] = start_date
    normalized["end_date"] = end_date

    if start_date is None:
        if _is_missing_profile_value(profile.start_date):
            missing_fields.append("start_date")
        elif start_invalid:
            validation_errors.append("start_date must use YYYY-MM-DD format.")
    if end_date is None:
        if _is_missing_profile_value(profile.end_date):
            missing_fields.append("end_date")
        elif end_invalid:
            validation_errors.append("end_date must use YYYY-MM-DD format.")

    if (
        start_date is not None
        and end_date is not None
        and end_date < start_date
    ):
        validation_errors.append("end_date must be on or after start_date.")

    normalized["availability"] = to_dict(profile.availability)
    normalized["skills"] = to_list(profile.skills)
    normalized["languages"] = to_list(profile.languages)

    # Keep required field list in one place and ensure deterministic ordering.
    missing_fields = [
        field for field in REQUIRED_PROFILE_FIELDS if field in missing_fields
    ]

    return normalized, missing_fields, validation_errors


@tool
def create_volunteer_profile(
    runtime: ToolRuntime[Context], profile: VolunteerProfile
) -> str:
    """Create a new volunteer profile for the logged-in user.

    Call this only after all required profile fields are collected and
    confirmed by the user.
    """
    print(f"Calling create_volunteer_profile")
    existing = safe_profile_lookup(
        runtime.context.supabase, runtime.context.user_id
    )
    if existing is not None:
        return "A profile already exists for this account. Use update instead."

    normalized_profile, missing_fields, validation_errors = (
        _normalize_profile_for_creation(profile)
    )
    if missing_fields or validation_errors:
        parts: list[str] = []
        if missing_fields:
            parts.append(
                "Missing required fields: " + ", ".join(missing_fields) + "."
            )
        if validation_errors:
            parts.append("Validation issues: " + " ".join(validation_errors))
        parts.append(
            "Do not call create_volunteer_profile again yet. Ask the user for "
            "the next missing field and call this tool only after all required "
            "fields are complete and confirmed."
        )
        return " ".join(parts)

    embedding = maybe_create_embedding(
        str(normalized_profile.get("preference") or "")
    )
    payload = _compact(
        {
            "user_id": runtime.context.user_id,
            "name": normalized_profile.get("name"),
            "contact": normalized_profile.get("contact"),
            "city": normalized_profile.get("city"),
            "zip_code": normalized_profile.get("zip_code"),
            "availability": normalized_profile.get("availability"),
            "skills": normalized_profile.get("skills"),
            "languages": normalized_profile.get("languages"),
            "h_week": normalized_profile.get("h_week"),
            "start_date": normalized_profile["start_date"].isoformat()
            if normalized_profile.get("start_date")
            else None,
            "end_date": normalized_profile["end_date"].isoformat()
            if normalized_profile.get("end_date")
            else None,
            "recurring": normalized_profile.get("recurring"),
            "preference": normalized_profile.get("preference"),
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


@tool
def create_volunteer_opportunity(
    runtime: ToolRuntime[Context], opportunity: OpportunityInput
) -> str:
    """Create a new volunteering opportunity (for NGO/coordinator workflows)."""
    print(f"Calling create_volunteer_opportunity")
    embedding_text = " ".join(
        [
            opportunity.organization,
            opportunity.title,
            opportunity.summary,
            " ".join(opportunity.tasks),
            " ".join(opportunity.required_skills),
        ]
    ).strip()
    embedding = maybe_create_embedding(embedding_text)

    payload = _compact(
        {
            "org": opportunity.organization,
            "title": opportunity.title,
            "summary": opportunity.summary,
            "tasks": opportunity.tasks,
            "required_skills": opportunity.required_skills,
            "optional_skills": opportunity.optional_skills,
            "language_requirements": opportunity.languages,
            "amount_volunteers": opportunity.amount_volunteers,
            "schedule": opportunity.schedule,
            "hours_week": opportunity.hours_week,
            "recurring": opportunity.recurring,
            "zip_code": opportunity.zip_code,
            "city": opportunity.city,
            "email": opportunity.email,
            "embedding": embedding,
        }
    )

    try:
        runtime.context.supabase.table("opportunities").insert(
            payload
        ).execute()
        return "Opportunity created successfully."
    except Exception:
        return """I could not create the opportunity due to a database schema
         mismatch. Please check the database schema and try again."""


class MappedSkill(BaseModel):
    original_skill: str = Field(description="The original skill to map")
    taxonomy_label: str | None = Field(
        description="The closest taxonomy label"
    )
    confidence: float = Field(
        description="The confidence score from 0.0 to 1.0"
    )


class SkillMappingResult(BaseModel):
    mappings: list[MappedSkill] = Field(
        description="The list of mapped skills"
    )


@tool
def map_skills_to_taxonomy(skills: list[str]) -> list[MappedSkill]:
    """Map free-text skills to taxonomy labels
    for robust downstream matching."""
    print("Calling map_skills_to_taxonomy")
    print(f"Mapping skills to taxonomy: {skills}")
    with open("cleaned_list.json", "r") as f:
        taxonomy = json.load(f)["items"]
    print(f"Taxonomy loaded {type(taxonomy)}")
    cleaned_skills = [skill.strip() for skill in skills if skill.strip()]
    if not cleaned_skills:
        return []

    return _map_skills(cleaned_skills, taxonomy)


def _map_skills(skills: list[str], taxonomy: list[str]) -> list[MappedSkill]:
    structured_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=get_secret("OPENAI_API_KEY"),
    )

    structured_llm = structured_llm.with_structured_output(SkillMappingResult)
    print("Model loaded")
    messages = [
        SystemMessage(
            content="""You are a skill taxonomy mapper. Your job is to map
                   free-text skills to the closest entry in the provided
                    taxonomy.

                    Instructions:
                    - Prefer specific matches over broad parent categories
                    - If a skill genuinely does not fit any taxonomy entry,
                     set taxonomy_id and taxonomy_label to null
                    - Assign a confidence score from 0.0 to 1.0"""
        ),
        HumanMessage(
            content=f"""TAXONOMY:
                     {json.dumps(taxonomy, indent=2)}
                     SKILLS TO MAP:
                     {json.dumps(skills)}"""
        ),
    ]

    result: SkillMappingResult = structured_llm.invoke(messages)
    print("result", result.mappings)
    return result.mappings
