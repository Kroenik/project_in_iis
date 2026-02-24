from langchain.tools import tool, ToolRuntime
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import supabase
from aux import get_embedding, get_profile, get_opportunities
from custom_classes import Context
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain_openai import ChatOpenAI
from datetime import date
from pydantic import BaseModel, Field


@tool
def get_volunteer_information(runtime: ToolRuntime[Context]):
    """Retrieves the full profile for the currently logged-in volunteer.
    Call this ONLY ONCE at the start of the conversation to understand the
    user's background.

    Args:
        runtime: The runtime context containing the user ID.

    Returns:
        A dictionary containing the volunteer's information.
    """
    supabase_client = runtime.context.supabase
    print(
        f"Getting volunteer information for user ID: {runtime.context.user_id}"
    )
    res = (
        supabase_client.table("volunteer_profiles")
        .select("*")
        .eq("user_id", runtime.context.user_id)
        .execute()
    )
    if res.data == []:
        return "The user does not yet have a profile. Please ask them to create one."
    else:
        del res.data[0]["preference_embedding"]
        print("Received results")
        return res.data[0]


# @tool
# def update_volunteer_information(runtime: ToolRuntime[Context]):
@tool
def get_opportunities_for_volunteer(runtime: ToolRuntime[Context]):
    """Retrieve the best matches for the currently logged-in volunteer, based
    on their profile and preferences"""
    print(f"Getting matches for volunteer ID: {runtime.context.user_id}")
    supabase_client = runtime.context.supabase
    profile = get_profile(runtime.context.user_id)
    opportunities = get_opportunities()
    matches = []

    for opportunity in opportunities:
        if opportunity["embedding"] is not None:
            similarity = cosine_similarity(
                [json.loads(profile["preference_embedding"])],
                [json.loads(opportunity["embedding"])],
            )
            print(similarity)
            if similarity > 0.5:
                matches.append(opportunity)
    return matches


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
    preference_embedding: list[float] | None = Field(
        default=None, exclude=True
    )


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
