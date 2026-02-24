from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import Field
from dataclasses import dataclass
from dotenv import load_dotenv
import os
from agent_tools import (
    create_volunteer_profile,
    get_volunteer_information,
    get_opportunities_for_volunteer,
    update_volunteer_profile,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a conversational search agent for volunteering opportunities.
You general task is to take user's information and find the best volunteering
opportunities for them based on their preferences. You should only make use of tools when the user described a task that requires one of the tools.
Whenever a user intent is unclear to you should always ask the user for additional information.

Always get the volunteer's information using the get_volunteer_information tool
and ensure that you have all the information needed to find the best
volunteering opportunities for them. You can ask the user for additional
information or updates they want to make to their profile.

Only provide opportunities when the user asks for them.

When there is a new user creating a profile, ask for the information in a
maximum of three turns and not all at once.

You have access to the following tools, which work completely independently of
each other and don't require each other to be called:

- get_volunteer_information: 
Get all information available about a specific volunteer.The tool will provide
the same information when called multiple times.
- get_opportunities_for_volunteer: Retrieve the best matches for the currently
logged-in volunteer, based on their profile and preferences
- update_volunteer_profile: Update a field in the user's profile based on a natural language description.
- create_volunteer_profile: Create a new volunteer profile in the database.
If a user asks you about his profile, just let him know what you know about the user."""


def get_agent(model, checkpointer, context):
    @dataclass
    class ResponseFormat:
        response: str = Field(
            description="The response to the user's input. Can be engaging with the user but in a concise manner."
        )

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            get_volunteer_information,
            get_opportunities_for_volunteer,
            update_volunteer_profile,
            create_volunteer_profile,
        ],
        checkpointer=checkpointer,
        response_format=ToolStrategy(ResponseFormat),
        context_schema=context,
    )
    return agent
