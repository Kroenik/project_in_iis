from typing import Any
from langgraph.graph.state import CompiledStateGraph
import streamlit as st
from agent import get_agent
from langgraph.checkpoint.memory import InMemorySaver
from custom_classes import Context
from langchain_openai import ChatOpenAI
from supabase import create_client, Client
from auth import require_auth, logout
import re
from aux import get_secret

require_auth()


def _ensure_runtime_secrets() -> dict[str, str]:
    openai_api_key = get_secret("OPENAI_API_KEY")
    supabase_url = get_secret("SUPABASE_URL")
    supabase_key = get_secret("SUPABASE_KEY")
    return {
        "OPENAI_API_KEY": openai_api_key,
        "SUPABASE_URL": supabase_url,
        "SUPABASE_KEY": supabase_key,
    }


def _initialize_chat() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi and welcome to ConVolunteer! I can help you "
                "create your volunteer profile, find matching volunteering "
                "opportunities, explain next steps to contact the organization"
                ".\n\n You can as well find examples above. Just let me know "
                "and if needed I will simply ask follow ups.",
            }
        ]


def _display_side_bar() -> None:
    st.sidebar.title("ConVolunteer")
    st.sidebar.caption("Conversational search for volunteering opportunities")
    st.sidebar.success("Logged in as " + st.session_state.user.email)
    st.sidebar.button("Logout", on_click=logout, use_container_width=True)
    st.sidebar.divider()


def _display_intro_section() -> None:
    st.title("Find your next volunteering opportunity")
    st.caption(
        "Tell me what you want to do and"
        "I will help you find the perfect opportunity."
    )
    st.write("- `Weekend opportunities in Linz`")
    st.write("- `I have IT skills and about 8 hours per week`")
    st.write("- `Help me create my profile`")
    st.write("- `I am an NGO coordinator and want to post a new opportunity`")


def _display_quick_actions() -> str | None:
    st.write("Quick start:")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    if col1.button("Find weekend options", use_container_width=True):
        return "Show me volunteering opportunities on weekends near Linz."
    if col2.button("Create my profile", use_container_width=True):
        return "Please help me create my volunteer profile."
    if col3.button("I have IT skills", use_container_width=True):
        return "I have IT skills and around 8 hours per week. What matches me?"
    if col4.button("Post an opportunity", use_container_width=True):
        return "I am an NGO coordinator and want to post a new volunteering opportunity."
    return None


def _display_message_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _get_supabase_client(secrets: dict[str, str]) -> Client:
    access_token = st.session_state.session.access_token
    refresh_token = st.session_state.session.refresh_token
    if (
        "supabase_client" not in st.session_state
        or st.session_state._supabase_access_token != access_token
    ):
        client = create_client(
            secrets["SUPABASE_URL"], secrets["SUPABASE_KEY"]
        )
        try:
            client.auth.set_session(access_token, refresh_token)
        except Exception as e:
            st.session_state.pop("session", None)
            st.session_state.pop("user", None)
            st.error(f"Failed to set session: {e}")
            st.stop()
        st.session_state.supabase_client = client
        st.session_state._supabase_access_token = access_token
    return st.session_state.supabase_client


def _load_agent(openai_api_key: str) -> CompiledStateGraph:
    existing_runtime = st.session_state.get("agent_runtime")
    if (
        isinstance(existing_runtime, dict)
        and existing_runtime.get("openai_api_key") == openai_api_key
        and existing_runtime.get("agent") is not None
    ):
        return existing_runtime["agent"]

    model = ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0.3,
        timeout=15,
        max_tokens=1000,
        max_retries=1,
        api_key=openai_api_key,
    )
    checkpointer = InMemorySaver()
    st.session_state.agent_runtime = {
        "agent": get_agent(
            model=model, checkpointer=checkpointer, context=Context
        ),
        "checkpointer": checkpointer,
        "openai_api_key": openai_api_key,
    }
    return st.session_state.agent_runtime["agent"]


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def _fallback_response(prompt: str) -> str | None:
    token_count = len(re.findall(r"\w+", prompt))
    if token_count >= 2:
        return None
    return (
        "It seems like your message is too short. Please provide "
        "more details\n\n"
        "For example: your city, available day/time, or skill."
    )


def _invoke_agent(
    prompt: str,
    agent: Any,
    context: Context,
    config: dict[str, Any],
) -> str:
    fallback = _fallback_response(prompt)
    if fallback is not None:
        return fallback

    with st.status("Processing your request...") as status:
        status.update(label="Understanding your request...", state="running")
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
                context=context,
            )
            structured = result.get("structured_response")
            response = getattr(structured, "response", None)
            if response and str(response).strip():
                status.update(label="Done", state="complete")
                return str(response)
            status.update(label="Done", state="complete")
            return (
                "I could not generate a full answer yet. "
                "Could you try again with one more detail?"
            )
        except Exception as exc:
            st.session_state.last_runtime_error = str(exc)
            status.update(label="Temporary issue occurred", state="error")
            return (
                "I ran into a temporary issue and could not complete that request. "
                "Please try again in a moment or rephrase your message."
            )


def _handle_prompt(
    selected_prompt: str,
    agent: CompiledStateGraph,
    context: Context,
    config: dict[str, Any],
) -> None:
    clean_prompt = selected_prompt.strip()
    if not clean_prompt:
        return
    _append_message(role="user", content=clean_prompt)
    with st.chat_message("user"):
        st.markdown(clean_prompt)
    with st.chat_message("assistant"):
        response = _invoke_agent(
            prompt=clean_prompt,
            agent=agent,
            context=context,
            config=config,
        )
        st.markdown(response)
        _append_message(role="assistant", content=response)


def run_app() -> None:
    try:
        secrets = _ensure_runtime_secrets()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    _initialize_chat()
    _display_side_bar()
    _display_intro_section()
    quick_action = _display_quick_actions()
    st.divider()

    supabase_client = _get_supabase_client(secrets)
    context = Context(
        user_id=st.session_state.user.id, supabase=supabase_client
    )
    config = {
        "configurable": {"thread_id": st.session_state.user.id},
        "recursion_limit": 8,
    }
    agent = _load_agent(secrets["OPENAI_API_KEY"])
    _display_message_history()

    selected_prompt = quick_action
    chat_prompt = st.chat_input("Type your message here...")
    if chat_prompt:
        selected_prompt = chat_prompt

    if selected_prompt:
        _handle_prompt(selected_prompt, agent, context, config)


run_app()
