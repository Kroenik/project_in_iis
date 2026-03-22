import streamlit as st
from agent import get_agent
from langgraph.checkpoint.memory import InMemorySaver
from custom_classes import Context
from langchain_openai import ChatOpenAI
from supabase import create_client
from auth import require_auth, logout

require_auth()


def _get_secret(name: str) -> str:
    value = st.secrets.get(name)
    if not value:
        raise RuntimeError(f"Required secret {name} is not set")
    return value


def _ensure_runtime_secrets() -> dict[str, str]:
    openai_api_key = _get_secret("OPENAI_API_KEY")
    supabase_url = _get_secret("SUPABASE_URL")
    supabase_key = _get_secret("SUPABASE_KEY")
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
                "content": "Hi and welcome to ConVolunteer! I can help you"
                "create your volunteer profile, find matching volunteering"
                "opportunities, explain next steps to contact the organization"
                ".\n\n You can as well find examples above. Just let me know"
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


api_key = st.secrets["OPENAI_API_KEY"]


@st.cache_resource
def get_supabase_client():
    client = create_client(
        st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
    )
    client.auth.set_session(
        st.session_state.session.access_token,
        st.session_state.session.refresh_token,
    )
    return client


@st.cache_resource
def load_agent():
    model = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.5, timeout=30, max_tokens=1000
    )
    checkpointer = InMemorySaver()
    return get_agent(model, checkpointer, Context), checkpointer


supabase_client = get_supabase_client()
config = {"configurable": {"thread_id": st.session_state.user.id}}
context = Context(user_id=st.session_state.user.id, supabase=supabase_client)
agent, checkpointer = load_agent()

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Talk to me to find volunteering opportunities!"):
#     agent_input = {"messages": [{"role": "user", "content": prompt}]}
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Add user message to chat history

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         result = agent.invoke(agent_input, config=config, context=context)
#         response = result["structured_response"].response

#         st.markdown(response)

#     st.session_state.messages.append(
#         {"role": "assistant", "content": response}
#     )


def run_app() -> None:
    try:
        _ensure_runtime_secrets()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    _initialize_chat()
    _display_side_bar()
    _display_intro_section()
    quick_action = _display_quick_actions()
    st.divider()


run_app()
