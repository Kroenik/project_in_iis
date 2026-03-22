import streamlit as st
from agent import get_agent
from langgraph.checkpoint.memory import InMemorySaver
from custom_classes import Context
from langchain_openai import ChatOpenAI
from supabase import create_client
from auth import require_auth, logout

require_auth()

st.sidebar.button("Logout", on_click=logout)
st.title("Volunteer Finder")


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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Talk to me to find volunteering opportunities!"):
    agent_input = {"messages": [{"role": "user", "content": prompt}]}
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        result = agent.invoke(agent_input, config=config, context=context)
        response = result["structured_response"].response

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )


def run_app() -> None:
    try:
        _ensure_runtime_secrets()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
