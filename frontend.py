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
        model="gpt-4o-mini", temperature=0.5, timeout=10, max_tokens=1000
    )
    checkpointer = InMemorySaver()
    return get_agent(model, checkpointer, Context), checkpointer


supabase_client = get_supabase_client()
config = {"configurable": {"thread_id": str(st.session_state.user.id)}}
context = Context(user_id=st.session_state.user.id, supabase=supabase_client)
agent, checkpointer = load_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
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
