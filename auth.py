# auth.py
import streamlit as st
from supabase import create_client

supabase = create_client(
    st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
)


def login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            res = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state.session = res.session
            st.session_state.user = res.user
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")


def logout():
    supabase.auth.sign_out()
    st.session_state.clear()


def require_auth():

    if "session" not in st.session_state:
        login_page()
        st.stop()
