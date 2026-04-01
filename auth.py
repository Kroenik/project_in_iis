# auth.py
import streamlit as st
from supabase import create_client, Client
from aux import get_secret


@st.cache_resource
def _get_supabase_client():
    return create_client(
        get_secret("SUPABASE_URL"), get_secret("SUPABASE_KEY")
    )


def _store_auth_state(result: object) -> None:
    session = getattr(result, "session", None)
    user = getattr(result, "user", None)
    if session is None or user is None:
        raise RuntimeError(
            "Authentication succeeded but no user session was returned."
        )
    st.session_state.session = session
    st.session_state.user = user


def _render_login_form(client: Client) -> None:
    st.subheader("Sign In")
    email = st.text_input("Email", key="auth_login_email")
    password = st.text_input(
        "Password", type="password", key="auth_login_password"
    )

    if st.button("Sign In", key="auth_sign_in_btn", use_container_width=True):
        if not email or not password:
            st.warning("Please enter both email and password.")
            return
        try:
            result = client.auth.sign_in_with_password(
                {"email": email.strip(), "password": password}
            )
            _store_auth_state(result)
            st.success("Signed in successfully.")
            st.rerun()
        except Exception:
            st.error(
                "Sign in failed. Please try again by clicking the sign in button."
            )


def _render_signup_form(client: Client) -> None:
    st.subheader("Create Account")
    email = st.text_input("Email", key="auth_signup_email")
    password = st.text_input(
        "Password",
        type="password",
        key="auth_signup_password",
        help="Use at least 8 characters.",
    )
    password_confirm = st.text_input(
        "Confirm Password", type="password", key="auth_signup_password_confirm"
    )

    if st.button(
        "Create Account", key="auth_sign_up_btn", use_container_width=True
    ):
        if not email or not password:
            st.warning("Please enter an email and password.")
            return
        if len(password) < 8:
            st.warning("Please use a password with at least 8 characters.")
            return
        if password != password_confirm:
            st.warning("Passwords do not match.")
            return
        try:
            client.auth.sign_up({"email": email.strip(), "password": password})
            st.success("Account created, you may now sign in.")
        except Exception as e:
            print(e)
            st.error("Account creation failed. Please try again.")


def login_page():
    st.title("ConvVolunteer")
    st.caption("Find volunteering opportunities with guided conversation.")

    try:
        client = _get_supabase_client()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    sign_in_tab, sign_up_tab = st.tabs(["Sign In", "Sign Up"])
    with sign_in_tab:
        _render_login_form(client)
    with sign_up_tab:
        _render_signup_form(client)


def logout():
    try:
        _get_supabase_client().auth.sign_out()
    except Exception:
        pass
    st.session_state.clear()
    st.rerun()


def require_auth():
    session = st.session_state.get("session")
    user = st.session_state.get("user")
    if not session or not user:
        login_page()
        st.stop()
