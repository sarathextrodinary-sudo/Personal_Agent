import datetime as dt
from typing import Dict, List

import requests
import streamlit as st

from agent_full import (
    build_commute_agent,
    create_authorization_url,
    exchange_oauth_code,
    fetch_upcoming_meetings,
    creds_from_dict,
    run_agent_turn,
)


@st.cache_resource
def get_pending_oauth_states() -> Dict[str, float]:
    return {}


def handle_oauth_callback() -> None:
    params = st.query_params
    if "error" in params:
        st.error(f"Google authentication error: {params['error']}")
        st.query_params.clear()
        return

    if "code" not in params or "state" not in params:
        return

    try:
        cred_dict, email = exchange_oauth_code(
            code=params["code"],
            received_state=params["state"],
            client_id=st.secrets["GOOGLE_CLIENT_ID"],
            client_secret=st.secrets["GOOGLE_CLIENT_SECRET"],
            redirect_uri=st.secrets["REDIRECT_URI"],
            pending_states=get_pending_oauth_states(),
        )
        st.session_state["google_creds"] = cred_dict
        st.session_state["user_email"] = email
    except Exception as exc:
        st.error(f"OAuth callback failed: {exc}")
    finally:
        st.query_params.clear()
        st.rerun()


def disconnect_google() -> None:
    raw = st.session_state.get("google_creds")
    if raw and raw.get("token"):
        try:
            requests.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": raw["token"]},
                timeout=5,
            )
        except Exception:
            pass
    st.session_state.pop("google_creds", None)
    st.session_state.pop("user_email", None)


st.set_page_config(page_title="Personal Agent — Calendar & Commute", layout="wide")
handle_oauth_callback()

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault(
    "chat_history",
    [
        {
            "role": "assistant",
            "content": (
                "Hi! Ask me about your upcoming calendar meetings or compare estimated "
                "bike, auto, cab, and EV cab fares across **Rapido, Ola, Uber, and SV Green** "
                "(for example: *'Compare cab and bike fares from Madhapur to my next meeting'*)."
            ),
        }
    ],
)

with st.sidebar:
    st.header("Google Account")
    raw_creds = st.session_state.get("google_creds")

    if not raw_creds:
        st.info("Connect your Google account to let the agent read your meeting locations.")
        auth_url = create_authorization_url(
            client_id=st.secrets["GOOGLE_CLIENT_ID"],
            client_secret=st.secrets["GOOGLE_CLIENT_SECRET"],
            redirect_uri=st.secrets["REDIRECT_URI"],
            pending_states=get_pending_oauth_states(),
        )
        st.link_button("Connect Google Account", auth_url, use_container_width=True)
    else:
        st.success(f"Signed in as **{st.session_state.get('user_email')}**")
        if st.button("Disconnect Google Account", use_container_width=True):
            disconnect_google()
            st.rerun()

        st.divider()
        st.subheader("Upcoming Meetings")
        creds_obj = creds_from_dict(raw_creds)
        if creds_obj:
            try:
                meetings = fetch_upcoming_meetings(creds_obj, max_results=5)
                if meetings:
                    for m in meetings:
                        loc = m["location"] or "No location set"
                        st.markdown(f"- **{m['summary']}** — {loc}")
                else:
                    st.caption("No upcoming events.")
            except Exception as exc:
                st.caption(f"Calendar load issue: {exc}")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

st.title("Personal Agent Chat (Calendar + Ride Fare Comparison)")

# Render existing conversation
for item in st.session_state["chat_history"]:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

# Handle new chat prompt
prompt = st.chat_input("Ask about meetings or compare Rapido, Ola, Uber & SV Green fares...")
if prompt:
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking calendar & comparing ride options..."):
            agent = build_commute_agent(
                groq_api_key=st.secrets["GROQ_API_KEY"],
                creds_dict=st.session_state.get("google_creds"),
            )
            prior_history: List[Dict[str, str]] = st.session_state["chat_history"][:-1]
            reply, tool_events = run_agent_turn(agent, prior_history, prompt)
            st.markdown(reply)

            if tool_events:
                with st.expander("Agent Tool Calls"):
                    for te in tool_events:
                        st.markdown(f"**`{te['tool']}`**")
                        st.code(te["output"], language="json")

    st.session_state["chat_history"].append({"role": "assistant", "content": reply})