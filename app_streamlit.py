import json
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from agent_full import (
    NAKSHATRA_TRIADS,
    build_commute_agent,
    create_authorization_url,
    creds_from_dict,
    evaluate_natal_star_day,
    exchange_oauth_code,
    fetch_upcoming_meetings,
    pick_surprise_fact,
    run_agent_turn,
)

ALL_NAKSHATRAS: List[str] = [star for _, group in NAKSHATRA_TRIADS for star in group]


def handle_oauth_callback() -> None:
    params = st.query_params
    if "error" in params:
        st.session_state["oauth_error"] = (
            f"Google returned an error: {params.get('error')} "
            f"({params.get('error_description', 'no extra details')})"
        )
        st.query_params.clear()
        st.rerun()

    if "code" not in params or "state" not in params:
        return

    try:
        cred_dict, email = exchange_oauth_code(
            code=str(params["code"]),
            received_state=str(params["state"]),
            client_id=str(st.secrets["GOOGLE_CLIENT_ID"]),
            client_secret=str(st.secrets["GOOGLE_CLIENT_SECRET"]),
            redirect_uri=str(st.secrets["REDIRECT_URI"]),
        )
        st.session_state["google_creds"] = cred_dict
        st.session_state["user_email"] = email
        st.session_state.pop("oauth_error", None)
    except Exception as exc:
        st.session_state["oauth_error"] = f"{type(exc).__name__}: {exc}"
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


def submit_prompt(prompt_text: str) -> None:
    cleaned = prompt_text.strip()
    if not cleaned:
        return

    st.session_state["chat_history"].append({"role": "user", "content": cleaned})
    with st.spinner("Working on your request..."):
        agent = build_commute_agent(
            groq_api_key=str(st.secrets["GROQ_API_KEY"]),
            creds_dict=st.session_state.get("google_creds"),
            model_name=str(st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")),
        )
        prior_history: List[Dict[str, str]] = st.session_state["chat_history"][:-1]
        reply, tool_events = run_agent_turn(agent, prior_history, cleaned)

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": reply,
            "tool_events": tool_events,
        }
    )
    st.rerun()


st.set_page_config(
    page_title="Personal Agent Hub",
    page_icon="🧭",
    layout="wide",
)
handle_oauth_callback()

st.session_state.setdefault(
    "chat_history",
    [
        {
            "role": "assistant",
            "content": (
                "Hi! Pick any of the four corner actions or ask me anything in the center hub."
            ),
            "tool_events": [],
        }
    ],
)
st.session_state.setdefault("active_panel", None)
st.session_state.setdefault("last_fact", None)

raw_creds = st.session_state.get("google_creds")
is_connected = bool(raw_creds)
upcoming_meetings: List[Dict[str, str]] = []

if is_connected:
    creds_obj = creds_from_dict(raw_creds)
    if creds_obj:
        try:
            upcoming_meetings = fetch_upcoming_meetings(creds_obj, max_results=5)
        except Exception as exc:
            st.session_state["calendar_warning"] = str(exc)

with st.sidebar:
    st.header("Google Account")
    if not is_connected:
        st.caption("Status: **Not connected**")
        st.info("Connect Google to check next meeting locations and infer your area from recent emails.")
        auth_url = create_authorization_url(
            client_id=str(st.secrets["GOOGLE_CLIENT_ID"]),
            client_secret=str(st.secrets["GOOGLE_CLIENT_SECRET"]),
            redirect_uri=str(st.secrets["REDIRECT_URI"]),
        )
        st.link_button("Connect Google Account", auth_url, use_container_width=True)
    else:
        st.success(f"Connected: **{st.session_state.get('user_email')}**")
        if st.button("Disconnect Google Account", use_container_width=True):
            disconnect_google()
            st.rerun()

        st.divider()
        st.subheader("Upcoming Meetings")
        if upcoming_meetings:
            for m in upcoming_meetings:
                loc = m.get("location") or "No location set"
                start = m.get("start", "")
                st.markdown(f"- **{m['summary']}**  \n  `{start}` · {loc}")
        else:
            st.caption("No upcoming events found.")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["active_panel"] = None
        st.rerun()

if st.session_state.get("oauth_error"):
    st.error(f"**Google Sign-In Issue:** {st.session_state['oauth_error']}")
    if st.button("Dismiss Error"):
        st.session_state.pop("oauth_error", None)
        st.rerun()

# Top-Left & Top-Right corner buttons
top_left, top_spacer, top_right = st.columns([2.2, 1.6, 2.2])
with top_left:
    if st.button(
        "COMMUTE ME TO MY NEXT MEETING PLACE",
        use_container_width=True,
        type="primary" if st.session_state["active_panel"] == "commute" else "secondary",
    ):
        st.session_state["active_panel"] = (
            None if st.session_state["active_panel"] == "commute" else "commute"
        )
with top_right:
    if st.button(
        "WHAT IS THE TOURIST PLACE NEAR ME",
        use_container_width=True,
        type="primary" if st.session_state["active_panel"] == "tourist" else "secondary",
    ):
        st.session_state["active_panel"] = (
            None if st.session_state["active_panel"] == "tourist" else "tourist"
        )

# Contextual Panels
if st.session_state["active_panel"] == "commute":
    with st.container(border=True):
        st.subheader("Commute to Next Meeting Place")
        next_meet = upcoming_meetings[0] if upcoming_meetings else None
        detected_dest = (next_meet.get("location") or "").strip() if next_meet else ""

        if next_meet:
            st.info(
                f"Next meeting: **{next_meet['summary']}** (`{next_meet.get('start', '')}`)"
                + (f" — Venue: **{detected_dest}**" if detected_dest else " — *No location in meeting request*")
            )

        c1, c2 = st.columns(2)
        with c1:
            current_loc = st.text_input(
                "Current / pickup location *",
                placeholder="Enter your current location",
                key="commute_pickup",
            )
        with c2:
            dest_loc = st.text_input(
                "Destination (meeting location) *",
                value=detected_dest,
                placeholder="Meeting location not found — enter destination",
                key="commute_dest",
            )

        if st.button("Compare Ride Fares", type="primary", key="submit_commute_btn"):
            if not current_loc.strip() or not dest_loc.strip():
                st.warning("Please enter both your current location and destination.")
            else:
                submit_prompt(
                    f"Compare estimated Bike, Auto, Cab, and GREEN SM (EV) fares "
                    f"from '{current_loc.strip()}' to '{dest_loc.strip()}'."
                )

elif st.session_state["active_panel"] == "tourist":
    with st.container(border=True):
        st.subheader("Tourist Places Near Me")
        st.caption(
            "Leave blank to check recent Gmail messages for your current/residential area, "
            "or type a location directly."
        )
        area_input = st.text_input(
            "Current area / residence (optional if Gmail is connected)",
            placeholder="e.g., Madhapur, Hyderabad",
            key="tourist_area_input",
        )
        if st.button("Show Nearest Tourist Places", type="primary", key="submit_tourist_btn"):
            if area_input.strip():
                submit_prompt(
                    f"What are the nearest tourist places near '{area_input.strip()}'? "
                    f"Include approximate distances and travel times."
                )
            elif is_connected:
                submit_prompt(
                    "Check my recent Gmail messages to identify my current location or residence area, "
                    "then show the nearest tourist places and attractions to that area."
                )
            else:
                st.warning("Enter an area or connect your Google account in the sidebar.")

elif st.session_state["active_panel"] == "star":
    with st.container(border=True):
        st.subheader("Today's Outlook Based on Your Natal Star")
        s1, s2 = st.columns(2)
        with s1:
            natal_star = st.selectbox(
                "Your natal star (Nakshatra) *",
                options=ALL_NAKSHATRAS,
                key="natal_star_select",
            )
        with s2:
            todays_star = st.selectbox(
                "Today's star (Nakshatra) *",
                options=ALL_NAKSHATRAS,
                key="todays_star_select",
            )

        star_dataset_raw = st.text_area(
            "Custom Tara dataset override (JSON keyed by 1..9 — optional)",
            placeholder='{"2": {"outlook": "Favorable", "effects": "..."}}',
            height=80,
            key="star_dataset_json",
        )

        if st.button("How Would Today Be?", type="primary", key="submit_star_btn"):
            custom_star_data = None
            if star_dataset_raw.strip():
                try:
                    custom_star_data = json.loads(star_dataset_raw.strip())
                except Exception as exc:
                    st.error(f"Invalid JSON in star dataset: {exc}")
                    st.stop()

            result = evaluate_natal_star_day(
                natal_star=natal_star,
                todays_star=todays_star,
                custom_dataset=custom_star_data,
            )
            submit_prompt(
                f"My natal star is '{natal_star}' and today's star is '{todays_star}'. "
                f"Here is the evaluated Tara result: {json.dumps(result, ensure_ascii=False)}. "
                f"Summarize clearly how today would be for me, including Tara name, lord, effects, and any note/remedy."
            )

elif st.session_state["active_panel"] == "fact":
    with st.container(border=True):
        st.subheader("Surprise Me With a Fact")
        facts_dataset_raw = st.text_area(
            "Custom facts list (JSON list of strings — optional)",
            placeholder='["Fact 1", "Fact 2"]',
            height=75,
            key="facts_dataset_json",
        )
        if st.button("Pick a Random Fact", type="primary", key="submit_fact_btn"):
            custom_facts: Optional[List[str]] = None
            if facts_dataset_raw.strip():
                try:
                    parsed = json.loads(facts_dataset_raw.strip())
                    if not isinstance(parsed, list):
                        st.error("Facts dataset must be a JSON list of strings.")
                        st.stop()
                    custom_facts = [str(x) for x in parsed]
                except Exception as exc:
                    st.error(f"Invalid JSON in facts dataset: {exc}")
                    st.stop()

            selected_fact = pick_surprise_fact(
                custom_facts=custom_facts,
                exclude_fact=st.session_state.get("last_fact"),
            )
            st.session_state["last_fact"] = selected_fact
            st.session_state["chat_history"].append(
                {"role": "user", "content": "Surprise me with a fact"}
            )
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": f"**Surprise Fact:** {selected_fact}",
                    "tool_events": [],
                }
            )
            st.rerun()

# Center Hub
st.markdown("### ASK ME ANYTHING YOU WANA KNOW?")

for item in st.session_state["chat_history"]:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        tool_events: Optional[List[Dict[str, Any]]] = item.get("tool_events")
        if item["role"] == "assistant" and tool_events:
            with st.expander("Agent Tool Calls"):
                for te in tool_events:
                    st.markdown(f"**`{te['tool']}`**")
                    st.code(te["output"], language="json")

prompt = st.chat_input("Ask me anything you wanna know...")
if prompt:
    submit_prompt(prompt)

# Bottom-Left & Bottom-Right corner buttons
st.markdown("<br>", unsafe_allow_html=True)
bot_left, bot_spacer, bot_right = st.columns([2.2, 1.6, 2.2])
with bot_left:
    if st.button(
        "WANT TO KNOW HOW WOULD TODAY BE BASED ON YOUR STAR",
        use_container_width=True,
        type="primary" if st.session_state["active_panel"] == "star" else "secondary",
    ):
        st.session_state["active_panel"] = (
            None if st.session_state["active_panel"] == "star" else "star"
        )
with bot_right:
    if st.button(
        "SURPRISE ME WITH A FACT",
        use_container_width=True,
        type="primary" if st.session_state["active_panel"] == "fact" else "secondary",
    ):
        st.session_state["active_panel"] = (
            None if st.session_state["active_panel"] == "fact" else "fact"
        )