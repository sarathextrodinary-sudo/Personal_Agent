import os
from datetime import date
import streamlit as st
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from agent_full import (
    append_daily_log,
    apply_toml_secrets_mapping,
    clear_ephemeral_google_credentials,
    create_calendar_event,
    get_google_auth_status,
    get_user_paths,
    list_upcoming_calendar_events,
    load_profile_and_ltm_summary,
    load_recent_conversation_messages,
    log_daily_activity,
    sanitize_user_id,
    save_conversation_turn,
    save_long_term_memory,
    save_user_onboarding_details,
    search_conversations,
    search_daily_memory,
    search_gmail,
    set_active_user,
    set_ephemeral_google_client,
    set_session_google_email,
    update_user_profile,
)


def sync_streamlit_secrets() -> None:
    """Load environment variables and optional [google_oauth] from st.secrets."""
    try:
        raw_secrets = dict(st.secrets)
    except Exception:
        return
    if raw_secrets:
        apply_toml_secrets_mapping(raw_secrets)


sync_streamlit_secrets()


def _clean_session_str(key: str) -> str:
    val: object = st.session_state.get(key)
    return val.strip() if isinstance(val, str) else ""


def _is_google_policy_accepted() -> bool:
    return bool(st.session_state.get("google_policy_accepted", False))


def _default_model_for_provider(provider: str) -> str:
    if provider.strip().lower() == "groq":
        return (
            os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
            or "qwen/qwen3.8-27b"
        )
    return os.getenv("OLLAMA_MODEL", "qwen3:8b").strip() or "qwen3:8b"


def build_llm_from_ui(provider: str, model_name: str):
    cleaned_provider = provider.strip().lower()
    cleaned_model = model_name.strip()

    if cleaned_provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml, Streamlit Cloud Secrets, or .env."
            )
        return ChatGroq(
            model=cleaned_model or _default_model_for_provider("groq"),
            temperature=0,
        )

    return ChatOllama(
        model=cleaned_model or _default_model_for_provider("ollama"),
        temperature=0,
    )


def sync_ephemeral_google_from_session() -> None:
    """Apply Google Mail ID hint and optional RAM-only OAuth client when policy is accepted."""
    if not _is_google_policy_accepted():
        set_session_google_email("")
        set_ephemeral_google_client("", "")
        return

    g_email = _clean_session_str("session_google_email")
    cid = _clean_session_str("ephemeral_google_client_id")
    csec = _clean_session_str("ephemeral_google_client_secret")

    set_session_google_email(g_email)
    if cid and csec:
        set_ephemeral_google_client(client_id=cid, client_secret=csec)
    else:
        set_ephemeral_google_client("", "")


@st.cache_resource
def get_shared_checkpointer():
    return InMemorySaver()


@st.cache_resource
def get_cached_agent(
    user_id: str,
    provider: str,
    model_name: str,
    context_snapshot: str,
):
    llm = build_llm_from_ui(provider, model_name)
    system_prompt = (
    "You are a personal AI assistant. "
    "Tailor your answers to the active user profile below and save updates to their isolated memory. "
    "Whenever the user asks to check their calendar, schedule, or upcoming events, call "
    "`list_upcoming_calendar_events`. Whenever they ask to check unread, recent, or specific emails, "
    "call `search_gmail` with an appropriate Gmail search query.\n\n"
    f"{context_snapshot}"
)
    return create_agent(
        model=llm,
        tools=[
            update_user_profile,
            save_long_term_memory,
            log_daily_activity,
            search_daily_memory,
            search_gmail,
            list_upcoming_calendar_events,
            create_calendar_event,
            search_conversations,
        ],
        system_prompt=system_prompt,
        checkpointer=get_shared_checkpointer(),
    )


def render_google_acceptance_policy() -> None:
    st.markdown(
        "**Google Account Sign-In & Access Policy**\n"
        "- **Browser Sign-In (No Password Stored):** You sign in directly on Google's official "
        "browser page using your Google account. Your Google password is never collected or stored.\n"
        "- **Gmail ID Hint:** Your Gmail address is passed as a sign-in hint (`login_hint`) and "
        "saved in your profile if accepted.\n"
        "- **Permissions:** Read-only Gmail (`gmail.readonly`) and Google Calendar event "
        "read/create (`calendar.readonly`, `calendar.events`).\n"
        "- **Credentials:** Reads `[google_oauth]` from TOML secrets or `credentials.json` if "
        "present; optional custom OAuth Client ID/Secret under *Advanced* stay in RAM only."
    )


def render_google_status_box() -> None:
    status = get_google_auth_status()
    mode = str(status.get("mode", "unconfigured"))
    msg = str(status.get("message", ""))
    accepted = _is_google_policy_accepted()

    if not accepted:
        st.caption("Accept the policy above to enable Gmail & Calendar access.")
    elif mode in {"ephemeral_ram", "token_file"}:
        st.success(msg)
    elif mode in {"toml_secrets", "credentials_file"}:
        st.info(msg)
    else:
        st.warning(msg)


def render_user_onboarding() -> None:
    st.title("Personal AI Agent")
    st.subheader("Identify Yourself to Load Your Private Workspace")
    st.caption(
        "Each user gets separate conversation history, profile preferences, and daily logs."
    )

    with st.form("user_login_form"):
        display_name = st.text_input(
            "Your Name or Workspace ID *",
            placeholder="e.g., Sarath",
            key="onboarding_display_name",
        )
        role_or_notes = st.text_input(
            "Role / Context (optional)",
            placeholder="e.g., AI Engineering Student / Project Lead",
            key="onboarding_role_notes",
        )

        st.markdown("#### Connect Google Mail & Calendar (Optional)")
        render_google_acceptance_policy()

        g_email = st.text_input(
            "Google Mail ID",
            value=_clean_session_str("session_google_email"),
            placeholder="e.g., yourname@gmail.com",
            key="onboarding_g_email",
        )
        policy_agreed = st.checkbox(
            "I accept the Google Account Sign-In & Access Policy to sign in via Google and use Gmail/Calendar.",
            value=_is_google_policy_accepted(),
            key="onboarding_policy_checkbox",
        )

        with st.expander("Advanced: Custom OAuth Client ID & Secret (RAM Only)", expanded=False):
            g_client_id = st.text_input(
                "Custom Google OAuth Client ID",
                value=_clean_session_str("ephemeral_google_client_id"),
                placeholder="xxxx.apps.googleusercontent.com",
                key="onboarding_g_cid",
            )
            g_client_secret = st.text_input(
                "Custom Google OAuth Client Secret",
                value=_clean_session_str("ephemeral_google_client_secret"),
                type="password",
                placeholder="Enter Client Secret (hidden, never saved)",
                key="onboarding_g_csec",
            )

        submitted = st.form_submit_button("Start / Resume My Session", type="primary")

    if submitted:
        clean_name = (display_name or "").strip()
        if not clean_name:
            st.warning("Please enter your name or a unique Workspace ID to continue.")
            return

        clean_email = (g_email or "").strip()
        clean_cid = (g_client_id or "").strip()
        clean_csec = (g_client_secret or "").strip()
        clean_role = (role_or_notes or "").strip()

        if (clean_email or clean_cid or clean_csec) and not policy_agreed:
            st.warning(
                "Please accept the policy checkbox to enable Google Mail & Calendar, "
                "or leave the Google fields blank to continue without Google access."
            )
            return

        st.session_state.google_policy_accepted = bool(policy_agreed)
        st.session_state.session_google_email = clean_email if policy_agreed else ""
        st.session_state.ephemeral_google_client_id = clean_cid if policy_agreed else ""
        st.session_state.ephemeral_google_client_secret = (
            clean_csec if policy_agreed else ""
        )

        user_id = set_active_user(clean_name)
        sync_ephemeral_google_from_session()
        save_user_onboarding_details(
            user_id=user_id,
            display_name=clean_name,
            role_or_notes=clean_role,
            google_email=clean_email if policy_agreed else "",
        )
        thread_id = f"{user_id}-{date.today().isoformat()}"
        st.session_state.active_user_id = user_id
        st.session_state.active_display_name = clean_name
        st.session_state.thread_id = thread_id
        st.session_state.messages = load_recent_conversation_messages(thread_id, user_id)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Personal AI Agent", page_icon="🤖", layout="wide")
    sync_streamlit_secrets()

    if "active_user_id" not in st.session_state:
        render_user_onboarding()
        return

    raw_uid = st.session_state.get("active_user_id")
    user_id = sanitize_user_id(raw_uid if isinstance(raw_uid, str) else "default_user")
    raw_dname = st.session_state.get("active_display_name")
    display_name = (
        raw_dname.strip() if isinstance(raw_dname, str) and raw_dname.strip() else user_id
    )

    set_active_user(user_id)
    sync_ephemeral_google_from_session()
    user_paths = get_user_paths(user_id)

    st.title(f"Personal AI Agent — {display_name}")

    default_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if default_provider not in {"ollama", "groq"}:
        default_provider = "ollama"

    with st.sidebar:
        st.header("Active User")
        st.info(f"**User:** {display_name} (`{user_id}`)")
        active_mail = _clean_session_str("session_google_email")
        if active_mail and _is_google_policy_accepted():
            st.caption(f"Google Mail: `{active_mail}` (Policy Accepted)")

        if st.button("Check Calendar & Unread Mail", key=f"check_google_{user_id}"):
            quick_prompt = "Check my upcoming calendar events and my unread emails."
            current_messages = st.session_state.get("messages")
            if not isinstance(current_messages, list):
                current_messages = []
                st.session_state.messages = current_messages

            active_provider = str(
                st.session_state.get("provider_select", os.getenv("LLM_PROVIDER", "groq"))
            )
            active_model = str(
                st.session_state.get("model_name_input", _default_model_for_provider(active_provider))
            )
            raw_thread = str(st.session_state.get("active_thread", "default"))
            active_scoped_thread = f"{user_id}::{raw_thread}"

            current_messages.append({"role": "user", "content": quick_prompt})
            append_daily_log(
                f"Streamlit user ({active_provider}:{active_model}): {quick_prompt}",
                user_id,
            )
            with st.spinner("Checking Calendar & Gmail..."):
                try:
                    quick_agent = get_cached_agent(
                        user_id=user_id,
                        provider=active_provider,
                        model_name=active_model,
                        context_snapshot=load_profile_and_ltm_summary(user_id),
                    )
                    result = quick_agent.invoke(
                        {"messages": [{"role": "user", "content": quick_prompt}]},
                        config={"configurable": {"thread_id": active_scoped_thread}},
                    )
                    reply = str(result["messages"][-1].content)
                except Exception as exc:
                    reply = f"Model/Agent error: {exc}"
            current_messages.append({"role": "assistant", "content": reply})
            append_daily_log(f"Streamlit assistant: {reply[:160]}", user_id)
            save_conversation_turn(active_scoped_thread, quick_prompt, reply, user_id)
            st.rerun()


        if st.button("Switch User", key=f"switch_user_{user_id}"):
            clear_ephemeral_google_credentials()
            for key in [
                "active_user_id",
                "active_display_name",
                "thread_id",
                "messages",
                "google_policy_accepted",
                "session_google_email",
                "ephemeral_google_client_id",
                "ephemeral_google_client_secret",
                f"last_provider_{user_id}",
                f"model_input_{user_id}",
            ]:
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.header("Google Mail & Calendar")
        render_google_acceptance_policy()

        sidebar_email = st.text_input(
            "Google Mail ID",
            value=_clean_session_str("session_google_email"),
            placeholder="e.g., yourname@gmail.com",
            key=f"sidebar_g_email_{user_id}",
        )
        sidebar_policy = st.checkbox(
            "Accept policy for Google sign-in, Gmail read & Calendar read/create",
            value=_is_google_policy_accepted(),
            key=f"sidebar_g_policy_{user_id}",
        )

        with st.expander("Advanced: Custom OAuth Client ID & Secret (RAM Only)", expanded=False):
            new_cid = st.text_input(
                "Custom Google Client ID",
                value=_clean_session_str("ephemeral_google_client_id"),
                key=f"sidebar_g_cid_{user_id}",
            )
            new_csec = st.text_input(
                "Custom Google Client Secret",
                value=_clean_session_str("ephemeral_google_client_secret"),
                type="password",
                key=f"sidebar_g_csec_{user_id}",
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Google", key=f"apply_google_{user_id}"):
                clean_m = (sidebar_email or "").strip()
                if (
                    clean_m or (new_cid or "").strip() or (new_csec or "").strip()
                ) and not sidebar_policy:
                    st.warning(
                        "Accept the policy checkbox first to enable Google Mail/Calendar."
                    )
                else:
                    st.session_state.google_policy_accepted = bool(sidebar_policy)
                    st.session_state.session_google_email = (
                        clean_m if sidebar_policy else ""
                    )
                    st.session_state.ephemeral_google_client_id = (
                        (new_cid or "").strip() if sidebar_policy else ""
                    )
                    st.session_state.ephemeral_google_client_secret = (
                        (new_csec or "").strip() if sidebar_policy else ""
                    )
                    sync_ephemeral_google_from_session()
                    if clean_m and sidebar_policy:
                        save_user_onboarding_details(
                            user_id=user_id,
                            display_name=display_name,
                            google_email=clean_m,
                        )
                    st.rerun()
        with col2:
            if st.button("Clear Google", key=f"clear_google_{user_id}"):
                st.session_state.google_policy_accepted = False
                st.session_state.pop("session_google_email", None)
                st.session_state.pop("ephemeral_google_client_id", None)
                st.session_state.pop("ephemeral_google_client_secret", None)
                clear_ephemeral_google_credentials()
                st.rerun()

        render_google_status_box()

        st.divider()
        st.header("Model & Provider")
        provider_key = f"provider_select_{user_id}"
        last_provider_key = f"last_provider_{user_id}"
        model_key = f"model_input_{user_id}"

        selected_provider = st.selectbox(
            "Provider",
            options=["ollama", "groq"],
            index=0 if default_provider == "ollama" else 1,
            key=provider_key,
        )

        # Reset Model Name automatically when the user switches Provider
        if st.session_state.get(last_provider_key) != selected_provider:
            st.session_state[last_provider_key] = selected_provider
            st.session_state[model_key] = _default_model_for_provider(selected_provider)
        elif model_key not in st.session_state:
            st.session_state[model_key] = _default_model_for_provider(selected_provider)

        raw_model = st.text_input(
            "Model Name",
            key=model_key,
        )
        default_model = _default_model_for_provider(selected_provider)
        selected_model = (
            raw_model.strip()
            if isinstance(raw_model, str) and raw_model.strip()
            else default_model
        )

        st.divider()
        st.header("Conversation Thread")
        default_thread = f"{user_id}-{date.today().isoformat()}"
        raw_curr_thread = st.session_state.get("thread_id")
        current_thread_val = (
            raw_curr_thread
            if isinstance(raw_curr_thread, str) and raw_curr_thread.strip()
            else default_thread
        )
        raw_entered = st.text_input(
            "Thread Name",
            value=current_thread_val,
            key=f"thread_input_{user_id}",
        )
        entered_thread = (
            raw_entered.strip()
            if isinstance(raw_entered, str) and raw_entered.strip()
            else default_thread
        )
        scoped_thread = (
            entered_thread
            if entered_thread.startswith(f"{user_id}-")
            else f"{user_id}-{sanitize_user_id(entered_thread)}"
        )
        if scoped_thread != st.session_state.get("thread_id"):
            st.session_state.thread_id = scoped_thread
            st.session_state.messages = load_recent_conversation_messages(
                scoped_thread, user_id
            )
            st.rerun()

        if st.button("Clear Screen Chat", key=f"clear_chat_{user_id}"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Your Private Memory")
        if user_paths["profile"].exists():
            with st.expander("User Profile", expanded=False):
                st.code(user_paths["profile"].read_text(encoding="utf-8"), language="json")
        if user_paths["ltm"].exists():
            with st.expander("Long-Term Memory", expanded=False):
                st.code(user_paths["ltm"].read_text(encoding="utf-8"), language="json")
        today_log = user_paths["daily"] / f"{date.today().isoformat()}.md"
        if today_log.exists():
            with st.expander("Today's Log", expanded=False):
                st.markdown(today_log.read_text(encoding="utf-8"))

    context_snapshot = load_profile_and_ltm_summary(user_id)
    try:
        agent = get_cached_agent(
            user_id=user_id,
            provider=selected_provider,
            model_name=selected_model,
            context_snapshot=context_snapshot,
        )
    except Exception as exc:
        st.error(f"Could not initialize agent: {exc}")
        st.stop()

    messages_list = st.session_state.get("messages")
    if not isinstance(messages_list, list):
        messages_list = []
        st.session_state.messages = messages_list

    for msg in messages_list:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_prompt = st.chat_input(f"Message your agent as {display_name}...")
    if not user_prompt:
        return

    messages_list.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    config: RunnableConfig = {"configurable": {"thread_id": scoped_thread}}
    append_daily_log(
        f"Streamlit user ({selected_provider}:{selected_model}): {user_prompt}",
        user_id,
    )

    spinner_text = (
        "Thinking (may open Google sign-in in your browser if needed)..."
        if _is_google_policy_accepted()
        else "Thinking..."
    )

    with st.chat_message("assistant"):
        with st.spinner(spinner_text):
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_prompt}]},
                    config=config,
                )
            except Exception as exc:
                err_msg = f"Model/Agent error: {exc}"
                st.error(err_msg)
                return
            reply = str(result["messages"][-1].content)
            st.markdown(reply)

    messages_list.append({"role": "assistant", "content": reply})
    append_daily_log(f"Streamlit assistant: {reply[:160]}", user_id)
    save_conversation_turn(scoped_thread, user_prompt, reply, user_id)


if __name__ == "__main__":
    main()