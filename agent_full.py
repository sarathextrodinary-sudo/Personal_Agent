import getpass
import json
import os
import re
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from google.auth.credentials import Credentials as BaseCredentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as OAuthUserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

BASE_DIR = Path(__file__).resolve().parent

# Automatically load environment variables from .env in the project folder
load_dotenv(dotenv_path=BASE_DIR / ".env")

MEM_ROOT = BASE_DIR / "memory" / "users"
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

# Read-only Gmail + read/write Calendar scopes (requires re-auth if expanded)
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# Active user & session-only Google state (OAuth Client ID/Secret stay in RAM only)
CURRENT_USER: ContextVar[str] = ContextVar("CURRENT_USER", default="default_user")
EPHEMERAL_GOOGLE_CLIENT_CONFIG: ContextVar[dict | None] = ContextVar(
    "EPHEMERAL_GOOGLE_CLIENT_CONFIG", default=None
)
EPHEMERAL_GOOGLE_CREDS: ContextVar[OAuthUserCredentials | None] = ContextVar(
    "EPHEMERAL_GOOGLE_CREDS", default=None
)
SESSION_GOOGLE_EMAIL: ContextVar[str] = ContextVar("SESSION_GOOGLE_EMAIL", default="")


def sanitize_user_id(raw_name: str) -> str:
    """Normalize a user's name/ID into a safe folder name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_name.strip().lower()).strip("_")
    return cleaned or "default_user"


def set_active_user(raw_name: str) -> str:
    """Set the active user context and ensure their memory directories exist."""
    user_id = sanitize_user_id(raw_name)
    CURRENT_USER.set(user_id)
    init_memory_files(user_id)
    return user_id


def get_user_paths(user_id: str | None = None) -> dict[str, Path]:
    uid = sanitize_user_id(user_id or CURRENT_USER.get())
    user_dir = MEM_ROOT / uid
    return {
        "root": user_dir,
        "daily": user_dir / "daily",
        "conv": user_dir / "conversations",
        "profile": user_dir / "user_profile.json",
        "ltm": user_dir / "long_term.json",
    }


def set_session_google_email(email: str) -> None:
    """Store optional Google Mail ID for the active session."""
    SESSION_GOOGLE_EMAIL.set(email.strip())


def get_session_google_email() -> str:
    return SESSION_GOOGLE_EMAIL.get().strip()


def set_ephemeral_google_client(
    client_id: str, client_secret: str, project_id: str = "ephemeral-session"
) -> None:
    """Store OAuth Desktop client details in memory only for the current session (never saved to disk)."""
    cid = client_id.strip()
    csec = client_secret.strip()
    if not cid or not csec:
        EPHEMERAL_GOOGLE_CLIENT_CONFIG.set(None)
        EPHEMERAL_GOOGLE_CREDS.set(None)
        return

    config_dict = {
        "installed": {
            "client_id": cid,
            "project_id": project_id.strip() or "ephemeral-session",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": csec,
            "redirect_uris": ["http://localhost"],
        }
    }
    EPHEMERAL_GOOGLE_CLIENT_CONFIG.set(config_dict)
    EPHEMERAL_GOOGLE_CREDS.set(None)


def clear_ephemeral_google_credentials() -> None:
    """Wipe user-supplied Google client details, email hint, and session token from memory."""
    EPHEMERAL_GOOGLE_CLIENT_CONFIG.set(None)
    EPHEMERAL_GOOGLE_CREDS.set(None)
    SESSION_GOOGLE_EMAIL.set("")


def get_google_auth_status() -> dict[str, object]:
    """Return a clear diagnostic snapshot of Google OAuth readiness without exposing secrets."""
    ephemeral_client = EPHEMERAL_GOOGLE_CLIENT_CONFIG.get()
    ephemeral_creds = EPHEMERAL_GOOGLE_CREDS.get()
    email_hint = get_session_google_email()

    if ephemeral_client:
        has_valid_token = bool(ephemeral_creds and ephemeral_creds.valid)
        has_refresh = bool(
            ephemeral_creds and getattr(ephemeral_creds, "refresh_token", None)
        )
        state = (
            "Authenticated (RAM session token active)"
            if (has_valid_token or has_refresh)
            else "Ready for browser sign-in via custom RAM OAuth client on first Gmail/Calendar action."
        )
        return {
            "mode": "ephemeral_ram",
            "ready": True,
            "authenticated": has_valid_token or has_refresh,
            "email_hint": email_hint,
            "message": state,
        }

    if TOKEN_FILE.exists():
        return {
            "mode": "token_file",
            "ready": True,
            "authenticated": True,
            "email_hint": email_hint,
            "message": "Saved token.json ready (will refresh automatically if needed).",
        }

    if CREDENTIALS_FILE.exists():
        return {
            "mode": "credentials_file",
            "ready": True,
            "authenticated": False,
            "email_hint": email_hint,
            "message": "credentials.json found — Google browser sign-in will launch when you read Gmail or use Calendar.",
        }

    return {
        "mode": "unconfigured",
        "ready": False,
        "authenticated": False,
        "email_hint": email_hint,
        "message": "Place credentials.json in the project folder or enter a custom OAuth Client ID & Secret under Advanced.",
    }


def get_llm():
    """Return ChatGroq or ChatOllama based on LLM_PROVIDER in .env or environment."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError(
                "LLM_PROVIDER is set to 'groq', but GROQ_API_KEY is missing in .env."
            )
        model_name = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
        print(f"Using Groq provider with model: {model_name}")
        return ChatGroq(model=model_name, temperature=0)

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        print(f"Using Ollama provider with model: {model_name}")
        ollama_kwargs: dict[str, object] = {
            "model": model_name,
            "temperature": 0,
        }
        return ChatOllama(**ollama_kwargs)  # type: ignore[arg-type]

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Expected 'ollama' or 'groq'."
    )


def init_memory_files(user_id: str | None = None) -> None:
    paths = get_user_paths(user_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["daily"].mkdir(parents=True, exist_ok=True)
    paths["conv"].mkdir(parents=True, exist_ok=True)

    if not paths["profile"].exists():
        paths["profile"].write_text(
            json.dumps(
                {
                    "user_id": paths["root"].name,
                    "display_name": paths["root"].name,
                    "google_email": "",
                    "role_or_notes": "",
                    "preferences": ["Keep responses concise and practical"],
                    "ongoing_projects": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if not paths["ltm"].exists():
        paths["ltm"].write_text(
            json.dumps(
                {
                    "decisions": [],
                    "facts": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def save_user_onboarding_details(
    user_id: str,
    display_name: str,
    role_or_notes: str = "",
    google_email: str = "",
) -> None:
    """Save initial user details entered at startup into their user_profile.json."""
    init_memory_files(user_id)
    paths = get_user_paths(user_id)
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    if display_name.strip():
        profile["display_name"] = display_name.strip()
    if role_or_notes.strip():
        profile["role_or_notes"] = role_or_notes.strip()
    if google_email.strip():
        profile["google_email"] = google_email.strip()
    paths["profile"].write_text(json.dumps(profile, indent=2), encoding="utf-8")


def append_daily_log(entry: str, user_id: str | None = None) -> None:
    paths = get_user_paths(user_id)
    today_file = paths["daily"] / f"{date.today().isoformat()}.md"
    timestamp = datetime.now().strftime("%H:%M:%S")
    with today_file.open("a", encoding="utf-8") as f:
        f.write(f"- [{timestamp}] {entry}\n")


def load_profile_and_ltm_summary(user_id: str | None = None) -> str:
    paths = get_user_paths(user_id)
    init_memory_files(user_id)
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    ltm = json.loads(paths["ltm"].read_text(encoding="utf-8"))
    today_file = paths["daily"] / f"{date.today().isoformat()}.md"
    daily_text = (
        today_file.read_text(encoding="utf-8")
        if today_file.exists()
        else "No entries yet today."
    )

    return (
        f"Active User Profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Long-Term Memory (Decisions & Facts):\n{json.dumps(ltm, indent=2)}\n\n"
        f"Today's Activity Log ({date.today().isoformat()}):\n{daily_text}"
    )


def _run_oauth_flow(flow: InstalledAppFlow) -> OAuthUserCredentials:
    """Run Google's browser OAuth consent flow with a timeout."""
    login_hint = get_session_google_email()
    print(
        "\n[Google OAuth] Opening Google browser sign-in (timeout: 90s). "
        "Sign in with your Google account in the browser..."
    )
    if login_hint:
        print(f"[Google OAuth] Sign-in hint: {login_hint}")

    auth_kwargs: dict[str, object] = {
        "port": 0,
        "open_browser": True,
        "timeout_seconds": 90,
    }
    if login_hint:
        auth_kwargs["login_hint"] = login_hint

    try:
        creds = flow.run_local_server(**auth_kwargs)  # type: ignore[arg-type]
    except TimeoutError as exc:
        raise TimeoutError(
            "Google browser sign-in timed out after 90 seconds. Check your browser for the Google login page "
            "and try again."
        ) from exc

    return cast(OAuthUserCredentials, creds)


def get_google_credentials() -> BaseCredentials:
    """
    Authenticate via Google OAuth using session-only client details if provided,
    otherwise using local project credentials.json/token.json.
    """
    ephemeral_client = EPHEMERAL_GOOGLE_CLIENT_CONFIG.get()
    ephemeral_creds = EPHEMERAL_GOOGLE_CREDS.get()

    if ephemeral_client:
        if ephemeral_creds and ephemeral_creds.valid:
            return ephemeral_creds
        if (
            ephemeral_creds
            and ephemeral_creds.expired
            and getattr(ephemeral_creds, "refresh_token", None)
        ):
            ephemeral_creds.refresh(Request())
            EPHEMERAL_GOOGLE_CREDS.set(ephemeral_creds)
            return ephemeral_creds

        flow = InstalledAppFlow.from_client_config(ephemeral_client, GOOGLE_SCOPES)
        new_creds = _run_oauth_flow(flow)
        EPHEMERAL_GOOGLE_CREDS.set(new_creds)
        return new_creds

    creds: BaseCredentials | None = None
    if TOKEN_FILE.exists():
        creds = OAuthUserCredentials.from_authorized_user_file(
            str(TOKEN_FILE), GOOGLE_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and getattr(creds, "refresh_token", None):
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    "credentials.json was not found in the project folder and no custom OAuth client "
                    "was entered under Advanced. Add credentials.json or enter a custom OAuth Client ID & Secret."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), GOOGLE_SCOPES
            )
            creds = _run_oauth_flow(flow)

        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _parse_iso_with_tz(dt_str: str, tz_name: str) -> datetime:
    """Parse an ISO-8601 datetime string and attach timezone if naive."""
    tz = ZoneInfo(tz_name)
    parsed = datetime.fromisoformat(dt_str.strip())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


# ---------------- MEMORY TOOLS (SCOPED TO ACTIVE USER) ----------------


@tool
def update_user_profile(
    section: Literal["preferences", "ongoing_projects"], value: str
) -> str:
    """Save a user preference or an ongoing project into the active user's persistent profile."""
    paths = get_user_paths()
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    items = profile.setdefault(section, [])
    if value not in items:
        items.append(value)
    paths["profile"].write_text(json.dumps(profile, indent=2), encoding="utf-8")
    append_daily_log(f"Updated profile ({section}): {value}")
    return f"Saved '{value}' under profile.{section}."


@tool
def save_long_term_memory(kind: Literal["decisions", "facts"], item: str) -> str:
    """Save a durable decision made or an important fact into the active user's long-term memory."""
    paths = get_user_paths()
    ltm = json.loads(paths["ltm"].read_text(encoding="utf-8"))
    bucket = ltm.setdefault(kind, [])
    entry = {"recorded_on": date.today().isoformat(), "content": item}
    bucket.append(entry)
    paths["ltm"].write_text(json.dumps(ltm, indent=2), encoding="utf-8")
    append_daily_log(f"Saved long-term {kind[:-1]}: {item}")
    return f"Saved to long-term {kind}: {item}"


@tool
def log_daily_activity(note: str) -> str:
    """Record a recent chat takeaway, task, or activity in today's daily memory log."""
    append_daily_log(note)
    return "Logged to today's daily memory."


@tool
def search_daily_memory(day_iso: str | None = None) -> str:
    """Read the active user's daily memory for a specific date (YYYY-MM-DD), defaulting to today."""
    paths = get_user_paths()
    target_day = day_iso or date.today().isoformat()
    target_file = paths["daily"] / f"{target_day}.md"
    if not target_file.exists():
        return f"No daily memory found for {target_day}."
    return target_file.read_text(encoding="utf-8")


# ---------------- GOOGLE CONNECTIVITY TOOLS ----------------


@tool
def search_gmail(query: str = "is:unread", max_results: int = 5) -> str:
    """Search Gmail messages in read-only mode using a Gmail search query."""
    try:
        creds = get_google_credentials()
        service = build("gmail", "v1", credentials=creds)
        res = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max(1, min(max_results, 15)))
            .execute()
        )
        messages = res.get("messages", [])
        if not messages:
            append_daily_log(f"GMail read-only search '{query}': 0 matches")
            return f"No Gmail messages found for query: {query}"

        summaries = []
        for msg_meta in messages:
            msg = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=msg_meta["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From", "Date"],
                )
                .execute()
            )
            headers = {
                h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])
            }
            snippet = msg.get("snippet", "").strip()
            summaries.append(
                f"- From: {headers.get('From', 'Unknown')}\n"
                f"  Subject: {headers.get('Subject', '(No Subject)')}\n"
                f"  Date: {headers.get('Date', '')}\n"
                f"  Snippet: {snippet}"
            )

        append_daily_log(f"GMail read-only search '{query}': {len(summaries)} matches")
        return "\n".join(summaries)
    except Exception as exc:
        return f"Gmail read-only error: {exc}"


@tool
def list_upcoming_calendar_events(max_results: int = 10) -> str:
    """List upcoming events from the primary Google Calendar in read-only mode."""
    try:
        creds = get_google_credentials()
        service = build("calendar", "v3", credentials=creds)
        now_iso = datetime.now(timezone.utc).isoformat()
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now_iso,
                maxResults=max(1, min(max_results, 25)),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        if not events:
            append_daily_log("GCalendar read-only check: no upcoming events")
            return "No upcoming events found on your primary Google Calendar."

        lines = []
        for event in events:
            start = event.get("start", {}).get("dateTime") or event.get("start", {}).get(
                "date", "Unknown start"
            )
            end = event.get("end", {}).get("dateTime") or event.get("end", {}).get(
                "date", "Unknown end"
            )
            summary = event.get("summary", "(No Title)")
            lines.append(f"- {summary} | Start: {start} | End: {end}")

        append_daily_log(f"GCalendar read-only check: {len(lines)} events")
        return "\n".join(lines)
    except Exception as exc:
        return f"Google Calendar read-only error: {exc}"


@tool
def create_calendar_event(
    summary: str,
    start_iso: str,
    end_iso: str | None = None,
    description: str = "",
    location: str = "",
    timezone_name: str = "Asia/Kolkata",
) -> str:
    """
    Create a new event on the user's primary Google Calendar.
    `start_iso` and `end_iso` should be ISO-8601 strings (e.g., '2026-11-23T15:00:00').
    If `end_iso` is omitted, defaults to 30 minutes after `start_iso`.
    """
    try:
        creds = get_google_credentials()
        service = build("calendar", "v3", credentials=creds)

        start_dt = _parse_iso_with_tz(start_iso, timezone_name)
        end_dt = (
            _parse_iso_with_tz(end_iso, timezone_name)
            if end_iso
            else start_dt + timedelta(minutes=30)
        )

        event_body: dict[str, object] = {
            "summary": summary.strip(),
            "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone_name},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": timezone_name},
        }
        if description.strip():
            event_body["description"] = description.strip()
        if location.strip():
            event_body["location"] = location.strip()

        created = (
            service.events()
            .insert(calendarId="primary", body=event_body)
            .execute()
        )
        link = created.get("htmlLink", "")
        append_daily_log(f"Created calendar event: '{summary}' at {start_dt.isoformat()}")
        return (
            f"Created event '{created.get('summary', summary)}' "
            f"from {start_dt.strftime('%Y-%m-%d %H:%M %Z')} to {end_dt.strftime('%Y-%m-%d %H:%M %Z')}."
            + (f"\nLink: {link}" if link else "")
        )
    except Exception as exc:
        return f"Google Calendar create error: {exc}"


@tool
def search_conversations(keyword: str) -> str:
    """Search the active user's saved conversation transcripts for a keyword."""
    paths = get_user_paths()
    matches = []
    for path in sorted(paths["conv"].glob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if keyword.lower() in text.lower():
                matches.append(f"--- {path.name} ---\n{text[:600]}")
    return (
        "\n\n".join(matches)
        if matches
        else f"No past conversation files matched '{keyword}'."
    )


def save_conversation_turn(
    thread_id: str, user_msg: str, assistant_msg: str, user_id: str | None = None
) -> None:
    paths = get_user_paths(user_id)
    conv_file = paths["conv"] / f"{thread_id}.jsonl"
    record = {
        "timestamp": datetime.now().isoformat(),
        "user": user_msg,
        "assistant": assistant_msg,
    }
    with conv_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_recent_conversation_messages(
    thread_id: str, user_id: str | None = None
) -> list[dict[str, str]]:
    """Reload saved messages for a user's thread when returning to the app."""
    paths = get_user_paths(user_id)
    conv_file = paths["conv"] / f"{thread_id}.jsonl"
    if not conv_file.exists():
        return []
    messages: list[dict[str, str]] = []
    for line in conv_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["assistant"]})
    return messages


def prompt_ephemeral_google_details() -> str:
    print("\n--- Google Mail & Calendar Access (Google Browser Sign-In) ---")
    print(
        "Enter your Google Mail ID to pre-hint your account on Google's browser sign-in page. "
        "We never ask for or store your Google password. Optional custom OAuth Client ID/Secret "
        "stay in RAM only if provided."
    )
    google_email = input("Google Mail ID (optional, press Enter to skip): ").strip()
    set_session_google_email(google_email)

    client_id = input("Custom OAuth Client ID (optional, press Enter to skip): ").strip()
    if not client_id:
        set_ephemeral_google_client("", "")
        return google_email

    client_secret = getpass.getpass("Custom OAuth Client Secret (hidden input): ").strip()
    if not client_secret:
        set_ephemeral_google_client("", "")
        print("Client Secret was empty; skipped custom OAuth client.\n")
        return google_email

    set_ephemeral_google_client(client_id=client_id, client_secret=client_secret)
    print("Custom OAuth client loaded in RAM for this session only (not saved).\n")
    return google_email


def main() -> None:
    print("=== Personal AI Agent Setup ===")
    raw_name = input("Enter your name or User ID: ").strip() or "default_user"
    role_notes = input("Role / brief note (optional, press Enter to skip): ").strip()

    user_id = set_active_user(raw_name)
    google_email = prompt_ephemeral_google_details()
    save_user_onboarding_details(
        user_id=user_id,
        display_name=raw_name,
        role_or_notes=role_notes,
        google_email=google_email,
    )

    llm = get_llm()
    checkpointer = InMemorySaver()
    context_snapshot = load_profile_and_ltm_summary(user_id)

    system_prompt = (
        "You are a personal AI assistant running in VS Code. "
        "Always address and personalize responses for the active user below. "
        "Use memory tools when the user shares preferences, ongoing projects, decisions, facts, or daily updates.\n\n"
        f"{context_snapshot}"
    )

    agent = create_agent(
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
        checkpointer=checkpointer,
    )

    thread_id = f"{user_id}-{date.today().isoformat()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    print(f"\nWelcome, {raw_name}! Isolated profile active at: memory/users/{user_id}/")
    print(f"Thread: {thread_id}. Type 'exit' to quit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            append_daily_log(f"Chat user: {user_input}", user_id)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            reply = result["messages"][-1].content
            append_daily_log(f"Chat assistant: {reply[:160]}", user_id)
            save_conversation_turn(thread_id, user_input, reply, user_id)
            print(f"\nAgent: {reply}\n")
    finally:
        clear_ephemeral_google_credentials()


if __name__ == "__main__":
    main()