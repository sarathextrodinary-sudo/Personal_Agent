import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


BASE_DIR = Path(__file__).resolve().parent

# Automatically load environment variables from .env in the project folder
load_dotenv(dotenv_path=BASE_DIR / ".env")

MEM_DIR = BASE_DIR / "memory"
DAILY_DIR = MEM_DIR / "daily"
CONV_DIR = MEM_DIR / "conversations"
PROFILE_FILE = MEM_DIR / "user_profile.json"
LONG_TERM_FILE = MEM_DIR / "long_term.json"

CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

# Read-only scopes for Gmail and Google Calendar
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
]


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
        return ChatOllama(model=model_name, temperature=0)

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Expected 'ollama' or 'groq'."
    )


def init_memory_files() -> None:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    CONV_DIR.mkdir(parents=True, exist_ok=True)

    if not PROFILE_FILE.exists():
        PROFILE_FILE.write_text(
            json.dumps(
                {
                    "preferences": ["Keep responses concise and practical"],
                    "ongoing_projects": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if not LONG_TERM_FILE.exists():
        LONG_TERM_FILE.write_text(
            json.dumps(
                {
                    "decisions": [],
                    "facts": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def append_daily_log(entry: str) -> None:
    today_file = DAILY_DIR / f"{date.today().isoformat()}.md"
    timestamp = datetime.now().strftime("%H:%M:%S")
    with today_file.open("a", encoding="utf-8") as f:
        f.write(f"- [{timestamp}] {entry}\n")


def load_profile_and_ltm_summary() -> str:
    profile = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    ltm = json.loads(LONG_TERM_FILE.read_text(encoding="utf-8"))
    today_file = DAILY_DIR / f"{date.today().isoformat()}.md"
    daily_text = (
        today_file.read_text(encoding="utf-8")
        if today_file.exists()
        else "No entries yet today."
    )

    return (
        f"User Profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Long-Term Memory (Decisions & Facts):\n{json.dumps(ltm, indent=2)}\n\n"
        f"Today's Activity Log ({date.today().isoformat()}):\n{daily_text}"
    )


from google.auth.credentials import Credentials as BaseCredentials

def get_google_credentials() -> BaseCredentials:
    """Authenticate via OAuth Desktop flow and cache the token locally."""
    creds: BaseCredentials | None = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), GOOGLE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and getattr(creds, "refresh_token", None):
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_FILE.name}. Download OAuth Desktop credentials "
                    "from Google Cloud Console and save them as credentials.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), GOOGLE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds

# ---------------- MEMORY TOOLS ----------------


@tool
def update_user_profile(
    section: Literal["preferences", "ongoing_projects"], value: str
) -> str:
    """Save a user preference or an ongoing project into the persistent user profile."""
    profile = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    items = profile.setdefault(section, [])
    if value not in items:
        items.append(value)
    PROFILE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    append_daily_log(f"Updated profile ({section}): {value}")
    return f"Saved '{value}' under profile.{section}."


@tool
def save_long_term_memory(kind: Literal["decisions", "facts"], item: str) -> str:
    """Save a durable decision made or an important fact into long-term memory."""
    ltm = json.loads(LONG_TERM_FILE.read_text(encoding="utf-8"))
    bucket = ltm.setdefault(kind, [])
    entry = {"recorded_on": date.today().isoformat(), "content": item}
    bucket.append(entry)
    LONG_TERM_FILE.write_text(json.dumps(ltm, indent=2), encoding="utf-8")
    append_daily_log(f"Saved long-term {kind[:-1]}: {item}")
    return f"Saved to long-term {kind}: {item}"


@tool
def log_daily_activity(note: str) -> str:
    """Record a recent chat takeaway, task, or activity in today's daily memory log."""
    append_daily_log(note)
    return "Logged to today's daily memory."


@tool
def search_daily_memory(day_iso: str | None = None) -> str:
    """Read daily memory for a specific date (YYYY-MM-DD), defaulting to today."""
    target_day = day_iso or date.today().isoformat()
    target_file = DAILY_DIR / f"{target_day}.md"
    if not target_file.exists():
        return f"No daily memory found for {target_day}."
    return target_file.read_text(encoding="utf-8")


# ---------------- READ-ONLY GOOGLE CONNECTIVITY TOOLS ----------------


@tool
def search_gmail(query: str = "is:unread", max_results: int = 5) -> str:
    """Search Gmail messages in read-only mode using a Gmail search query (e.g., 'is:unread', 'from:hr', 'subject:leave')."""
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
    """List upcoming events from the user's primary Google Calendar in read-only mode."""
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
def slack_action(
    channel: str, message: str, mode: Literal["read_recent", "post"] = "read_recent"
) -> str:
    """Read recent Slack messages or post a message to a channel. Uses local mock until Slack token is wired."""
    append_daily_log(f"Slack {mode} in #{channel}: {message}")
    if mode == "read_recent":
        return f"[Slack mock - #{channel}] No live Slack token configured yet; last check noted: {message}"
    return f"[Slack mock - post to #{channel}] Queued message: {message}"


@tool
def search_conversations(keyword: str) -> str:
    """Search saved past conversation transcripts in memory/conversations/ for a keyword."""
    matches = []
    for path in sorted(CONV_DIR.glob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if keyword.lower() in text.lower():
                matches.append(f"--- {path.name} ---\n{text[:600]}")
    return (
        "\n\n".join(matches)
        if matches
        else f"No past conversation files matched '{keyword}'."
    )


def save_conversation_turn(thread_id: str, user_msg: str, assistant_msg: str) -> None:
    conv_file = CONV_DIR / f"{thread_id}.jsonl"
    record = {
        "timestamp": datetime.now().isoformat(),
        "user": user_msg,
        "assistant": assistant_msg,
    }
    with conv_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    init_memory_files()
    llm = get_llm()
    checkpointer = InMemorySaver()

    context_snapshot = load_profile_and_ltm_summary()
    system_prompt = (
        "You are a personal AI assistant running in VS Code. "
        "Use your memory tools when the user shares preferences, ongoing projects, decisions, facts, or daily updates. "
        "Use read-only Gmail and Google Calendar tools when asked about emails or upcoming schedule, "
        "and search_conversations for past chat history.\n\n"
        f"{context_snapshot}"
    )

    thread_id = f"session-{date.today().isoformat()}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    agent = create_agent(
        model=llm,
        tools=[
            update_user_profile,
            save_long_term_memory,
            log_daily_activity,
            search_daily_memory,
            search_gmail,
            list_upcoming_calendar_events,
            slack_action,
            search_conversations,
        ],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )

    thread_id = f"session-{date.today().isoformat()}"
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Agent initialized (thread: {thread_id}). Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        append_daily_log(f"Chat user: {user_input}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        reply = result["messages"][-1].content
        append_daily_log(f"Chat assistant: {reply[:160]}")
        save_conversation_turn(thread_id, user_input, reply)
        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    main()