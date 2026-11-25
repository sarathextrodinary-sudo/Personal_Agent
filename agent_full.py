import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import SecretStr

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]

os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

def build_client_config(
    client_id: str, client_secret: str, redirect_uri: str
) -> Dict[str, Any]:
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }


def _oauth_signing_key(client_secret: str) -> bytes:
    return hashlib.sha256(f"oauth-state:{client_secret}".encode("utf-8")).digest()


def create_signed_state(
    client_secret: str, code_verifier: str, ttl_seconds: int = 900
) -> str:
    """Create a stateless, tamper-evident OAuth state token containing PKCE verifier + expiry."""
    exp = int(time.time()) + ttl_seconds
    nonce = secrets.token_urlsafe(16)
    encoded_verifier = base64.urlsafe_b64encode(
        code_verifier.encode("utf-8")
    ).decode("ascii")
    payload = f"{exp}.{nonce}.{encoded_verifier}".encode("utf-8")
    sig = hmac.new(
        _oauth_signing_key(client_secret), payload, hashlib.sha256
    ).hexdigest()
    token = f"{exp}.{nonce}.{encoded_verifier}.{sig}".encode("utf-8")
    return base64.urlsafe_b64encode(token).decode("ascii")


def verify_and_extract_state(state: str, client_secret: str) -> Optional[str]:
    """Verify HMAC signature and expiration; return code_verifier if valid."""
    try:
        decoded = base64.urlsafe_b64decode(state.encode("ascii")).decode("utf-8")
        exp_str, nonce, encoded_verifier, sig = decoded.split(".", 3)
        if not nonce or not encoded_verifier or int(time.time()) > int(exp_str):
            return None
        payload = f"{exp_str}.{nonce}.{encoded_verifier}".encode("utf-8")
        expected_sig = hmac.new(
            _oauth_signing_key(client_secret), payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        return base64.urlsafe_b64decode(encoded_verifier.encode("ascii")).decode(
            "utf-8"
        )
    except Exception:
        return None


def create_authorization_url(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> str:
    code_verifier = secrets.token_urlsafe(64)
    state = create_signed_state(client_secret, code_verifier=code_verifier)

    flow = Flow.from_client_config(
        build_client_config(client_id, client_secret, redirect_uri),
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False,
    )
    code_challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=state,
        prompt="consent",
        code_challenge=code_challenge,
        code_challenge_method="S256",
    )
    return auth_url


def exchange_oauth_code(
    code: str,
    received_state: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> Tuple[Dict[str, Any], str]:
    code_verifier = verify_and_extract_state(received_state, client_secret)
    if not code_verifier:
        raise ValueError("Invalid or expired OAuth state parameter.")

    flow = Flow.from_client_config(
        build_client_config(client_id, client_secret, redirect_uri),
        scopes=SCOPES,
        state=received_state,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False,
    )
    flow.fetch_token(code=code, code_verifier=code_verifier)
    creds = flow.credentials

    oauth2_service = build("oauth2", "v2", credentials=creds)
    user_info = oauth2_service.userinfo().get().execute()

    cred_dict = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": getattr(creds, "token_uri", None)
        or "https://oauth2.googleapis.com/token",
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }
    return cred_dict, user_info.get("email", "Connected User")


def creds_from_dict(raw: Optional[Dict[str, Any]]) -> Optional[Credentials]:
    if not raw:
        return None
    return Credentials(
        token=raw["token"],
        refresh_token=raw.get("refresh_token"),
        token_uri=raw["token_uri"],
        client_id=raw["client_id"],
        client_secret=raw["client_secret"],
        scopes=raw.get("scopes", []),
    )


def fetch_upcoming_meetings(
    creds: Credentials, max_results: int = 8
) -> List[Dict[str, str]]:
    service = build("calendar", "v3", credentials=creds)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now_iso,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    meetings: List[Dict[str, str]] = []
    for ev in events_result.get("items", []):
        start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get(
            "date", ""
        )
        meetings.append(
            {
                "summary": ev.get("summary", "(No title)"),
                "start": start,
                "location": ev.get("location", "").strip(),
            }
        )
    return meetings


def search_gmail_messages_raw(
    creds: Credentials,
    query: str = "",
    max_results: int = 5,
) -> List[Dict[str, str]]:
    """Search Gmail messages using standard Gmail search syntax (e.g. 'is:unread', 'from:hr@company.com', 'flight')."""
    service = build("gmail", "v1", credentials=creds)
    safe_limit = max(1, min(int(max_results), 10))

    list_kwargs: Dict[str, Any] = {
        "userId": "me",
        "maxResults": safe_limit,
    }
    if query.strip():
        list_kwargs["q"] = query.strip()

    resp = service.users().messages().list(**list_kwargs).execute()
    messages = resp.get("messages", [])

    results: List[Dict[str, str]] = []
    for item in messages:
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=item["id"],
                format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            )
            .execute()
        )
        headers = {
            h.get("name", ""): h.get("value", "")
            for h in msg.get("payload", {}).get("headers", [])
        }
        results.append(
            {
                "id": item.get("id", ""),
                "subject": headers.get("Subject", "(No subject)"),
                "from": headers.get("From", "(Unknown sender)"),
                "date": headers.get("Date", ""),
                "snippet": msg.get("snippet", ""),
            }
        )
    return results


def geocode_place(place: str) -> Optional[Tuple[float, float, str]]:
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": place, "format": "json", "limit": 1},
        headers={"User-Agent": "PersonalAgentAchyut/1.0"},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]


def get_route_distance_and_duration(
    origin: Tuple[float, float], dest: Tuple[float, float]
) -> Tuple[float, float]:
    lat1, lon1 = origin
    lat2, lon2 = dest
    try:
        osrm_url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}"
        )
        resp = requests.get(osrm_url, params={"overview": "false"}, timeout=8)
        resp.raise_for_status()
        route = resp.json()["routes"][0]
        distance_km = round(route["distance"] / 1000.0, 2)
        duration_min = round(route["duration"] / 60.0, 1)
        return max(distance_km, 0.5), max(duration_min, 3.0)
    except Exception:
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        km = round(2 * r * math.asin(math.sqrt(a)) * 1.35, 2)
        mins = round(km * 2.5, 1)
        return max(km, 0.5), max(mins, 3.0)


def compare_ride_fares(
    distance_km: float, duration_min: float, surge_multiplier: float = 1.0
) -> List[Dict[str, Any]]:
    # Format: (provider, ride_type, mode, base_inr, per_km_inr, per_min_inr, min_fare_inr)
    rate_cards = [
        ("Rapido", "Bike Taxi", "Bike", 15.0, 6.0, 0.5, 25),
        ("Ola", "Bike", "Bike", 18.0, 6.5, 0.5, 25),
        ("Uber", "Moto", "Bike", 18.0, 7.0, 0.5, 28),
        ("Rapido", "Auto", "Auto", 25.0, 11.0, 1.0, 35),
        ("Ola", "Auto", "Auto", 28.0, 12.0, 1.0, 40),
        ("Uber", "Auto", "Auto", 30.0, 12.5, 1.0, 40),
        ("Rapido", "Cab Economy", "Cab", 45.0, 14.0, 1.0, 65),
        ("Ola", "Mini / Prime", "Cab", 50.0, 15.0, 1.2, 75),
        ("Uber", "Uber Go", "Cab", 55.0, 16.0, 1.2, 80),
        ("GREEN SM", "EV Sedan / SUV", "EV Cab", 50.0, 14.0, 1.0, 75),
    ]
    rows: List[Dict[str, Any]] = []
    for provider, category, mode, base, per_km, per_min, min_fare in rate_cards:
        raw_est = (base + (distance_km * per_km) + (duration_min * per_min)) * surge_multiplier
        est = max(min_fare, round(raw_est))
        low = max(min_fare, int(round(est * 0.95)))
        high = max(low, int(round(est * 1.08)))
        rows.append(
            {
                "provider": provider,
                "ride_type": category,
                "mode": mode,
                "estimated_fare_inr": f"₹{low} – ₹{high}",
                "midpoint_inr": est,
                "est_time": f"{int(round(duration_min))} mins",
            }
        )
    rows.sort(key=lambda r: r["midpoint_inr"])
    return rows


def build_commute_agent(
    groq_api_key: str,
    creds_dict: Optional[Dict[str, Any]] = None,
    model_name: str = "llama-3.3-70b-versatile",
):
    """Build a LangGraph agent with Calendar meeting lookup, Gmail search, and ride fare comparison tools."""

    @tool
    def get_calendar_meetings(max_results: int = 8) -> str:
        """Fetch upcoming Google Calendar events along with their scheduled start time and location."""
        creds = creds_from_dict(creds_dict)
        if creds is None:
            return json.dumps(
                {
                    "connected": False,
                    "message": "User has not connected their Google Calendar yet.",
                }
            )
        try:
            events = fetch_upcoming_meetings(creds, max_results=max_results)
            return json.dumps({"connected": True, "meetings": events})
        except Exception as exc:
            return json.dumps({"connected": False, "error": str(exc)})

    @tool
    def compare_commute_fares(
        pickup_location: str,
        destination_location: str,
        surge_multiplier: float = 1.0,
    ) -> str:
        """Compare estimated bike, auto, cab, and EV cab fares across Rapido, Ola, Uber, and GREEN SM between pickup_location and destination_location."""
        origin_geo = geocode_place(pickup_location)
        dest_geo = geocode_place(destination_location)
        if not origin_geo or not dest_geo:
            return json.dumps(
                {
                    "error": "Could not geocode pickup or destination. Please include area/city details."
                }
            )

        dist_km, dur_min = get_route_distance_and_duration(
            (origin_geo[0], origin_geo[1]),
            (dest_geo[0], dest_geo[1]),
        )
        fares = compare_ride_fares(dist_km, dur_min, surge_multiplier=surge_multiplier)
        return json.dumps(
            {
                "pickup_resolved": origin_geo[2],
                "destination_resolved": dest_geo[2],
                "distance_km": dist_km,
                "duration_min": dur_min,
                "surge_multiplier": surge_multiplier,
                "fares": fares,
            }
        )

    @tool
    def search_gmail_messages(query: str = "", max_results: int = 5) -> str:
        """Search the user's Gmail inbox/messages using a Gmail search query (such as 'meeting', 'invoice', 'is:unread', or a sender name) and return subject, sender, date, and snippet."""
        creds = creds_from_dict(creds_dict)
        if creds is None:
            return json.dumps(
                {
                    "connected": False,
                    "message": "User has not connected their Google account yet.",
                }
            )
        try:
            emails = search_gmail_messages_raw(
                creds, query=query, max_results=max_results
            )
            return json.dumps({"connected": True, "query": query, "messages": emails})
        except Exception as exc:
            return json.dumps({"connected": False, "error": str(exc)})

    resolved_key = str(groq_api_key or os.environ.get("GROQ_API_KEY") or "").strip()
    llm = ChatGroq(
        model=str(model_name),
        api_key=SecretStr(resolved_key) if resolved_key else None,
        temperature=0.2,
    )

    system_prompt = (
        "You are a helpful personal assistant in India. "
        "When a user asks about upcoming meetings or wants to travel to a meeting location, "
        "call `get_calendar_meetings` first to check meeting locations. "
        "When a user asks to check, find, or summarize emails, call `search_gmail_messages` "
        "with a concise Gmail search query. "
        "When comparing ride prices between two places or to an office/meeting venue, "
        "call `compare_commute_fares` and present a clear comparison across Rapido, Ola, Uber, and GREEN SM "
        "in INR (₹), noting that fares are rough estimates rather than live app quotes."
    )

    return create_agent(
        model=llm,
        tools=[get_calendar_meetings, compare_commute_fares, search_gmail_messages],
        system_prompt=system_prompt,
    )


def run_agent_turn(
    agent,
    history: List[Dict[str, str]],
    user_message: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Run one conversational turn through LangGraph and return assistant reply + tool trace."""
    lc_messages = []
    for msg in history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    lc_messages.append(HumanMessage(content=user_message))

    state_out = agent.invoke({"messages": lc_messages})
    all_msgs = state_out.get("messages", [])

    tool_events: List[Dict[str, str]] = []
    final_reply = ""
    for m in all_msgs:
        if isinstance(m, ToolMessage):
            tool_events.append({"tool": str(m.name or "tool"), "output": str(m.content)})
        elif isinstance(m, AIMessage) and m.content:
            final_reply = str(m.content)

    return final_reply, tool_events