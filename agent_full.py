import base64
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import random
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

os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]

# Replace or extend this dictionary once you share your full dataset
DEFAULT_STAR_DATASET: Dict[str, Dict[str, str]] = {
    "Janma (1st / 10th / 19th)": {
        "outlook": "Mixed / Caution",
        "guidance": "Good for routine self-care and planning; avoid impulsive risks or heated arguments.",
    },
    "Sampat (2nd / 11th / 20th)": {
        "outlook": "Very Favorable",
        "guidance": "Strong day for financial decisions, purchases, networking, and productive meetings.",
    },
    "Vipat (3rd / 12th / 21st)": {
        "outlook": "Challenging",
        "guidance": "Double-check travel plans, communications, and commitments; keep a calm pace.",
    },
    "Kshema (4th / 13th / 22nd)": {
        "outlook": "Favorable",
        "guidance": "Supportive for stability, family matters, learning, and steady progress.",
    },
    "Pratyak (5th / 14th / 23rd)": {
        "outlook": "Moderate Obstacles",
        "guidance": "Expect minor delays; stick to familiar workflows and postpone high-stakes negotiations.",
    },
    "Sadhana (6th / 15th / 24th)": {
        "outlook": "Highly Favorable",
        "guidance": "Excellent for completing pending tasks, interviews, execution, and achieving goals.",
    },
    "Naidhana (7th / 16th / 25th)": {
        "outlook": "Caution",
        "guidance": "Prioritize rest, health, and low-risk work; avoid major new beginnings.",
    },
    "Mitra (8th / 17th / 26th)": {
        "outlook": "Favorable",
        "guidance": "Great for teamwork, collaborations, reaching out to friends, and creative work.",
    },
    "Parama Mitra (9th / 18th / 27th)": {
        "outlook": "Very Favorable",
        "guidance": "Auspicious for important discussions, travel, learning, and long-term initiatives.",
    },
}

TARA_KEYS = list(DEFAULT_STAR_DATASET.keys())

# Replace or extend this list with your own facts dataset
DEFAULT_FACTS_DATASET: List[str] = [
    "A day on Venus is longer than a year on Venus because it rotates on its axis more slowly than it orbits the Sun.",
    "Octopuses have three hearts, nine brains, and blue copper-based blood.",
    "Honey never spoils when kept sealed and dry—archaeologists have found pots of honey thousands of years old that were still edible.",
    "Bananas are berries botanically, whereas strawberries are not.",
    "Water can boil and freeze at the same time under the right combination of temperature and pressure (the triple point).",
]

DEFAULT_FACTS_DATASET: List[str] = [
    (
        "c. 800 BCE — Baudhayana's Sulba Sutra stated the right-triangle diagonal rule "
        "(known globally as the Pythagorean theorem) along with Pythagorean triples and "
        "an approximation of √2 accurate to five decimal places."
    ),
    (
        "c. 400 BCE — Texts such as the Arthashastra documented systematic rainfall measurement "
        "using rain gauges to estimate agricultural yield and classify land for revenue."
    ),
    (
        "c. 600 BCE — The Sushruta Samhita recorded detailed surgical anatomy, named over "
        "a hundred surgical instruments, and described reconstructive procedures such as rhinoplasty."
    ),
    (
        "c. 5000–4000 BCE — Archaeobotanical and textile evidence shows cotton cultivation and "
        "spinning in the Indus Valley well before its wider spread across the ancient world."
    ),
    (
        "499 CE — In the Aryabhatiya, Aryabhata calculated the length of the sidereal year as "
        "365 days, 6 hours, 12 minutes, and 30 seconds (within minutes of modern measurements) "
        "and gave a close approximation of π (pi)."
    ),
    (
        "499 CE — Aryabhata explained lunar and solar eclipses as shadows cast by the Earth "
        "and Moon rather than mythical swallowing, and described Earth's axial rotation."
    ),
    (
        "628 CE — Brahmagupta's Brahmasphutasiddhanta formally established rules for arithmetic "
        "operations with zero (shunya) and negative numbers as mathematical quantities."
    ),
    (
        "c. 2500–500 BCE — Archaeochemical residue analysis and Ayurvedic traditions trace the "
        "culinary and medicinal use of spices like turmeric, ginger, and garlic in India back "
        "millennia, with modern pharmacology confirming bioactive compounds such as curcumin."
    ),
]

def pick_surprise_fact(
    custom_facts: Optional[List[str]] = None,
    exclude_fact: Optional[str] = None,
) -> str:
    pool = [f.strip() for f in (custom_facts or DEFAULT_FACTS_DATASET) if str(f).strip()]
    if not pool:
        return "No facts available in the dataset."
    candidates = [f for f in pool if f != exclude_fact] if len(pool) > 1 else pool
    return random.choice(candidates)


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
        include_granted_scopes="false",
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
    """Search Gmail messages using standard Gmail search syntax."""
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
        raw_est = (
            base + (distance_km * per_km) + (duration_min * per_min)
        ) * surge_multiplier
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


def find_nearby_tourist_places(area: str, limit: int = 6) -> List[Dict[str, Any]]:
    geo = geocode_place(area)
    if not geo:
        return []
    lat, lon, resolved_name = geo
    delta = 0.08
    viewbox = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": f"tourist attraction near {area}",
            "format": "json",
            "limit": limit,
            "viewbox": viewbox,
            "bounded": 1,
        },
        headers={"User-Agent": "PersonalAgentAchyut/1.0"},
        timeout=8,
    )
    resp.raise_for_status()
    places: List[Dict[str, Any]] = []
    for item in resp.json():
        p_lat, p_lon = float(item["lat"]), float(item["lon"])
        dist_km, dur_min = get_route_distance_and_duration((lat, lon), (p_lat, p_lon))
        places.append(
            {
                "name": item.get("display_name", "").split(",")[0],
                "full_address": item.get("display_name", ""),
                "distance_km": dist_km,
                "est_drive_mins": int(round(dur_min)),
                "from_area": resolved_name,
            }
        )
    places.sort(key=lambda x: x["distance_km"])
    return places


def evaluate_natal_star_day(
    natal_star: str,
    todays_star: str,
    custom_dataset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    norm_map = {s.lower(): idx for idx, s in enumerate(NAKSHATRAS)}
    n_idx = norm_map.get(natal_star.strip().lower())
    t_idx = norm_map.get(todays_star.strip().lower())
    if n_idx is None or t_idx is None:
        return {"error": f"Unknown star name. Choose from: {', '.join(NAKSHATRAS)}"}

    count = ((t_idx - n_idx) % 27) + 1
    tara_index = (count - 1) % 9
    tara_label = TARA_KEYS[tara_index]

    dataset = custom_dataset if custom_dataset else DEFAULT_STAR_DATASET
    entry = (
        dataset.get(natal_star)
        or dataset.get(tara_label)
        or DEFAULT_STAR_DATASET[tara_label]
    )
    return {
        "natal_star": NAKSHATRAS[n_idx],
        "todays_star": NAKSHATRAS[t_idx],
        "star_count_from_janma": count,
        "tara_category": tara_label,
        "prediction": entry,
    }

NAKSHATRA_TRIADS: List[Tuple[str, List[str]]] = [
    ("Ketu", ["Ashwini", "Magha", "Mula"]),
    ("Venus", ["Bharani", "Purva Phalguni", "Purva Ashadha"]),
    ("Sun", ["Krittika", "Uttara Phalguni", "Uttara Ashadha"]),
    ("Moon", ["Rohini", "Hasta", "Shravana"]),
    ("Mars", ["Mrigashira", "Chitra", "Dhanishta"]),
    ("Rahu", ["Ardra", "Swati", "Shatabhisha"]),
    ("Jupiter", ["Punarvasu", "Vishakha", "Purva Bhadrapada"]),
    ("Saturn", ["Pushya", "Anuradha", "Uttara Bhadrapada"]),
    ("Mercury", ["Ashlesha", "Jyeshtha", "Revati"]),
]

DEFAULT_STAR_DATASET: Dict[int, Dict[str, Any]] = {
    1: {
        "tara_name": "Janma Tara",
        "tara_telugu": "జన్మతార",
        "lord": "Sun (సూర్యుడు)",
        "vahana": "Peacock (నెమలి)",
        "is_favorable": False,
        "outlook": "Unfavorable / Caution",
        "dosha_duration": "~3 hours",
        "effects": (
            "Mental distress, restlessness, headache, and difficulty with food or proper sleep."
        ),
        "remedy_or_note": (
            "Donating a pumpkin (గుమ్మడిపండు) brings some relief. Avoid for auspicious events in the 1st Navaka."
        ),
    },
    2: {
        "tara_name": "Sampat Tara",
        "tara_telugu": "సంపత్తార",
        "lord": "Mercury (బుధుడు)",
        "vahana": "Horse (గుఱ్ఱము)",
        "is_favorable": True,
        "outlook": "Favorable (Good)",
        "dosha_duration": "None",
        "effects": (
            "Gains of all kinds, good financial income, business/trade efforts can succeed, and sensible judgment across matters."
        ),
        "remedy_or_note": "Favorable Tara.",
    },
    3: {
        "tara_name": "Vipat Tara",
        "tara_telugu": "విపత్తార",
        "lord": "Rahu (రాహువు)",
        "vahana": "Goat (మేక)",
        "is_favorable": False,
        "outlook": "Unfavorable / Caution",
        "dosha_duration": "~1 hour",
        "effects": (
            "Baseless quarrels, associating with people beneath one's standing, and physical discomforts."
        ),
        "remedy_or_note": (
            "Do not donate jaggery (బెల్లం). Avoid for auspicious events in the 2nd Navaka."
        ),
    },
    4: {
        "tara_name": "Kshema Tara",
        "tara_telugu": "క్షేమతార",
        "lord": "Jupiter (గురుడు)",
        "vahana": "Elephant (ఏనుగు)",
        "is_favorable": True,
        "outlook": "Favorable (Good)",
        "dosha_duration": "None",
        "effects": (
            "Participating in sacred/religious activities, meeting new people, gaining social respect and hospitality, and steady income."
        ),
        "remedy_or_note": "Favorable Tara.",
    },
    5: {
        "tara_name": "Pratyak Tara",
        "tara_telugu": "ప్రత్యక్తార",
        "lord": "Ketu (కేతువు)",
        "vahana": "Crow (కాకి)",
        "is_favorable": False,
        "outlook": "Unfavorable / Caution",
        "dosha_duration": "~3 hours",
        "effects": (
            "Harsh remarks from others, plans not clicking, and unexpected obstacles in tasks."
        ),
        "remedy_or_note": (
            "Do not donate salt (ఉప్పు). Avoid for auspicious events in the 3rd Navaka."
        ),
    },
    6: {
        "tara_name": "Sadhana Tara",
        "tara_telugu": "సాధనతార",
        "lord": "Moon (చంద్రుడు)",
        "vahana": "Fox (నక్క)",
        "is_favorable": True,
        "outlook": "Favorable (Good)",
        "dosha_duration": "None",
        "effects": (
            "Meeting accomplished/eminent people, a calm and peaceful mind, and meeting new women."
        ),
        "remedy_or_note": "Favorable Tara.",
    },
    7: {
        "tara_name": "Naidhana Tara",
        "tara_telugu": "నైధనతార",
        "lord": "Shani / Saturn (శని)",
        "vahana": "Lion (సింహము)",
        "is_favorable": False,
        "outlook": "Unfavorable / Caution",
        "dosha_duration": "~3 hours",
        "effects": (
            "Getting pulled into unnecessary disputes, arguments, and counter-arguments."
        ),
        "remedy_or_note": (
            "Donating gold (సువర్ణదానము) brings some relief. Avoid across all Navakas for auspicious events."
        ),
    },
    8: {
        "tara_name": "Mitra Tara",
        "tara_telugu": "మిత్రతార",
        "lord": "Shukra / Venus (శుక్రుడు)",
        "vahana": "Garuda (గరుత్మంతుడు)",
        "is_favorable": True,
        "outlook": "Favorable (Good)",
        "dosha_duration": "None",
        "effects": (
            "Interest in music and literature, harmony with women/spouse, meeting new people, good income, and scholarly/scriptural discussions."
        ),
        "remedy_or_note": "Favorable Tara.",
    },
    9: {
        "tara_name": "Parama Maitra Tara",
        "tara_telugu": "పరమమైత్రతార",
        "lord": "Kuja / Mars (కుజుడు)",
        "vahana": "Swan (హంస)",
        "is_favorable": True,
        "outlook": "Favorable (Good)",
        "dosha_duration": "None",
        "effects": (
            "Possible unexpected hassles alongside enthusiasm for bold/adventurous actions, detachment, and low desire to dominate others."
        ),
        "remedy_or_note": "Favorable Tara.",
    },
}


def _triad_index_for_star(star_name: str) -> Optional[int]:
    cleaned = star_name.strip().lower()
    for idx, (_, stars) in enumerate(NAKSHATRA_TRIADS):
        if any(s.lower() == cleaned for s in stars):
            return idx
    return None


def evaluate_natal_star_day(
    natal_star: str,
    todays_star: str,
    custom_dataset: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    n_triad = _triad_index_for_star(natal_star)
    t_triad = _triad_index_for_star(todays_star)
    if n_triad is None or t_triad is None:
        all_stars = [s for _, group in NAKSHATRA_TRIADS for s in group]
        return {
            "error": f"Unknown star name. Expected one of: {', '.join(all_stars)}"
        }

    # Cell value 1..9 in the 9x9 Gochara Tara table
    tara_num = ((t_triad - n_triad) % 9) + 1
    dataset = custom_dataset if custom_dataset else DEFAULT_STAR_DATASET
    entry = dataset.get(tara_num) or dataset.get(str(tara_num)) or DEFAULT_STAR_DATASET[tara_num]

    return {
        "natal_star": natal_star.strip(),
        "natal_triad_index": n_triad + 1,
        "todays_star": todays_star.strip(),
        "todays_triad_index": t_triad + 1,
        "tara_number": tara_num,
        "prediction": entry,
    }


def build_commute_agent(
    groq_api_key: str,
    creds_dict: Optional[Dict[str, Any]] = None,
    model_name: str = "llama-3.3-70b-versatile",
):
    """Build a LangGraph agent with Calendar, Gmail, Commute Fares, Tourist Places, Natal Star, and Facts tools."""

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
        """Search the user's Gmail inbox/messages using a Gmail search query and return subject, sender, date, and snippet."""
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

    @tool
    def get_tourist_places_near_area(area: str, limit: int = 6) -> str:
        """Find nearby tourist attractions around a given neighborhood, landmark, or city area."""
        try:
            places = find_nearby_tourist_places(area, limit=limit)
            return json.dumps({"area": area, "places": places})
        except Exception as exc:
            return json.dumps({"area": area, "error": str(exc)})

    @tool
    def get_natal_star_outlook(natal_star: str, todays_star: str) -> str:
        """Evaluate how today looks based on the user's natal star (Nakshatra) and today's transit star."""
        return json.dumps(evaluate_natal_star_day(natal_star, todays_star))

    @tool
    def get_surprise_fact() -> str:
        """Select a single surprise fact from the configured facts dataset."""
        return json.dumps({"fact": pick_surprise_fact()})

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
        "When a user asks to check, find, or summarize emails or infer their area/residence from emails, "
        "call `search_gmail_messages` with a concise Gmail search query. "
        "When comparing ride prices between two places or to an office/meeting venue, "
        "call `compare_commute_fares` and present a clear comparison across Rapido, Ola, Uber, and GREEN SM "
        "in INR (₹), noting that fares are rough estimates rather than live app quotes. "
        "When looking for tourist spots near an area, call `get_tourist_places_near_area`. "
        "When asked for a surprise fact, call `get_surprise_fact` and present the returned fact directly."
    )

    return create_agent(
        model=llm,
        tools=[
            get_calendar_meetings,
            compare_commute_fares,
            search_gmail_messages,
            get_tourist_places_near_area,
            get_natal_star_outlook,
            get_surprise_fact,
        ],
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