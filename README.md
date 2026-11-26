# 🧭 Personal Travel & Work Assistant (`Personal_Agent_Achyut`)

An AI-powered **Personal Travel & Work Assistant** built with **Python, Streamlit, LangGraph, Groq, and Google OAuth 2.0**, designed specifically for corporate employees traveling on business trips or client-site project deployments.

---

## 🎯 Problem Statement

Corporate employees traveling to unfamiliar cities for business trips, client meetings, or onsite project deployments often lose productive hours switching across multiple tools to check meeting schedules, extract venue/stay addresses from emails, compare local ride fares, navigate a new city, and track reimbursable out-of-pocket expenses.

This assistant eliminates that context-switching by unifying **calendar-aware commute fare comparisons**, **email-based location discovery**, a **Trip Expense Ledger with HR reimbursement policy tallying**, and **personalized downtime/wellness touchpoints** inside a single conversational workspace.

---

## ✨ Key Features

1. **Commute to Next Meeting Place**
   - Connects securely to **Google Calendar** via OAuth 2.0 (with PKCE and signed state verification) to detect upcoming meetings and venues.
   - Calculates real driving distances and durations via **Nominatim** and **OSRM**.
   - Compares estimated fares in INR (`₹`) across **Rapido**, **Ola**, **Uber**, and **GREEN SM (EV)** covering Bike, Auto, Cab, and EV Cab options.
2. **Nearby Tourist Places (Downtime Discovery)**
   - Infers your stay or residential neighborhood from recent **Gmail** messages (or accepts a manual location) and recommends nearby attractions with driving distances and travel times.
3. **Trip Expense Ledger & HR Reimbursement Policy Tally**
   - Logs **Transport** (Bike/Auto, Cab/EV), **Meals & Per Diem**, **Client Hospitality**, and **Miscellaneous** charges via either the interactive UI panel or natural language chat.
   - Automatically evaluates every entry against configurable **HR daily category caps** and **receipt thresholds** (`> ₹250`), computing **Total Claimed**, **HR Eligible**, and **Out-of-Policy** amounts with one-click **CSV export**.
4. **Daily Outlook Based on Natal Star (*Tara Bala*)**
   - Evaluates daily transit favorability using the traditional 9-triad *Gochara Tara* system across all 27 *Nakshatras*, displaying the Tara name, ruling lord, *vahana*, dosha duration, effects, and remedies.
5. **Surprise Me With a Fact**
   - Surfaces curated historical and scientific discoveries from ancient India (covering mathematics, astronomy, surgery, metallurgy/botany, and meteorology) for quick mental resets.

---

## 🏗️ Tech Stack

- **Frontend / UI:** Streamlit (`app_streamlit.py`)
- **Agent Orchestration:** LangGraph (`create_agent`) + LangChain Tools (`agent_full.py`)
- **LLM Inference:** Groq (`ChatGroq` — default `llama-3.3-70b-versatile`)
- **Google Integrations:** Google Calendar API v3, Gmail API v1, Google OAuth 2.0 (`google-auth-oauthlib`)
- **Geolocation & Routing:** OpenStreetMap Nominatim API & OSRM Routing Engine

---

## 📂 Project Structure

```text
Personal_Agent_Achyut/
├── agent_full.py          # LangGraph agent, Google OAuth helpers, routing/fare tools, Tara Bala, & HR expense tally logic
├── app_streamlit.py       # Streamlit 4-corner hub UI, chat interface, & Trip Expense Ledger dashboard
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml       # Local secrets (Groq API key, Google OAuth Client ID/Secret, Redirect URI)
└── README.md              # Project overview and setup instructions
```

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository & Create a Virtual Environment
```bash
git clone <your-repo-url>
cd Personal_Agent_Achyut
python -m venv .venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets (`.streamlit/secrets.toml`)
Create a `.streamlit/secrets.toml` file in the project root with your credentials:
```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.3-70b-versatile"
GOOGLE_CLIENT_ID = "your_google_oauth_client_id"
GOOGLE_CLIENT_SECRET = "your_google_oauth_client_secret"
REDIRECT_URI = "http://localhost:8501"
```

> **Note:** Make sure `http://localhost:8501` (and your deployed Streamlit Cloud URL, if hosted) is added under **Authorized redirect URIs** in your Google Cloud Console OAuth 2.0 Client settings.

### 4. Run the Application
```bash
streamlit run app_streamlit.py
```

---

## 🧾 Default HR Reimbursement Policy Rules

The built-in HR policy engine enforces daily category caps and receipt requirements (all customizable directly in the **HR Policy Rules** tab):

| Category | Default Daily Cap (`₹`) | Receipt Rule |
| :--- | :---: | :--- |
| **Transport - Bike/Auto** | `₹500` | Mandatory above `₹250` |
| **Transport - Cab/EV** | `₹1,500` | Mandatory above `₹250` |
| **Meals & Per Diem** | `₹1,000` | Mandatory above `₹250` |
| **Client Hospitality** | `₹2,500` | Mandatory above `₹250` |
| **Miscellaneous** | `₹500` | Mandatory above `₹250` |

---

## 🔮 Planned Enhancements

- **One-Click Fare-to-Ledger Logging:** Send an estimated commute fare (e.g., GREEN SM or Uber) directly from the commute comparison card into the Trip Expense Ledger in a single click.
- **Live Panchangam Auto-Fetch:** Automatically populate today's transit *Nakshatra* based on the user's current city.