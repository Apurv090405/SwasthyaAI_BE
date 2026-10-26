# 🧠 SwasthyaAI — Python Backend

AI-powered clinical health assistant backend built with **FastAPI**, **LangGraph**, and **LangChain**. It orchestrates multiple specialized clinical agents via a LangGraph Supervisor to analyze patient conversations, generate clinical reports, and power voice-based health interactions.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Server                      │
│                   (app.py)                           │
├─────────────┬───────────────┬───────────────────────┤
│  /analyze   │  /agent_chat  │  /voice & /telnyx     │
│  transcript │  single-agent │  WebSocket voice      │
└──────┬──────┴───────┬───────┴───────────┬───────────┘
       │              │                   │
       ▼              ▼                   ▼
 ┌──────────┐  ┌────────────┐     ┌──────────────┐
 │ LangGraph│  │ Individual │     │ Pipecat/     │
 │Supervisor│  │  Agent     │     │ Sarvam TTS   │
 │ Pipeline │  │  Chat      │     │ Voice Agent  │
 └────┬─────┘  └────────────┘     └──────────────┘
      │
      ▼
 ┌──────────────────────────────────────────┐
 │          Clinical Agent Team             │
 │  • diagnosis_agent                       │
 │  • rag_agent (LanceDB + embeddings)      │
 │  • google_search_agent (SerpAPI)         │
 │  • clinical_rules_agent                  │
 │  • risk_scoring_agent                    │
 │  • differential_diagnosis_agent          │
 │  • triage_decision_agent                 │
 │  • recommendation_agent                  │
 │  • report_formatting_agent              │
 │  • eval_agent                            │
 └──────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| AI Orchestration | LangGraph Supervisor |
| LLM | OpenAI GPT-4.1 / Google Gemini 2.5 Flash |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | LanceDB |
| Search | SerpAPI |
| Voice | Pipecat + Sarvam AI |
| Telephony | Telnyx / Exotel / Vobiz |
| Database | Supabase (PostgreSQL) |
| Package Manager | uv |

## Prerequisites

- **Python** ≥ 3.11
- **uv** ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- API keys (see `.env.sample`)

## Quick Start

```bash
# 1. Clone and enter the directory
cd swasthya-ai-python

# 2. Copy and fill environment variables
cp .env.sample .env
# Edit .env with your API keys

# 3. Install dependencies
uv sync

# 4. Run the server
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000** with docs at **http://localhost:8000/docs**.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + Supabase status |
| `GET` | `/clinical_agents` | List available clinical agents |
| `POST` | `/agent_chat` | Chat with a single clinical agent |
| `POST` | `/analyze_transcript` | Full clinical supervisor analysis |
| `POST` | `/voice/start` | Start voice call session |
| `WS` | `/ws/voice/{call_id}` | Voice WebSocket stream |
| `POST` | `/telnyx/answer` | Telnyx call webhook |
| `WS` | `/ws/telnyx/{call_id}` | Telnyx media WebSocket |
| `GET` | `/dashboard/stats` | Dashboard KPI stats |
| `GET` | `/dashboard/calls` | Paginated call history |

## Project Structure

```
swasthya-ai-python/
├── app.py                  # FastAPI entry point & route registration
├── agents.py               # LangGraph agent definitions
├── config.py               # Settings & environment variables
├── llm_factory.py          # LLM provider factory (OpenAI/Gemini)
├── models.py               # Data models
├── tools.py                # Agent tools (search, RAG, etc.)
├── webhook.py              # Webhook handlers
├── api/
│   ├── dashboard_routes.py # Dashboard API endpoints
│   ├── telnyx_routes.py    # Telnyx telephony integration
│   ├── voice_routes.py     # Voice call endpoints
│   └── web_call_routes.py  # Web-based call endpoints
├── services/
│   ├── post_call_pipeline.py    # Post-call clinical analysis
│   ├── supabase_dashboard.py    # Dashboard data operations
│   └── supabase_voice.py       # Voice call DB operations
├── prompts/                # Agent system prompts
├── supabase/
│   └── migrations/         # SQL migration files
├── lancedb_store/          # Local vector DB storage
├── pyproject.toml          # Project dependencies (uv)
└── uv.lock                 # Locked dependencies
```

## Database Setup

Run the SQL migrations in your Supabase SQL Editor in order:

1. `SwathyaAI-FE-main/supabase/migrations/001_profiles_and_auth.sql`
2. `SwathyaAI-FE-main/supabase/migrations/002_voice_calls.sql`
3. `swasthya-ai-python/supabase/migrations/003_dashboard_tables.sql`

## Environment Variables

See [`.env.sample`](.env.sample) for all required and optional environment variables.

**Required:**
- `OPENAI_API_KEY` or `GEMINI_API_KEY` — at least one LLM key
- `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` — for data persistence

**Optional:**
- `SERPAPI_KEY` — for Google search agent
- `SARVAM_API_KEY` — for voice (TTS/STT)
- `TELNYX_API_KEY` — for telephony
- `DEEPGRAM_API_KEY` — for speech-to-text

## License

Private — All rights reserved.
