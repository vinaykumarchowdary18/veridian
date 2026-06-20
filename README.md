# Veridian — Adversarial Business Intelligence Agent

> Ask any business question. Four AI models debate the answer before you see it.

**Kaggle AI Agents Intensive Vibe Coding Capstone 2026 — Track: Agents for Business**

🎥 [Demo Video](https://youtu.be/aSBQtWnbwJo) | 💻 [GitHub](https://github.com/vinaykumarchowdary18/veridian)

---

## The Problem

Business decision-makers ask AI tools questions every day — about markets, competitors, strategies, risks. The problem is that every major AI tool gives you **one answer from one model**, with no way to know if it is accurate, complete, or missing critical caveats.

There is no cross-validation. No audit trail. No transparency about what the model is uncertain about. You either trust it blindly or you do not use it at all.

This is especially damaging for high-stakes business decisions where a wrong answer has real consequences — budget allocation, technology adoption, market entry, hiring strategy.

---

## The Solution

Veridian solves this by running every question through an **adversarial multi-agent debate** before returning anything to the user.

Instead of one model answering, four models with different architectures and training paradigms work against each other:

- An **Analyst** drafts the best possible answer grounded in live web evidence
- Two independent **Auditors** challenge it in parallel — one focused on data integrity, one on strategic completeness
- An **Arbiter** reconciles both verdicts, scores consensus, and either finalizes the answer, requests a revision, or escalates for a complete redraft

The user only sees the output after this debate has completed. Every dissenting point is logged in a transparent audit trail so the user knows exactly what the system disagreed on.

---

## Architecture

```
Your Question
      │
      ▼
[Tavily Web Search]
  Live evidence fetched before any model sees the question
      │
      ▼
[Analyst — Gemini 2.5 Flash]
  Drafts structured intelligence brief:
  Executive Summary / Key Findings / Risks / Recommended Actions
      │
      ├─────────────────────────────────┐
      ▼                                 ▼
[Auditor A — Llama 3.3-70b]    [Auditor B — Llama 3.1-8b-instant]
  via Groq                        via Groq
  Focus: Data integrity           Focus: Strategy & completeness
  & logical consistency           & actionability
  (runs independently)            (runs independently, in parallel)
      │                                 │
      └──────────────┬──────────────────┘
                     ▼
          [Arbiter — GPT-4o-mini]
            via GitHub Models
            Scores consensus = avg(auditor scores)
            Directive: finalize | revise | escalate
                     │
         ┌───────────┴────────────┐
         │                        │
    finalize                   revise/escalate
         │                        │
         ▼                   loop back (max 2 rounds)
  ✅ Intelligence Brief
     + Confidence score
     + Audit trail
     + Sources
```

### Why four different models?

Each model comes from a different provider and training paradigm:

| Role | Model | Provider | Focus |
|---|---|---|---|
| Analyst | Gemini 2.5 Flash | Google | Draft generation |
| Auditor A | Llama 3.3-70b-versatile | Meta via Groq | Data & logic audit |
| Auditor B | Llama 3.1-8b-instant | Meta via Groq | Strategy & completeness audit |
| Arbiter | GPT-4o-mini | OpenAI via GitHub Models | Consensus scoring |
| Evidence | Tavily Search | Tavily | Live web retrieval |

No two agents share infrastructure. This prevents correlated failures and echo chamber effects.

### Agent concepts demonstrated

| Kaggle Concept | Implementation |
|---|---|
| Multi-agent system | 4-agent adversarial pipeline with orchestrated debate loop |
| Agent skills | Evidence retrieval tool, structured output parsing, iterative refinement |
| MCP Server | `mcp_server.py` — exposes Veridian as MCP tool for Claude Desktop |
| Security features | Rate limiting, prompt injection detection, input sanitisation (`core/security.py`) |
| Deployability | FastAPI server + static frontend, runs locally or any cloud host |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- 5 free API keys (all have free tiers, no credit card required)

### Step 1 — Clone the repo

```bash
git clone https://github.com/vinaykumarchowdary18/veridian
cd veridian
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Get your API keys

| Key | Where to get it | Free? |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | ✅ Yes |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | ✅ Yes |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | ✅ Yes |
| `GITHUB_TOKEN` | GitHub → Settings → Developer settings → Fine-grained tokens → scope: **Models: read** | ✅ Yes |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | ✅ Yes (1000/month) |

### Step 4 — Configure environment

```bash
cp .env.example .env
# Open .env and paste your 5 keys
```

### Step 5 — Run

```bash
uvicorn server:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## File Structure

```
veridian/
├── server.py                 ← FastAPI server — main entry point
├── app.py                    ← Gradio fallback UI
├── mcp_server.py             ← MCP server for Claude Desktop
├── mcp_config.json           ← Claude Desktop MCP configuration
│
├── core/
│   ├── config.py             ← Environment variable loading & validation
│   ├── models.py             ← Pydantic data models (IntelBrief, etc.)
│   ├── logger.py             ← Rich-powered structured logging
│   ├── orchestrator.py       ← Debate loop engine
│   └── security.py           ← Rate limiting, injection detection, sanitisation
│
├── agents/
│   ├── _openai_compat.py     ← Shared async HTTP caller for OpenAI-compatible APIs
│   ├── proposer.py           ← Gemini 2.5 Flash — Analyst agent
│   ├── critic_a.py           ← Llama 3.3-70b — Data Auditor agent
│   ├── critic_b.py           ← Llama 3.1-8b-instant — Strategy Auditor agent
│   └── arbiter.py            ← GPT-4o-mini — Arbiter agent
│
├── tools/
│   └── evidence.py           ← Tavily web search tool
│
├── static/
│   └── index.html            ← Custom dark UI frontend
│
├── outputs/                  ← Auto-created: JSON + Markdown per query
├── .env.example              ← API key template
└── requirements.txt
```

---

## Security Features

`core/security.py` implements a 5-layer security pipeline on every request:

1. **Rate limiting** — sliding window, 10 requests per 60 seconds per IP
2. **Input sanitisation** — strips null bytes and control characters
3. **Prompt injection detection** — 15 regex patterns blocking instruction overrides, role hijacking, and data exfiltration attempts
4. **Content filtering** — blocks harmful or off-topic requests
5. **API key validation** — checks all 5 keys at startup for presence and placeholder values

---

## MCP Server

Veridian exposes itself as an MCP-compatible tool server via `mcp_server.py`:

**Tools available:**
- `run_intelligence_brief(question)` — runs the full 4-agent debate and returns a structured brief
- `get_confidence_explanation(confidence, consensus_score, debate_rounds)` — interprets what the scores mean

**Connect to Claude Desktop:**

```json
{
  "mcpServers": {
    "veridian": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/veridian"
    }
  }
}
```

---

## Output Format

Every query produces:

- **UI output** — structured brief with tabbed sections (Executive Summary, Key Findings, Risks, Actions, Sources, Audit Trail)
- **`outputs/<timestamp>.md`** — human-readable Markdown report
- **`outputs/<timestamp>.json`** — full machine-readable record including all agent verdicts

### Confidence levels

| Score | Meaning |
|---|---|
| ≥ 0.75 | 🟢 High — auditors broadly agreed, safe to act on |
| 0.50–0.74 | 🟡 Medium — some disagreement, verify key claims |
| < 0.50 | 🔴 Low — significant dissent, check sources manually |

---

## Built on AMAV

Veridian is built on the AMAV (Adversarial Multi-Agent Validation) architecture, originally developed for academic research validation and graduate school application assistance. Veridian adapts this architecture for business intelligence use cases.

Original AMAV repo: [github.com/vinaykumarchowdary18/-Adversarial-Multi-Agent-Validation](https://github.com/vinaykumarchowdary18/-Adversarial-Multi-Agent-Validation)
