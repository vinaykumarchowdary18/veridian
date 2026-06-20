# Veridian — Adversarial Business Intelligence

> Ask any business question. Four AI models debate the answer before you see it.

---

## What it does

Most AI tools give you one answer from one model. You have no idea if it's accurate, complete, or missing critical risks.

Veridian runs every question through an **adversarial multi-agent debate** before returning anything:

```
Your Question
      │
      ▼
[Tavily]  ←── Live web evidence fetched first
      │
      ▼
[Analyst — Gemini 2.5 Flash]
  Drafts a structured intelligence brief
      │
      ├──────────────────────────────┐
      ▼                              ▼
[Auditor A — Llama 3.3-70b     [Auditor B — DeepSeek v3
   via Groq]                      via OpenRouter]
 Data & Logic Audit               Strategy & Completeness Audit
 (independent, parallel)          (independent, parallel)
      │                              │
      └──────────────┬───────────────┘
                     ▼
          [Arbiter — GPT-4o-mini
            via GitHub Models]
         Reconciles verdicts, scores consensus
         Decides: finalize | revise | escalate
                     │
           (loops up to MAX_DEBATE_ROUNDS)
                     │
                     ▼
         ✅ Validated Intelligence Brief
            + Confidence score
            + Audit trail
            + Dissenting points logged
```

Each model is a **different family** (Google / Meta / DeepSeek / OpenAI) running on **different infrastructure** — no shared biases, no echo chamber.

---

## Output structure

Every question produces a structured brief with:

| Section | Content |
|---|---|
| **Executive Summary** | 2-3 sentence direct answer |
| **Key Findings** | 5+ validated findings with evidence citations |
| **Market / Context Analysis** | Substantive analysis grounded in live data |
| **Risks & Caveats** | What the brief does NOT guarantee |
| **Recommended Actions** | Concrete next steps for a decision-maker |
| **Confidence Score** | How much the auditors agreed |
| **Audit Trail** | Every unresolved dissent, logged transparently |
| **Sources** | Live web sources from Tavily |

---

## Example questions

```
What are the fastest growing B2B SaaS markets in 2025?
Compare cloud cost optimization strategies for mid-size companies
What skills are most in demand for data engineering roles right now?
Is generative AI adoption slowing down in enterprise?
What are the biggest risks of adopting microservices for a growing startup?
Which industries are seeing the highest ROI from AI automation?
What does the job market look like for software engineers in 2025?
What are the biggest challenges in MLOps adoption at scale?
```

---

## Confidence levels

| Consensus Score | Meaning |
|---|---|
| ≥ 0.75 | 🟢 High — auditors broadly agreed, safe to act on |
| 0.50–0.74 | 🟡 Medium — some disagreement, verify key claims |
| < 0.50 | 🔴 Low — significant unresolved critique, check sources manually |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/vinaykumarchowdary18/veridian
cd veridian
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Open .env and fill in your 5 keys
```

| Key | Provider | Free tier? |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio | ✅ Yes |
| `GROQ_API_KEY` | Groq Cloud | ✅ Yes |
| `OPENROUTER_API_KEY` | OpenRouter | ✅ Yes (DeepSeek free) |
| `GITHUB_TOKEN` | GitHub (Models: read scope) | ✅ Yes |
| `TAVILY_API_KEY` | Tavily Search | ✅ Yes (1000/month) |

> **GitHub token:** Go to GitHub → Settings → Developer settings → Fine-grained personal access tokens → Create token with **Models: read** permission.

### 4. Run the Gradio UI

```bash
python app.py
```

Opens at `http://localhost:7860`

---

## File structure

```
veridian/
├── app.py                    ← Gradio UI (run this)
│
├── core/
│   ├── config.py             ← Loads & validates all .env keys
│   ├── models.py             ← Pydantic data models
│   ├── logger.py             ← Rich logging
│   └── orchestrator.py       ← Debate loop engine
│
├── agents/
│   ├── _openai_compat.py     ← Shared async HTTP caller
│   ├── proposer.py           ← Gemini 2.5 Flash (Analyst)
│   ├── critic_a.py           ← Llama 3.3-70b via Groq (Data Auditor)
│   ├── critic_b.py           ← DeepSeek v3 via OpenRouter (Strategy Auditor)
│   └── arbiter.py            ← GPT-4o-mini via GitHub Models (Arbiter)
│
├── tools/
│   └── evidence.py           ← Tavily web search wrapper
│
├── outputs/                  ← Auto-created: JSON + Markdown reports per query
│
├── .env.example              ← Key template
├── requirements.txt
└── README.md
```

---

## Tuning

In your `.env`:

```env
MAX_DEBATE_ROUNDS=2        # raise to 3 for higher-stakes questions (slower)
MIN_CONSENSUS_SCORE=0.72   # raise to 0.85 for max accuracy, lower to 0.65 for speed
```

---

## Architecture notes

- **No shared state between auditors** — Auditor B has not seen Auditor A's output. Enforced by independent parallel async calls.
- **Model diversity by design** — Gemini (Google), Llama (Meta), DeepSeek (Chinese open-weight), GPT (OpenAI). Four different training paradigms, four different failure modes.
- **Evidence-first** — Tavily fetches live web context before any model sees the question, grounding all four agents in current facts.
- **Transparent uncertainty** — Dissenting minority critique points are always logged, even when overruled.
- **Saves everything** — Every query saved to `./outputs/` as JSON (machine-readable) and Markdown (human-readable).

---

## Built on

AMAV (Adversarial Multi-Agent Validation) architecture — originally built for academic research validation, adapted here for business intelligence.

GitHub: [vinaykumarchowdary18/veridian](https://github.com/vinaykumarchowdary18/veridian)
