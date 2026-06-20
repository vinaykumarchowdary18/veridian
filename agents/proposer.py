"""
agents/proposer.py — Gemini 2.5 Flash as the Proposer.
Reframed for business intelligence: drafts a structured analytical brief
grounded in live evidence. Writes to survive adversarial critique.
"""
import json
import httpx
from core.config import Config
from core.models import EvidencePacket, Proposal
from core.logger import get_logger

log = get_logger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_SYSTEM = """You are the Analyst in Veridian, an adversarial business intelligence system.

Your job: write the BEST POSSIBLE analytical brief answering the user's business question.

Structure your response EXACTLY as follows:

## Executive Summary
2-3 sentences. Direct answer to the question. No hedging.

## Key Findings
- Finding 1 (cite evidence where available)
- Finding 2
- Finding 3
- Finding 4
- Finding 5 (at least 5 findings)

## Market / Context Analysis
2-3 paragraphs of substantive analysis. Ground every claim in the evidence provided.
Mention specific numbers, trends, or data points where available.

## Risks & Caveats
- Risk or caveat 1
- Risk or caveat 2
- Risk or caveat 3

## Recommended Actions
Concrete, specific next steps a decision-maker can act on immediately.
- Action 1
- Action 2
- Action 3

## Reasoning Trace
Brief internal logic: how you weighted the evidence, what you prioritised, what you left out and why.

---
Rules:
- Be direct and specific. Vague generalities will be rejected by critics.
- Cite evidence snippets by number [1], [2] etc. when available.
- Every claim must be defensible. Two independent AI critics will challenge you.
- Non-technical language. Write for a business decision-maker, not a researcher.
"""


async def _call_gemini(api_key: str, model: str, system: str, user: str) -> str:
    url = _ENDPOINT.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 3000},
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response shape: {e}\n{json.dumps(data)[:400]}")


class ProposerAgent:
    def __init__(self, config: Config):
        self._config = config

    async def propose(self, question: str, evidence: EvidencePacket) -> Proposal:
        log.info(f"[bold purple]Analyst (Proposer)[/bold purple] (Gemini {self._config.gemini_model}) drafting brief…")

        evidence_block = _format_evidence(evidence)
        user_prompt = (
            f"BUSINESS QUESTION:\n{question}\n\n"
            f"{evidence_block}\n\n"
            f"Write your analytical brief now. Follow the structure exactly."
        )

        text = await _call_gemini(
            api_key=self._config.gemini_api_key,
            model=self._config.gemini_model,
            system=_SYSTEM,
            user=user_prompt,
        )

        # Split off reasoning trace
        reasoning = None
        if "## Reasoning Trace" in text:
            parts = text.split("## Reasoning Trace", 1)
            answer_body = parts[0].strip()
            reasoning = parts[1].strip()
        else:
            answer_body = text.strip()

        log.info("Analyst draft complete.")
        return Proposal(
            content=answer_body,
            model=self._config.gemini_model,
            reasoning=reasoning,
        )


def _format_evidence(evidence: EvidencePacket) -> str:
    if not evidence.snippets and not evidence.raw_answer:
        return "[No live evidence retrieved — answering from model knowledge only.]"
    lines = ["--- Live Evidence (Tavily Web Search) ---"]
    if evidence.raw_answer:
        lines.append(f"Direct answer from search: {evidence.raw_answer}")
    for i, (s, u) in enumerate(zip(evidence.snippets, evidence.urls), 1):
        lines.append(f"[{i}] Source: {u}\n{s[:400]}")
    return "\n\n".join(lines)
