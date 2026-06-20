"""
agents/critic_a.py — Critic A: Llama 3.3-70b via Groq.
Business intelligence focus: factual accuracy, data integrity, logical consistency.
"""
import json
from core.config import Config
from core.models import Proposal, EvidencePacket, CriticVerdict, CritiquePoint
from core.logger import get_logger
from agents._openai_compat import openai_compat_call

log = get_logger(__name__)

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM = """You are Critic A in Veridian, an adversarial business intelligence validation system.

Your role: DATA INTEGRITY & LOGIC AUDITOR.

Challenge the analytical brief on these dimensions:
1. Factual accuracy — are claims supported by the evidence provided? Flag unsupported assertions.
2. Data integrity — are numbers, percentages, and statistics used correctly and in context?
3. Logical consistency — do conclusions actually follow from the evidence? Spot leaps in reasoning.
4. Overgeneralisation — does the brief make sweeping claims from limited data?
5. Recency — is the analysis based on current data, or could it be outdated?

Be adversarial but fair. Your job is to make the brief better, not to reject everything.

Respond ONLY with valid JSON matching this schema exactly:
{
  "verdict": "accept" | "revise" | "reject",
  "score": <float 0.0-1.0>,
  "summary": "<one sentence overall assessment>",
  "critique_points": [
    {
      "severity": "critical" | "major" | "minor",
      "category": "factual" | "logical" | "completeness" | "clarity" | "business_impact",
      "description": "<specific problem with the brief>",
      "suggested_fix": "<concrete fix>"
    }
  ]
}
No markdown. No preamble. Raw JSON only."""


class CriticAgentA:
    CRITIC_ID = "critic_a"

    def __init__(self, config: Config):
        self._config = config

    async def critique(self, proposal: Proposal, evidence: EvidencePacket) -> CriticVerdict:
        log.info(f"[bold teal]Critic A — Data Auditor[/bold teal] (Groq/{self._config.groq_model}) evaluating…")

        evidence_block = _format_evidence(evidence)
        user_prompt = (
            f"ORIGINAL EVIDENCE THE BRIEF WAS BASED ON:\n{evidence_block}\n\n"
            f"ANALYTICAL BRIEF TO AUDIT:\n{proposal.content}\n\n"
            f"Apply your data integrity audit now. Respond with JSON only."
        )

        raw = await openai_compat_call(
            endpoint=_ENDPOINT,
            api_key=self._config.groq_api_key,
            model=self._config.groq_model,
            system=_SYSTEM,
            user=user_prompt,
            temperature=0.1,
            max_tokens=1500,
        )

        data = _parse_json(raw)
        points = [CritiquePoint(**p) for p in data.get("critique_points", [])]

        verdict = CriticVerdict(
            critic_id=self.CRITIC_ID,
            model=self._config.groq_model,
            verdict=data.get("verdict", "revise"),
            score=float(data.get("score", 0.5)),
            critique_points=points,
            summary=data.get("summary", ""),
        )
        log.info(f"Critic A verdict: {verdict.verdict} (score={verdict.score:.2f}, {len(points)} points)")
        return verdict


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Critic A returned invalid JSON: {e}\nRaw: {raw[:300]}")


def _format_evidence(evidence: EvidencePacket) -> str:
    if not evidence.snippets:
        return "[No external evidence was provided to the analyst]"
    return "\n".join(f"[{i}] {s[:300]}" for i, s in enumerate(evidence.snippets, 1))
