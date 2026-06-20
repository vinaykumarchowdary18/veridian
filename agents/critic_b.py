"""
agents/critic_b.py — Critic B: DeepSeek v3 via OpenRouter.
Business intelligence focus: strategic completeness, alternative perspectives, actionability.
Intentionally uses a different model family than Critic A for genuine diversity.
"""
import json
from core.config import Config
from core.models import Proposal, EvidencePacket, CriticVerdict, CritiquePoint
from core.logger import get_logger
from agents._openai_compat import openai_compat_call

log = get_logger(__name__)

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM = """You are Critic B in Veridian, an adversarial business intelligence validation system.

Your role: STRATEGY & COMPLETENESS AUDITOR.

You have NOT seen Critic A's evaluation. Be fully independent.

Challenge the analytical brief on these dimensions:
1. Strategic completeness — what important angles, competitors, or scenarios are missing?
2. Actionability — are the recommended actions concrete and realistic, or just generic advice?
3. Alternative perspectives — what would a skeptic or contrarian expert say about this analysis?
4. Audience fit — is this actually useful for a business decision-maker, or too academic/vague?
5. Missing risks — what downside scenarios or edge cases were not considered?

Be adversarial but constructive. Push for a better brief, not a perfect one.

Respond ONLY with valid JSON matching this schema exactly:
{
  "verdict": "accept" | "revise" | "reject",
  "score": <float 0.0-1.0>,
  "summary": "<one sentence overall assessment>",
  "critique_points": [
    {
      "severity": "critical" | "major" | "minor",
      "category": "factual" | "logical" | "completeness" | "clarity" | "business_impact",
      "description": "<specific gap or weakness in the brief>",
      "suggested_fix": "<concrete improvement>"
    }
  ]
}
No markdown. No preamble. Raw JSON only."""


class CriticAgentB:
    CRITIC_ID = "critic_b"

    def __init__(self, config: Config):
        self._config = config

    async def critique(self, proposal: Proposal, evidence: EvidencePacket) -> CriticVerdict:
        log.info(f"[bold coral]Critic B — Strategy Auditor[/bold coral] (Groq/{self._config.groq_model_b}) evaluating…")

        evidence_block = _format_evidence(evidence)
        user_prompt = (
            f"ORIGINAL EVIDENCE THE BRIEF WAS BASED ON:\n{evidence_block}\n\n"
            f"ANALYTICAL BRIEF TO AUDIT:\n{proposal.content}\n\n"
            f"Apply your strategy audit now. Respond with JSON only."
        )

        last_error = None
        for attempt in range(1, 4):
            try:
                raw = await openai_compat_call(
                    endpoint=_ENDPOINT,
                    api_key=self._config.groq_api_key,
                    model=self._config.groq_model_b,
                    system=_SYSTEM,
                    user=user_prompt,
                    temperature=0.1,
                    max_tokens=1500,
                )
                data = _parse_json(raw)
                points = [CritiquePoint(**p) for p in data.get("critique_points", [])]
                verdict = CriticVerdict(
                    critic_id=self.CRITIC_ID,
                    model=self._config.openrouter_model,
                    verdict=data.get("verdict", "revise"),
                    score=float(data.get("score", 0.5)),
                    critique_points=points,
                    summary=data.get("summary", ""),
                )
                log.info(f"Critic B verdict: {verdict.verdict} (score={verdict.score:.2f}, {len(points)} points)")
                return verdict

            except Exception as e:
                last_error = e
                log.warning(
                    f"Critic B attempt {attempt}/3 failed: {e}. "
                    f"{'Retrying…' if attempt < 3 else 'Giving up.'}"
                )

        log.warning("Critic B failed all 3 attempts. Using neutral fallback verdict.")
        return CriticVerdict(
            critic_id=self.CRITIC_ID,
            model=self._config.groq_model_b,
            verdict="revise",
            score=0.5,
            critique_points=[],
            summary=f"Critic B unavailable after 3 attempts ({last_error}). Treat as neutral.",
        )


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("Critic B returned empty response.")
    if cleaned.startswith("##") or all(c in "* \n\t\\()." for c in cleaned):
        raise ValueError(f"Critic B returned non-JSON garbage. Raw: {cleaned[:100]}")
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Critic B returned invalid JSON: {e}\nRaw: {raw[:300]}")


def _format_evidence(evidence: EvidencePacket) -> str:
    if not evidence.snippets:
        return "[No external evidence was provided to the analyst]"
    return "\n".join(f"[{i}] {s[:300]}" for i, s in enumerate(evidence.snippets, 1))
