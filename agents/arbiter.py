"""
agents/arbiter.py — Arbiter: GPT-4o-mini via GitHub Models.
Reconciles both critic verdicts, scores consensus, issues final directive.
"""
import json
from core.config import Config
from core.models import Proposal, EvidencePacket, CriticVerdict, ArbiterDecision
from core.logger import get_logger
from agents._openai_compat import openai_compat_call

log = get_logger(__name__)

_ENDPOINT = "https://models.github.ai/inference/chat/completions"

_SYSTEM = """You are the Arbiter in Veridian, an adversarial business intelligence validation system.

You receive an analytical brief and two fully independent auditor evaluations.

Your job:
  1. Calculate consensus_score using the EXACT formula below.
  2. Identify which critique points are valid (accept) vs nitpicking / wrong (reject).
  3. Issue one of three directives.

CONSENSUS SCORE FORMULA — follow this exactly:
  Step 1: avg = (critic_a_score + critic_b_score) / 2
  Step 2: if abs(critic_a_score - critic_b_score) > 0.3, subtract 0.05 from avg
  Step 3: if BOTH verdicts are "reject", cap score at 0.45
  Step 4: if BOTH verdicts are "accept", floor score at 0.65
  Step 5: consensus_score = final value from above steps

CRITICAL: "revise" is NOT "reject". A "revise" verdict with score 0.70 means 70% quality.
Two "revise" verdicts both at 0.70 = avg of 0.70 = consensus_score of 0.70. Do NOT output 0.40 in this case.
Only cap at 0.45 when BOTH verdicts are literally the string "reject".

DIRECTIVE RULES:
  consensus_score >= MIN_CONSENSUS_SCORE  →  "finalize"
  0.5 <= consensus_score < MIN_CONSENSUS_SCORE  →  "revise" (provide full revised_proposal)
  consensus_score < 0.5  →  "escalate"

When revising, produce a COMPLETE improved brief that:
- Fixes all accepted critique points
- Maintains structure: Executive Summary, Key Findings, Market Analysis, Risks, Recommended Actions
- Is ready for a business decision-maker to act on immediately

Respond ONLY with valid JSON matching this schema exactly:
{
  "consensus_score": <float 0.0-1.0>,
  "accepted_points": ["<point>", ...],
  "rejected_points": ["<point>", ...],
  "directive": "finalize" | "revise" | "escalate",
  "revised_proposal": "<full improved brief, or null if directive is not revise>",
  "reasoning": "<2-3 sentences explaining your ruling and the score calculation>"
}
No markdown. No preamble. Raw JSON only."""


class ArbiterAgent:
    ARBITER_ID = "arbiter"

    def __init__(self, config: Config):
        self._config = config
        self._min_score = config.min_consensus_score

    async def arbitrate(
        self,
        proposal: Proposal,
        verdict_a: CriticVerdict,
        verdict_b: CriticVerdict,
        evidence: EvidencePacket,
        round_number: int = 1,
    ) -> ArbiterDecision:
        log.info(
            f"[bold orange]Arbiter[/bold orange] (GitHub/{self._config.github_model}) "
            f"ruling on round {round_number}…"
        )

        user_prompt = _build_prompt(proposal, verdict_a, verdict_b, evidence, self._min_score)

        raw = await openai_compat_call(
            endpoint=_ENDPOINT,
            api_key=self._config.github_token,
            model=self._config.github_model,
            system=_SYSTEM,
            user=user_prompt,
            temperature=0.1,
            max_tokens=2500,
            extra_headers={
                "X-GitHub-Api-Version": "2026-03-10",
                "Accept": "application/vnd.github+json",
            },
        )

        data = _parse_json(raw)

        # Safety: if arbiter still returns 0.4 despite revise verdicts, correct it
        score = float(data.get("consensus_score", 0.5))
        va_score = verdict_a.score
        vb_score = verdict_b.score
        va_verdict = verdict_a.verdict
        vb_verdict = verdict_b.verdict

        # Override arbiter math if it's clearly wrong
        if va_verdict != "reject" and vb_verdict != "reject" and score <= 0.42:
            corrected = (va_score + vb_score) / 2
            log.warning(
                f"Arbiter returned suspicious score {score:.2f} despite non-reject verdicts. "
                f"Correcting to {corrected:.2f} based on auditor scores."
            )
            score = corrected

        # Sanitise revised_proposal — must be str or None, never a dict/list
        raw_rp = data.get("revised_proposal")
        if isinstance(raw_rp, str) and raw_rp.strip():
            revised_proposal = raw_rp.strip()
        else:
            revised_proposal = None

        directive = data.get("directive", "revise")

        # Override directive based on corrected score
        if score >= self._min_score:
            directive = "finalize"
            revised_proposal = None
            log.info(f"Score {score:.2f} >= min {self._min_score} — finalizing.")

        decision = ArbiterDecision(
            consensus_score=score,
            accepted_points=data.get("accepted_points", []),
            rejected_points=data.get("rejected_points", []),
            directive=directive,
            revised_proposal=revised_proposal,
            reasoning=data.get("reasoning", ""),
        )

        log.info(
            f"Arbiter ruling: {decision.directive} "
            f"(consensus={decision.consensus_score:.2f}, "
            f"accepted={len(decision.accepted_points)}, "
            f"rejected={len(decision.rejected_points)})"
        )
        return decision


def _build_prompt(
    proposal: Proposal,
    verdict_a: CriticVerdict,
    verdict_b: CriticVerdict,
    evidence: EvidencePacket,
    min_score: float,
) -> str:
    def fmt_verdict(v: CriticVerdict) -> str:
        points = "\n".join(
            f"  [{p.severity.upper()}] ({p.category}) {p.description}"
            + (f"\n    Fix: {p.suggested_fix}" if p.suggested_fix else "")
            for p in v.critique_points
        )
        return (
            f"Auditor ID: {v.critic_id}  Model: {v.model}\n"
            f"Verdict: {v.verdict}  Score: {v.score:.2f}\n"
            f"Summary: {v.summary}\n"
            f"Points:\n{points if points else '  (none)'}"
        )

    evidence_block = (
        "[No external evidence available]"
        if not evidence.snippets
        else "\n".join(f"[{i}] {s[:200]}" for i, s in enumerate(evidence.snippets, 1))
    )

    expected_avg = (verdict_a.score + verdict_b.score) / 2

    return (
        f"MIN_CONSENSUS_SCORE: {min_score}\n"
        f"EXPECTED CONSENSUS SCORE (pre-calculated for you): {expected_avg:.2f} "
        f"(= ({verdict_a.score:.2f} + {verdict_b.score:.2f}) / 2)\n\n"
        f"=== ANALYTICAL BRIEF (by {proposal.model}) ===\n{proposal.content}\n\n"
        f"=== AUDITOR A EVALUATION ===\n{fmt_verdict(verdict_a)}\n\n"
        f"=== AUDITOR B EVALUATION ===\n{fmt_verdict(verdict_b)}\n\n"
        f"=== EVIDENCE ===\n{evidence_block}\n\n"
        f"Issue your ruling now. Remember: use the pre-calculated score above as your starting point. "
        f"Respond with JSON only."
    )


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Arbiter returned invalid JSON: {e}\nRaw: {raw[:300]}")
