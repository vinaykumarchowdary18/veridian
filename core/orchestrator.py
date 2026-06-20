"""
core/orchestrator.py — the Veridian debate loop engine.
Wires: Evidence → Analyst → [Auditor A ‖ Auditor B] → Arbiter → repeat or finalize.
Returns a fully validated IntelBrief.
"""
import asyncio
import re
from datetime import datetime, timezone

from core.config import Config
from core.models import (
    EvidencePacket, Proposal, CriticVerdict,
    DebateRound, ArbiterDecision, IntelBrief,
)
from core.logger import get_logger
from tools.evidence import EvidenceTool
from agents.proposer import ProposerAgent
from agents.critic_a import CriticAgentA
from agents.critic_b import CriticAgentB
from agents.arbiter import ArbiterAgent

log = get_logger(__name__)


class Orchestrator:
    """
    Runs the full Veridian debate pipeline for a single business question.

    Pipeline per round:
      1. Analyst drafts a structured intelligence brief (Gemini).
      2. Auditor A and Auditor B critique in parallel (Groq + OpenRouter).
      3. Arbiter reconciles, scores, decides (GitHub Models / GPT).
      4. If directive == "revise", use revised brief and loop.
      5. If directive == "escalate", re-draft with stronger grounding.
      6. If directive == "finalize" OR max rounds reached → emit IntelBrief.
    """

    def __init__(self, config: Config):
        self._cfg = config
        self._evidence_tool = EvidenceTool(api_key=config.tavily_api_key)
        self._proposer = ProposerAgent(config)
        self._critic_a = CriticAgentA(config)
        self._critic_b = CriticAgentB(config)
        self._arbiter = ArbiterAgent(config)

    async def run(self, question: str) -> IntelBrief:
        log.info(f"\n{'='*60}")
        log.info(f"[bold white]Veridian — Business Intelligence Debate[/bold white]")
        log.info(f"Question: {question[:120]}{'…' if len(question) > 120 else ''}")
        log.info(f"Max rounds: {self._cfg.max_debate_rounds}  |  "
                 f"Min consensus: {self._cfg.min_consensus_score}")
        log.info(f"{'='*60}\n")

        # Step 0: gather live evidence once
        evidence: EvidencePacket = await self._evidence_tool.fetch(question)

        rounds: list[DebateRound] = []
        current_proposal: Proposal | None = None
        last_decision: ArbiterDecision | None = None

        for round_num in range(1, self._cfg.max_debate_rounds + 1):
            log.info(f"\n[bold]── Round {round_num}/{self._cfg.max_debate_rounds} ──[/bold]")

            # Step 1: Propose / re-propose
            if current_proposal is None:
                current_proposal = await self._proposer.propose(question, evidence)
            elif last_decision and last_decision.directive == "revise" and last_decision.revised_proposal:
                log.info("Using arbiter's revised brief for next round.")
                current_proposal = Proposal(
                    content=last_decision.revised_proposal,
                    model=f"{current_proposal.model}+arbiter",
                )
            else:
                log.info("Escalation — re-drafting from scratch with stronger grounding.")
                escalation_task = (
                    f"{question}\n\n"
                    f"IMPORTANT — PREVIOUS DRAFT FAILED AUDIT REVIEW.\n"
                    f"Prior arbiter reasoning: {last_decision.reasoning if last_decision else 'N/A'}\n"
                    f"Be especially rigorous. Every claim must be grounded in evidence. "
                    f"Make recommended actions more specific and actionable."
                )
                current_proposal = await self._proposer.propose(escalation_task, evidence)

            # Step 2: Both auditors critique in parallel
            log.info("Running Auditor A and Auditor B in parallel…")
            verdict_a, verdict_b = await asyncio.gather(
                self._critic_a.critique(current_proposal, evidence),
                self._critic_b.critique(current_proposal, evidence),
            )

            # Step 3: Arbiter rules
            decision = await self._arbiter.arbitrate(
                current_proposal, verdict_a, verdict_b, evidence, round_num
            )
            last_decision = decision

            debate_round = DebateRound(
                round_number=round_num,
                proposal=current_proposal,
                critique_a=verdict_a,
                critique_b=verdict_b,
                arbiter_notes=decision.reasoning,
            )
            rounds.append(debate_round)

            _log_round_summary(round_num, verdict_a, verdict_b, decision)

            if decision.directive == "finalize":
                log.info(f"\n✅ [bold green]Consensus reached[/bold green] at round {round_num}.")
                break

            if round_num == self._cfg.max_debate_rounds:
                log.warning(
                    f"\n⚠️  Max rounds ({self._cfg.max_debate_rounds}) reached. "
                    f"Finalizing with best available brief."
                )
                break

        # Build final structured IntelBrief
        final_content = _pick_final_content(current_proposal, last_decision)
        final_score = last_decision.consensus_score if last_decision else 0.5
        dissents = _collect_dissents(rounds)

        parsed = _parse_sections(final_content)

        brief = IntelBrief(
            question=question,
            executive_summary=parsed.get("executive_summary", ""),
            key_findings=parsed.get("key_findings", []),
            risks_and_caveats=parsed.get("risks_and_caveats", []),
            recommended_actions=parsed.get("recommended_actions", []),
            confidence=max(0.0, min(1.0, final_score)),
            consensus_score=final_score,
            debate_rounds=len(rounds),
            sources=evidence.urls,
            dissenting_points=dissents,
            models_used={
                "analyst": self._cfg.gemini_model,
                "auditor_a": self._cfg.groq_model,
                "auditor_b": self._cfg.groq_model_b,
                "arbiter": self._cfg.github_model,
                "evidence": "tavily-web-search",
            },
            raw_answer=final_content,
        )

        log.info(
            f"\n[bold]Intel brief ready[/bold] | "
            f"confidence={brief.confidence:.0%} | "
            f"consensus={brief.consensus_score:.2f} | "
            f"rounds={brief.debate_rounds} | "
            f"sources={len(brief.sources)}"
        )
        return brief


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_final_content(proposal: Proposal | None, decision: ArbiterDecision | None) -> str:
    if decision and decision.directive == "revise" and decision.revised_proposal:
        return decision.revised_proposal
    if proposal:
        return proposal.content
    return "[No brief could be produced]"


def _collect_dissents(rounds: list[DebateRound]) -> list[str]:
    if not rounds:
        return []
    last = rounds[-1]
    dissents = []
    for verdict in (last.critique_a, last.critique_b):
        for pt in verdict.critique_points:
            if pt.severity in ("critical", "major"):
                dissents.append(
                    f"[{verdict.critic_id}/{pt.severity}] {pt.description}"
                )
    return dissents


def _parse_sections(text: str) -> dict:
    """
    Parse structured markdown sections from the analyst's brief
    into typed fields for the IntelBrief model.
    """
    result = {
        "executive_summary": "",
        "key_findings": [],
        "risks_and_caveats": [],
        "recommended_actions": [],
    }

    # Executive summary
    m = re.search(
        r"##\s*Executive Summary\s*\n(.*?)(?=\n##|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        result["executive_summary"] = m.group(1).strip()

    # Key findings — bullet list
    m = re.search(
        r"##\s*Key Findings\s*\n(.*?)(?=\n##|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        result["key_findings"] = _extract_bullets(m.group(1))

    # Risks & caveats
    m = re.search(
        r"##\s*Risks.*?\n(.*?)(?=\n##|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        result["risks_and_caveats"] = _extract_bullets(m.group(1))

    # Recommended actions
    m = re.search(
        r"##\s*Recommended Actions\s*\n(.*?)(?=\n##|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if m:
        result["recommended_actions"] = _extract_bullets(m.group(1))

    # Fallback: if executive summary empty, use first 300 chars
    if not result["executive_summary"] and text:
        result["executive_summary"] = text[:300].strip()

    return result


def _extract_bullets(text: str) -> list[str]:
    """Extract markdown bullet points from a section."""
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "• ", "* ", "· ")):
            bullets.append(line[2:].strip())
        elif re.match(r"^\d+\.\s", line):
            bullets.append(re.sub(r"^\d+\.\s", "", line).strip())
    return [b for b in bullets if b]


def _log_round_summary(
    round_num: int,
    verdict_a: CriticVerdict,
    verdict_b: CriticVerdict,
    decision: ArbiterDecision,
) -> None:
    log.info(
        f"\n  Round {round_num} summary:\n"
        f"    Auditor A  → {verdict_a.verdict:8s}  score={verdict_a.score:.2f}\n"
        f"    Auditor B  → {verdict_b.verdict:8s}  score={verdict_b.score:.2f}\n"
        f"    Arbiter    → {decision.directive:8s}  consensus={decision.consensus_score:.2f}\n"
        f"    Reasoning  : {decision.reasoning[:120]}{'…' if len(decision.reasoning)>120 else ''}"
    )
