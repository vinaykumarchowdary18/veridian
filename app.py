"""
app.py — Veridian Gradio UI
Adversarial Business Intelligence — question in, validated brief out.
"""
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

from core.config import load_config
from core.orchestrator import Orchestrator
from core.models import IntelBrief


# ── output saver ─────────────────────────────────────────────────────────────

def _save_outputs(brief: IntelBrief, outputs_dir: str = "./outputs") -> tuple[Path, Path]:
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = brief.question[:40].replace(" ", "_").replace("/", "-")
    slug = "".join(c for c in slug if c.isalnum() or c in ("_", "-"))

    json_path = out / f"{ts}_{slug}.json"
    md_path = out / f"{ts}_{slug}.md"

    json_path.write_text(brief.model_dump_json(indent=2), encoding="utf-8")

    md_lines = [
        f"# Veridian Intelligence Brief\n",
        f"**Question:** {brief.question}\n",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
        f"**Confidence:** {brief.confidence:.0%}  |  "
        f"**Consensus:** {brief.consensus_score:.2f}  |  "
        f"**Debate Rounds:** {brief.debate_rounds}\n\n---\n",
        f"## Executive Summary\n{brief.executive_summary}\n",
    ]
    if brief.key_findings:
        md_lines.append("\n## Key Findings\n")
        md_lines += [f"- {f}\n" for f in brief.key_findings]
    if brief.risks_and_caveats:
        md_lines.append("\n## Risks & Caveats\n")
        md_lines += [f"- {r}\n" for r in brief.risks_and_caveats]
    if brief.recommended_actions:
        md_lines.append("\n## Recommended Actions\n")
        md_lines += [f"- {a}\n" for a in brief.recommended_actions]
    if brief.sources:
        md_lines.append("\n## Sources\n")
        md_lines += [f"- {u}\n" for u in brief.sources]
    if brief.dissenting_points:
        md_lines.append("\n## Minority Dissents (unresolved)\n")
        md_lines += [f"- {d}\n" for d in brief.dissenting_points]
    md_lines.append("\n## Models Used\n")
    for role, model in brief.models_used.items():
        md_lines.append(f"- **{role}**: `{model}`\n")

    md_path.write_text("".join(md_lines), encoding="utf-8")
    return json_path, md_path


# ── confidence badge ──────────────────────────────────────────────────────────

def _confidence_badge(score: float) -> str:
    if score >= 0.75:
        return f"🟢 High Confidence ({score:.0%})"
    elif score >= 0.50:
        return f"🟡 Medium Confidence ({score:.0%})"
    else:
        return f"🔴 Low Confidence ({score:.0%}) — verify manually"


# ── core run function ─────────────────────────────────────────────────────────

async def _run_veridian(question: str) -> IntelBrief:
    config = load_config()
    orchestrator = Orchestrator(config)
    return await orchestrator.run(question)


# ── Gradio handler ────────────────────────────────────────────────────────────

def run_query(question: str):
    if not question or not question.strip():
        return (
            "⚠️ Please enter a business question.",
            "", "", "", "", "", "", ""
        )

    try:
        brief = asyncio.run(_run_veridian(question.strip()))
    except EnvironmentError as e:
        return (
            f"❌ Configuration error: {e}",
            "", "", "", "", "", "", ""
        )
    except Exception as e:
        return (
            f"❌ Error: {e}",
            "", "", "", "", "", "", ""
        )

    # Save outputs
    try:
        _save_outputs(brief)
    except Exception:
        pass  # don't crash UI on save failure

    # Confidence
    confidence_str = _confidence_badge(brief.confidence)
    meta = (
        f"{confidence_str}  ·  "
        f"Consensus Score: {brief.consensus_score:.2f}  ·  "
        f"Debate Rounds: {brief.debate_rounds}  ·  "
        f"Sources: {len(brief.sources)}"
    )

    # Key findings
    findings_md = "\n".join(f"• {f}" for f in brief.key_findings) if brief.key_findings else "No findings extracted."

    # Risks
    risks_md = "\n".join(f"⚠ {r}" for r in brief.risks_and_caveats) if brief.risks_and_caveats else "None flagged."

    # Actions
    actions_md = "\n".join(f"→ {a}" for a in brief.recommended_actions) if brief.recommended_actions else "None provided."

    # Sources
    sources_md = "\n".join(f"[{i}] {u}" for i, u in enumerate(brief.sources, 1)) if brief.sources else "No sources."

    # Dissents
    dissents_md = (
        "\n".join(f"• {d}" for d in brief.dissenting_points)
        if brief.dissenting_points
        else "✅ No unresolved dissents — all auditors reached consensus."
    )

    # Models
    models_md = "\n".join(f"**{role}**: `{model}`" for role, model in brief.models_used.items())

    return (
        meta,
        brief.executive_summary,
        findings_md,
        risks_md,
        actions_md,
        sources_md,
        dissents_md,
        models_md,
    )


# ── Example questions ─────────────────────────────────────────────────────────

EXAMPLES = [
    ["What are the fastest growing B2B SaaS markets in 2025?"],
    ["Compare cloud cost optimization strategies for mid-size tech companies"],
    ["What are the most in-demand skills for data engineering roles right now?"],
    ["Is generative AI adoption slowing down in enterprise in 2025?"],
    ["What are the key risks of adopting microservices architecture for a growing startup?"],
    ["Which industries are seeing the highest ROI from AI automation?"],
    ["What does the job market look like for software engineers in 2025?"],
    ["What are the biggest challenges in MLOps adoption at scale?"],
]


# ── UI Layout ─────────────────────────────────────────────────────────────────

CSS = """
#header { text-align: center; margin-bottom: 10px; }
#question_box { font-size: 16px; }
#submit_btn { background: #1a56db; color: white; font-size: 16px; }
#meta_row { font-size: 14px; color: #555; font-weight: bold; padding: 8px 0; }
.section-label { font-weight: bold; color: #1a56db; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Veridian — Business Intelligence", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🔍 Veridian — Adversarial Business Intelligence
        ### Ask any business or market question. Four AI models debate the answer before you see it.
        *Powered by Gemini · Llama · DeepSeek · GPT-4o-mini · Tavily live search*
        """,
        elem_id="header"
    )

    with gr.Row():
        with gr.Column(scale=5):
            question_input = gr.Textbox(
                label="Your Business Question",
                placeholder="e.g. What are the fastest growing SaaS markets in 2025?",
                lines=2,
                elem_id="question_box",
            )
        with gr.Column(scale=1, min_width=120):
            submit_btn = gr.Button("Analyse →", variant="primary", elem_id="submit_btn")

    gr.Examples(
        examples=EXAMPLES,
        inputs=question_input,
        label="Try an example",
    )

    gr.Markdown("---")

    # Results
    meta_output = gr.Markdown(label="", elem_id="meta_row")

    with gr.Tabs():
        with gr.TabItem("📋 Executive Summary"):
            summary_output = gr.Markdown()

        with gr.TabItem("🔑 Key Findings"):
            findings_output = gr.Markdown()

        with gr.TabItem("⚠️ Risks & Caveats"):
            risks_output = gr.Markdown()

        with gr.TabItem("✅ Recommended Actions"):
            actions_output = gr.Markdown()

        with gr.TabItem("🌐 Sources"):
            sources_output = gr.Markdown()

        with gr.TabItem("🔬 Audit Trail"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Unresolved Dissents**")
                    dissents_output = gr.Markdown()
                with gr.Column():
                    gr.Markdown("**Models Used**")
                    models_output = gr.Markdown()

    gr.Markdown(
        """
        ---
        **How it works:** Your question → Live web search (Tavily) → Analyst drafts a brief (Gemini 2.5 Flash)
        → Two independent auditors critique it in parallel (Llama 3.3 via Groq · DeepSeek v3 via OpenRouter)
        → Arbiter scores consensus and either finalizes, revises, or escalates (GPT-4o-mini via GitHub Models)
        → You receive a validated brief with confidence score and full audit trail.

        *Outputs are saved to `./outputs/` as JSON and Markdown for your records.*
        """,
        elem_id="footer_note"
    )

    submit_btn.click(
        fn=run_query,
        inputs=[question_input],
        outputs=[
            meta_output,
            summary_output,
            findings_output,
            risks_output,
            actions_output,
            sources_output,
            dissents_output,
            models_output,
        ],
    )

    question_input.submit(
        fn=run_query,
        inputs=[question_input],
        outputs=[
            meta_output,
            summary_output,
            findings_output,
            risks_output,
            actions_output,
            sources_output,
            dissents_output,
            models_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=False)
