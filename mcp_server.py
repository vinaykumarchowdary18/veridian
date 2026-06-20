"""
mcp_server.py — Veridian MCP Server
Exposes the Veridian debate engine as an MCP-compatible tool server.
Any MCP-compatible client (Claude Desktop, ADK, etc.) can call run_intelligence_brief().

Run: python mcp_server.py
"""
import asyncio
import json
import sys
from typing import Any

from core.config import load_config
from core.orchestrator import Orchestrator


# ── MCP Protocol constants ────────────────────────────────────────────────────

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "veridian",
    "version": "1.0.0",
    "description": "Adversarial Business Intelligence — four AI models debate every answer.",
}

TOOLS = [
    {
        "name": "run_intelligence_brief",
        "description": (
            "Ask any business or market question. Veridian fetches live web evidence, "
            "then runs an adversarial multi-agent debate: Analyst drafts → two Auditors "
            "critique in parallel → Arbiter scores consensus. Returns a validated "
            "intelligence brief with confidence score, key findings, risks, recommended "
            "actions, and full audit trail."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The business or market question to analyse.",
                }
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_confidence_explanation",
        "description": (
            "Explain what a given Veridian confidence score means and how to interpret "
            "the audit trail for a previously returned brief."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0 from a brief.",
                },
                "consensus_score": {
                    "type": "number",
                    "description": "Consensus score between 0.0 and 1.0 from a brief.",
                },
                "debate_rounds": {
                    "type": "integer",
                    "description": "Number of debate rounds the brief required.",
                },
            },
            "required": ["confidence"],
        },
    },
]


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def handle_run_intelligence_brief(args: dict) -> dict:
    question = args.get("question", "").strip()
    if not question:
        return {"error": "question is required and cannot be empty."}

    if len(question) > 1000:
        return {"error": "Question too long. Please keep it under 1000 characters."}

    try:
        config = load_config()
        orchestrator = Orchestrator(config)
        brief = await orchestrator.run(question)

        return {
            "question": brief.question,
            "executive_summary": brief.executive_summary,
            "key_findings": brief.key_findings,
            "risks_and_caveats": brief.risks_and_caveats,
            "recommended_actions": brief.recommended_actions,
            "confidence": round(brief.confidence, 3),
            "consensus_score": round(brief.consensus_score, 3),
            "debate_rounds": brief.debate_rounds,
            "sources": brief.sources,
            "dissenting_points": brief.dissenting_points,
            "models_used": brief.models_used,
        }
    except EnvironmentError as e:
        return {"error": f"Configuration error: {e}"}
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}


def handle_get_confidence_explanation(args: dict) -> dict:
    confidence = float(args.get("confidence", 0))
    consensus = float(args.get("consensus_score", confidence))
    rounds = int(args.get("debate_rounds", 1))

    if confidence >= 0.75:
        level = "High"
        interpretation = (
            "Both auditors broadly agreed with the analyst's brief. "
            "This answer is safe to act on, though you should still verify "
            "critical decisions with primary sources."
        )
    elif confidence >= 0.50:
        level = "Medium"
        interpretation = (
            "The auditors raised some valid concerns that were partially addressed. "
            "Verify the key claims before making significant decisions. "
            "Check the dissenting points section for what was flagged."
        )
    else:
        level = "Low"
        interpretation = (
            "Significant disagreement between auditors. The brief may contain "
            "unsupported claims or missing perspectives. "
            "Treat as a starting point only — manual verification strongly recommended."
        )

    rounds_note = (
        "The brief was finalised in one round — auditors accepted it quickly."
        if rounds == 1
        else f"The brief required {rounds} debate rounds — it was challenged and revised before finalisation."
    )

    return {
        "confidence_level": level,
        "score": confidence,
        "interpretation": interpretation,
        "rounds_note": rounds_note,
        "recommendation": (
            "Act on it directly."
            if confidence >= 0.75
            else "Spot-check key claims."
            if confidence >= 0.5
            else "Use as a starting point only."
        ),
    }


# ── MCP JSON-RPC handler ──────────────────────────────────────────────────────

async def handle_request(request: dict) -> dict | None:
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    def ok(result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    # Notifications — no response
    if req_id is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return ok({
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": SERVER_INFO,
            "capabilities": {"tools": {}},
        })

    elif method == "tools/list":
        return ok({"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name == "run_intelligence_brief":
            result = await handle_run_intelligence_brief(tool_args)
            is_error = "error" in result
            return ok({
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "isError": is_error,
            })

        elif tool_name == "get_confidence_explanation":
            result = handle_get_confidence_explanation(tool_args)
            return ok({
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "isError": False,
            })

        else:
            return err(-32601, f"Unknown tool: {tool_name}")

    elif method == "ping":
        return ok({})

    else:
        return err(-32601, f"Method not found: {method}")


# ── stdio transport ───────────────────────────────────────────────────────────

async def run_stdio():
    """Run MCP server over stdio (standard MCP transport)."""
    sys.stderr.write(f"Veridian MCP Server v{SERVER_INFO['version']} started\n")
    sys.stderr.flush()

    loop = asyncio.get_event_loop()

    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
                continue

            response = await handle_request(request)

            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        except EOFError:
            break
        except Exception as e:
            sys.stderr.write(f"MCP server error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    asyncio.run(run_stdio())
