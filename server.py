"""
server.py — Veridian FastAPI server
Serves the static frontend and exposes /analyse endpoint.
Includes full security layer: rate limiting, input validation, prompt injection detection.
Run: uvicorn server:app --reload --port 8000
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import load_config
from core.orchestrator import Orchestrator
from core.security import validate_question, validate_api_keys
from core.logger import get_logger

log = get_logger(__name__)

app = FastAPI(title="Veridian", docs_url=None, redoc_url=None)

# CORS — restrict to same origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Startup validation ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_checks():
    try:
        config = load_config()
        warnings = validate_api_keys(config)
        if warnings:
            for w in warnings:
                log.warning(f"API key warning: {w}")
        else:
            log.info("All API keys validated successfully.")
    except EnvironmentError as e:
        log.error(f"Startup failed — missing keys: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/analyse")
async def analyse(req: QuestionRequest, request: Request):
    # Get client IP for rate limiting
    ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()

    # Security validation
    validation = validate_question(req.question, ip=ip)
    if not validation.valid:
        status = 429 if "Too many requests" in validation.error else 400
        return JSONResponse({"error": validation.error}, status_code=status)

    # Load config
    try:
        config = load_config()
    except EnvironmentError as e:
        return JSONResponse({"error": f"Server configuration error: {e}"}, status_code=500)

    # Run debate pipeline
    try:
        orchestrator = Orchestrator(config)
        brief = await orchestrator.run(validation.sanitised_question)
        return JSONResponse(brief.model_dump())
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return JSONResponse(
            {"error": "Analysis failed. Please try again."},
            status_code=500,
        )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "veridian", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
