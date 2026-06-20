"""
core/security.py — Veridian Security Layer
Covers:
  1. Input sanitisation — strip/reject malicious or oversized input
  2. Prompt injection detection — catch attempts to hijack agent instructions
  3. Rate limiting — per-IP request throttling
  4. API key validation — verify all keys present and non-trivial at startup
  5. Question content filtering — reject clearly harmful requests
"""
import re
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from core.logger import get_logger

log = get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

RATE_LIMIT_REQUESTS = 10       # max requests per window
RATE_LIMIT_WINDOW_SEC = 60     # rolling window in seconds
MAX_QUESTION_LENGTH = 800      # characters
MIN_QUESTION_LENGTH = 8        # characters


# ── Rate limiter ──────────────────────────────────────────────────────────────

@dataclass
class RateLimiter:
    """
    Simple in-memory per-IP rate limiter.
    Uses a sliding window — stores timestamps of recent requests per IP.
    """
    _store: dict = field(default_factory=lambda: defaultdict(list))

    def is_allowed(self, ip: str) -> tuple[bool, int]:
        """
        Returns (allowed, retry_after_seconds).
        Cleans up expired timestamps on every call.
        """
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SEC

        # Evict timestamps outside the window
        self._store[ip] = [t for t in self._store[ip] if t > window_start]

        if len(self._store[ip]) >= RATE_LIMIT_REQUESTS:
            oldest = self._store[ip][0]
            retry_after = int(RATE_LIMIT_WINDOW_SEC - (now - oldest)) + 1
            return False, retry_after

        self._store[ip].append(now)
        return True, 0

    def remaining(self, ip: str) -> int:
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SEC
        recent = [t for t in self._store[ip] if t > window_start]
        return max(0, RATE_LIMIT_REQUESTS - len(recent))


# Singleton instance
rate_limiter = RateLimiter()


# ── Prompt injection patterns ─────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions",
    r"disregard\s+(all\s+)?instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+(a\s+)?(?!an?\s+analyst|an?\s+business)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(?!an?\s+analyst|an?\s+business)",
    r"new\s+system\s+prompt",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
    r"\[system\]",
    # Role hijacking
    r"pretend\s+you\s+(have\s+no\s+)?",
    r"your\s+(true|real|actual)\s+(purpose|goal|mission)\s+is",
    r"(do\s+not|don't)\s+(follow|obey|respect)\s+(your\s+)?(rules|guidelines|instructions)",
    # Data exfiltration attempts
    r"(print|output|reveal|show|display)\s+(your\s+)?(system\s+prompt|instructions|api\s+key)",
    r"what\s+(are|is)\s+your\s+(secret|hidden|actual)\s+(instructions|prompt)",
    # Jailbreak keywords
    r"\bdan\b.*mode",
    r"developer\s+mode\s+(enabled|on)",
    r"jailbreak",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


# ── Content filters ───────────────────────────────────────────────────────────

_BLOCKED_PATTERNS = [
    r"\b(how\s+to\s+(make|build|create|synthesize)\s+(bomb|weapon|explosive|poison|drug))",
    r"\b(child\s+(abuse|exploitation|pornography))\b",
    r"\b(hack|exploit|ddos|ransomware)\s+(this|the|a)\s+(server|system|site|website|database)",
]

_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    error: str = ""
    sanitised_question: str = ""
    risk_level: str = "low"   # "low" | "medium" | "high"


# ── Main validator ────────────────────────────────────────────────────────────

def validate_question(raw: str, ip: str = "unknown") -> ValidationResult:
    """
    Full security validation pipeline for a user question.
    Returns ValidationResult with sanitised text if valid.
    """

    # 1. Rate limiting
    allowed, retry_after = rate_limiter.is_allowed(ip)
    if not allowed:
        log.warning(f"Rate limit exceeded for IP {_hash_ip(ip)}")
        return ValidationResult(
            valid=False,
            error=f"Too many requests. Please wait {retry_after} seconds before trying again.",
            risk_level="medium",
        )

    # 2. Basic presence check
    if not raw or not raw.strip():
        return ValidationResult(valid=False, error="Question cannot be empty.")

    # 3. Length checks
    stripped = raw.strip()
    if len(stripped) < MIN_QUESTION_LENGTH:
        return ValidationResult(
            valid=False,
            error=f"Question too short. Please ask a complete question (minimum {MIN_QUESTION_LENGTH} characters).",
        )

    if len(stripped) > MAX_QUESTION_LENGTH:
        return ValidationResult(
            valid=False,
            error=f"Question too long. Please keep it under {MAX_QUESTION_LENGTH} characters (yours: {len(stripped)}).",
        )

    # 4. Sanitise — strip null bytes, control characters, excessive whitespace
    sanitised = _sanitise(stripped)

    # 5. Prompt injection detection
    for pattern in _INJECTION_RE:
        if pattern.search(sanitised):
            log.warning(f"Prompt injection attempt detected from IP {_hash_ip(ip)}: {sanitised[:80]}")
            return ValidationResult(
                valid=False,
                error="Your question contains patterns that look like attempts to override the AI system. Please ask a genuine business question.",
                risk_level="high",
            )

    # 6. Content filtering
    for pattern in _BLOCKED_RE:
        if pattern.search(sanitised):
            log.warning(f"Blocked content from IP {_hash_ip(ip)}: {sanitised[:80]}")
            return ValidationResult(
                valid=False,
                error="This type of question cannot be processed. Please ask a business or market intelligence question.",
                risk_level="high",
            )

    # 7. All clear
    log.info(f"Question validated — {len(sanitised)} chars, IP {_hash_ip(ip)}, remaining: {rate_limiter.remaining(ip)}/{RATE_LIMIT_REQUESTS}")
    return ValidationResult(
        valid=True,
        sanitised_question=sanitised,
        risk_level="low",
    )


def validate_api_keys(config) -> list[str]:
    """
    Validate that all required API keys are present and non-trivial.
    Returns list of warning strings (empty = all good).
    """
    warnings = []
    checks = {
        "GEMINI_API_KEY": config.gemini_api_key,
        "GROQ_API_KEY": config.groq_api_key,
        "OPENROUTER_API_KEY": config.openrouter_api_key,
        "GITHUB_TOKEN": config.github_token,
        "TAVILY_API_KEY": config.tavily_api_key,
    }
    placeholder_patterns = [
        "your_", "placeholder", "changeme", "example",
        "xxx", "yyy", "zzz", "test", "dummy",
    ]
    for name, value in checks.items():
        if not value:
            warnings.append(f"{name} is missing")
        elif len(value) < 8:
            warnings.append(f"{name} looks too short to be valid")
        elif any(p in value.lower() for p in placeholder_patterns):
            warnings.append(f"{name} appears to be a placeholder — replace with a real key")
    return warnings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise(text: str) -> str:
    """Strip control characters, null bytes, and normalise whitespace."""
    # Remove null bytes and other control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {3,}", " ", text)
    return text.strip()


def _hash_ip(ip: str) -> str:
    """One-way hash IP for logging — don't log raw IPs."""
    return hashlib.sha256(ip.encode()).hexdigest()[:8]
