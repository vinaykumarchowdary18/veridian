"""
tools/evidence.py — Tavily-powered live evidence retrieval.
Called once before any agent sees the question.
Grounds the entire debate in current web facts.
"""
import asyncio
from tavily import TavilyClient
from core.models import EvidencePacket
from core.logger import get_logger

log = get_logger(__name__)


class EvidenceTool:
    def __init__(self, api_key: str, max_results: int = 6):
        self._client = TavilyClient(api_key=api_key)
        self._max_results = max_results

    async def fetch(self, query: str) -> EvidencePacket:
        log.info(f"[bold]Evidence[/bold] → searching: {query[:80]}…")
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=self._max_results,
                    include_answer=True,
                ),
            )
        except Exception as exc:
            log.warning(f"Tavily search failed ({exc}); continuing without live evidence.")
            return EvidencePacket(query=query, snippets=[], urls=[])

        snippets: list[str] = []
        urls: list[str] = []
        for item in result.get("results", []):
            content = item.get("content", "").strip()
            url = item.get("url", "")
            if content:
                snippets.append(content)
            if url:
                urls.append(url)

        packet = EvidencePacket(
            query=query,
            snippets=snippets[:self._max_results],
            urls=urls[:self._max_results],
            raw_answer=result.get("answer"),
        )
        log.info(f"Evidence gathered: {len(snippets)} snippets from {len(urls)} sources.")
        return packet
