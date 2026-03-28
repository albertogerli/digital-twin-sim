"""Web research module — gathers online context before scenario analysis."""

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Search query generation prompt
SEARCH_QUERY_PROMPT = """Given this simulation scenario brief, generate 3-5 focused web search queries
to gather real-world context, recent news, key stakeholders, and public opinion data.

BRIEF: {brief}

Respond with JSON:
{{
  "queries": ["query1", "query2", "query3"],
  "key_entities": ["entity1", "entity2"],
  "context_needed": "what specific context would make the simulation more realistic"
}}"""

# Context synthesis prompt
SYNTHESIS_PROMPT = """You are a research analyst. Synthesize these web search results into
a concise context briefing for a simulation designer.

ORIGINAL SCENARIO BRIEF: {brief}

SEARCH RESULTS:
{search_results}

Produce a structured context document (max 2000 words) covering:
1. BACKGROUND: Key facts, timeline, and current state of the issue
2. KEY STAKEHOLDERS: Real people, organizations, and their known positions
3. PUBLIC OPINION: Surveys, social media sentiment, demographic splits
4. RECENT DEVELOPMENTS: Latest news and events (last 6 months)
5. CONTROVERSIAL ASPECTS: Points of tension, polarization triggers
6. ECONOMIC/SOCIAL DATA: Relevant statistics and numbers

Be specific — use real names, real numbers, real quotes when available.
If information is not found, say so rather than making things up."""


async def research_context(
    brief: str,
    llm,
    progress_callback=None,
) -> str:
    """Research web context for a scenario brief.

    Returns a synthesized context string to enhance the brief analysis.
    """
    try:
        # Step 1: Generate search queries
        if progress_callback:
            await progress_callback("round_phase", {
                "phase": "web_research",
                "message": "Generazione query di ricerca..."
            })

        query_result = await llm.generate_json(
            prompt=SEARCH_QUERY_PROMPT.format(brief=brief),
            temperature=0.3,
            max_output_tokens=500,
            component="web_research",
        )

        queries = query_result.get("queries", [])
        if not queries:
            queries = [brief[:100]]

        logger.info(f"Web research: {len(queries)} queries generated")

        # Step 2: Search the web
        if progress_callback:
            await progress_callback("round_phase", {
                "phase": "web_research",
                "message": f"Ricerca online ({len(queries)} query)..."
            })

        search_results = await _execute_searches(queries)

        if not search_results.strip():
            logger.warning("No web results found, proceeding without context")
            return ""

        # Step 3: Synthesize results
        if progress_callback:
            await progress_callback("round_phase", {
                "phase": "web_research",
                "message": "Sintesi dei risultati..."
            })

        context = await llm.generate_text(
            prompt=SYNTHESIS_PROMPT.format(
                brief=brief,
                search_results=search_results[:8000],
            ),
            temperature=0.3,
            max_output_tokens=3000,
            component="web_research",
        )

        logger.info(f"Web research: synthesized {len(context)} chars of context")
        return context

    except Exception as e:
        logger.warning(f"Web research failed (non-fatal): {e}")
        return ""


async def _execute_searches(queries: list[str]) -> str:
    """Execute web searches using DuckDuckGo HTML (no API key needed)."""
    import aiohttp
    import ssl
    import certifi

    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    # Try with certifi SSL context first, fall back to no-verify
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    try:
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    except Exception:
        connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        for query in queries[:5]:
            try:
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        snippets = _extract_snippets(html)
                        if snippets:
                            results.append(f"\n### Query: {query}\n")
                            for s in snippets[:5]:
                                results.append(f"- {s}")
                            logger.info(f"Search '{query[:50]}': {len(snippets)} snippets")
                        else:
                            logger.warning(f"Search '{query[:50]}': no snippets extracted")
                    else:
                        logger.warning(f"Search '{query[:50]}': status {resp.status}")
                await asyncio.sleep(0.5)  # Rate limit
            except Exception as e:
                logger.warning(f"Search failed for '{query[:50]}': {e}")
                continue

    if not results:
        logger.warning("All web searches returned empty — SSL or network issue?")

    return "\n".join(results)


def _extract_snippets(html: str) -> list[str]:
    """Extract search result snippets from DuckDuckGo HTML."""
    snippets = []
    # Extract result snippets
    for match in re.finditer(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL):
        text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        if text and len(text) > 30:
            snippets.append(text)
    # Also try result titles
    for match in re.finditer(r'class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL):
        text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        if text and len(text) > 10:
            snippets.append(f"[Title] {text}")
    return snippets[:8]
