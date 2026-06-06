"""
Agentic Browsing Node.

Listens for `task.execute.search` payloads.
Uses DuckDuckGo to find relevant URLs and BeautifulSoup to scrape the
top result, returning the cleaned text back to the pipeline to augment
the Language Model's generation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hbllm.network.messages import Message
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class BrowserNode(Node):
    """
    Service node that acts as the model's "web browser".
    """

    def __init__(self, node_id: str) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.ACTION,
            capabilities=["web_search", "web_scrape"],
        )

    async def on_start(self) -> None:
        """Subscribe to search execution topics."""
        logger.info("Starting BrowserNode")
        await self.bus.subscribe("task.execute.search", self.handle_search)

    async def on_stop(self) -> None:
        logger.info("Stopping BrowserNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_search(self, message: Message) -> Message | None:
        """
        Handles `task.execute.search` messages.
        Payload expects:
            query: str -> the search term
            max_results: int -> optional (default 1)
        """
        payload = message.payload
        query = payload.get("query")
        max_results = int(payload.get("max_results", 1))

        if not query:
            return message.create_error("Missing 'query' payload parameter")

        try:
            import asyncio

            def _search_and_scrape() -> list[dict[str, Any]]:
                import concurrent.futures

                import requests  # type: ignore
                from bs4 import BeautifulSoup  # type: ignore
                from lxml.html import document_fromstring  # type: ignore

                logger.info("Executing Web Search for: '%s'", query)

                # Bypass the duckduckgo_search DDGS library which has a bug
                # that hardcodes the 'bing' backend and returns empty results.
                # POST directly to the DDG HTML endpoint instead.
                search_headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://html.duckduckgo.com/",
                }
                search_resp = requests.post(
                    "https://html.duckduckgo.com/html",
                    data={"q": query, "b": ""},
                    headers=search_headers,
                    timeout=10,
                )
                search_resp.raise_for_status()

                tree = document_fromstring(search_resp.content)
                search_results: list[dict[str, str]] = []
                seen: set[str] = set()
                for div in tree.xpath("//div[h2]"):
                    hrefs = div.xpath("./a/@href")
                    href = str(hrefs[0]) if hrefs else ""
                    if (
                        not href
                        or href in seen
                        or href.startswith(
                            (
                                "http://www.google.com/search?q=",
                                "https://duckduckgo.com/y.js?ad_domain",
                            )
                        )
                    ):
                        continue
                    seen.add(href)
                    titles = div.xpath("./h2/a/text()")
                    title = str(titles[0]).strip() if titles else ""
                    bodies = div.xpath("./a//text()")
                    body = "".join(str(x) for x in bodies).strip() if bodies else ""
                    search_results.append({"title": title, "href": href, "body": body})
                    if len(search_results) >= max_results:
                        break

                def scrape_url(result: dict[str, Any]) -> dict[str, Any]:
                    url = str(result.get("href", ""))
                    title = str(result.get("title", ""))
                    snippet = str(result.get("body", ""))
                    logger.info("Scraping URL: %s", url)

                    try:
                        # Fetch page content with a generic browser user-agent
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        }
                        # Use 4 second timeout to fail fast and fallback to snippet rather than blocking the pipeline
                        resp = requests.get(url, headers=headers, timeout=4)
                        resp.raise_for_status()

                        soup = BeautifulSoup(resp.content, "html.parser")

                        # Remove scripts and styles
                        for script in soup(["script", "style", "nav", "footer", "header"]):
                            script.extract()

                        text = str(soup.get_text(separator=" "))

                        # Clean whitespace
                        text = re.sub(r"\s+", " ", text).strip()

                        # Truncate to save context window (roughly 1000 words max)
                        words = text.split()
                        if len(words) > 1000:
                            text = " ".join(words[:1000]) + "... [TRUNCATED]"

                        return {
                            "title": title,
                            "url": url,
                            "search_snippet": snippet,
                            "page_content": text,
                        }

                    except (
                        RuntimeError,
                        ValueError,
                        TypeError,
                        OSError,
                        KeyError,
                        ConnectionError,
                        requests.RequestException,
                    ) as e:
                        logger.warning("Failed to scrape %s: %s", url, e)
                        # Fallback to just the snippet
                        return {
                            "title": title,
                            "url": url,
                            "search_snippet": snippet,
                            "page_content": f"[Could not scrape full text: {e}]",
                        }

                results: list[dict[str, Any]] = []
                if search_results:
                    # Scrape URLs in parallel
                    workers = min(len(search_results), max_results, 5)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        results = list(executor.map(scrape_url, search_results))

                return results

            # Run in thread to not block the asyncio message loop
            search_results_data = await asyncio.to_thread(_search_and_scrape)

            if not search_results_data:
                return message.create_response({"results": [], "text": "No search results found."})

            # Format nicely for the LLM
            formatted_text = f"Web Search Results for '{query}':\n\n"
            for r in search_results_data:
                formatted_text += f"---\n🌐 **{r['title']}**\nURL: {r['url']}\nSummary: {r['search_snippet']}\n\nContent:\n{r['page_content']}\n\n"

            return message.create_response(
                {"results": search_results_data, "text": formatted_text, "domain": "browser"}
            )

        except ImportError as ie:
            logger.error("Missing dependency: %s", ie)
            return message.create_error(
                "Missing dependencies. Please run: pip install duckduckgo-search beautifulsoup4 requests"
            )
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Web search failed: %s", e)
            return message.create_error(str(e))
