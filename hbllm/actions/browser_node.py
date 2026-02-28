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

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class BrowserNode(Node):
    """
    Service node that acts as the model's "web browser".
    """

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.ACTION, capabilities=["web_search", "web_scrape"])

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
            
            def _search_and_scrape():
                from duckduckgo_search import DDGS
                import requests
                from bs4 import BeautifulSoup

                logger.info("Executing Web Search for: '%s'", query)
                results = []
                
                with DDGS() as ddgs:
                    # Get top N links
                    search_results = list(ddgs.text(query, max_results=max_results))
                    
                    for result in search_results:
                        url = result.get("href")
                        title = result.get("title")
                        snippet = result.get("body")
                        logger.info("Scraping URL: %s", url)
                        
                        try:
                            # Fetch page content with a generic browser user-agent
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            }
                            resp = requests.get(url, headers=headers, timeout=10)
                            resp.raise_for_status()
                            
                            soup = BeautifulSoup(resp.content, "html.parser")
                            
                            # Remove scripts and styles
                            for script in soup(["script", "style", "nav", "footer", "header"]):
                                script.extract()
                                
                            text = soup.get_text(separator=' ')
                            
                            # Clean whitespace
                            text = re.sub(r'\s+', ' ', text).strip()
                            
                            # Truncate to save context window (roughly 1000 words max)
                            words = text.split()
                            if len(words) > 1000:
                                text = " ".join(words[:1000]) + "... [TRUNCATED]"
                                
                            results.append({
                                "title": title,
                                "url": url,
                                "search_snippet": snippet,
                                "page_content": text
                            })
                            
                        except Exception as e:
                            logger.warning("Failed to scrape %s: %s", url, e)
                            # Fallback to just the snippet
                            results.append({
                                "title": title,
                                "url": url,
                                "search_snippet": snippet,
                                "page_content": f"[Could not scrape full text: {e}]"
                            })
                            
                return results

            # Run in thread to not block the asyncio message loop
            search_results = await asyncio.to_thread(_search_and_scrape)
            
            if not search_results:
                return message.create_response({"results": [], "text": "No search results found."})
                
            # Format nicely for the LLM
            formatted_text = f"Web Search Results for '{query}':\n\n"
            for r in search_results:
                formatted_text += f"---\n🌐 **{r['title']}**\nURL: {r['url']}\nSummary: {r['search_snippet']}\n\nContent:\n{r['page_content']}\n\n"

            return message.create_response({
                "results": search_results,
                "text": formatted_text,
                "domain": "browser"
            })
            
        except ImportError as ie:
            logger.error("Missing dependency: %s", ie)
            return message.create_error("Missing dependencies. Please run: pip install duckduckgo-search beautifulsoup4 requests")
        except Exception as e:
            logger.error("Web search failed: %s", e)
            return message.create_error(str(e))
