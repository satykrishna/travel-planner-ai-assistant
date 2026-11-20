"""Bounded HTTP requests for the notebook's external search providers."""

import os

import requests
from langchain_core.tools import tool

PROVIDER_TIMEOUT = (5, 30)


class GoogleSearch:
    """Small adapter retaining the notebook's get_dict interface."""

    def __init__(self, params):
        self.params = params

    def get_dict(self):
        try:
            response = requests.get(
                "https://serpapi.com/search.json", params=self.params, timeout=PROVIDER_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            # requests exception strings can contain the API key in the URL.
            raise RuntimeError("Search provider request failed") from None


@tool("web_search_tool")
def web_search_tool(query: str) -> dict:
    """Web Search Tool: retrieve current travel information and source links."""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": 3,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
            },
            timeout=PROVIDER_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "answer": data.get("answer"),
            "results": [
                {key: item.get(key) for key in ("title", "url", "content")}
                for item in data.get("results", [])
            ],
        }
    except (requests.RequestException, ValueError, KeyError):
        return {"status": "error", "message": "Web search unavailable; do not invent sources."}
