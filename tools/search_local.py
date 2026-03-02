from typing import Dict
from .base import Tool

_FAKE_KB = {
    "semiconductor supply chain": "Global semiconductor supply chains depend on East Asian fabrication hubs...",
    "inventory management": "Inventory management failures often amplify supply chain shocks...",
}


class SearchLocalKnowledgeTool(Tool):
    name = "search_local_knowledge"
    description = "Searches a static local knowledge base."

    def run(self, input: Dict) -> str:
        query = input.get("query")
        if not query:
            return "No query provided."

        query_lower = query.lower()
        for key, value in _FAKE_KB.items():
            if key in query_lower:
                return value

        return "No relevant information found in local knowledge base."