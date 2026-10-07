from classes.context_manager import ContextResult
from classes.open_router_client import OpenRouterClient


class FactsExtractor:
    """Extract episodic memories and semantic facts from conversation via structured LLM output."""

    def __init__(self, extraction_client: OpenRouterClient) -> None:
        self._extraction_client = extraction_client

    def build_facts_extraction_prompt(self) -> None:
        """Build the facts extraction prompt."""

    def extract_facts_from_context_result(self, context_result: ContextResult) -> None:
        """Extract facts from a context result."""
