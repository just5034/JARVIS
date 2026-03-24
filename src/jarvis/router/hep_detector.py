"""HEP subdomain detection — keyword + classifier hybrid."""

from __future__ import annotations

from jarvis.config import RouterConfig


class HEPDetector:
    """Detects whether a physics/code query is HEP-specific."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self._keywords = set(config.hep_subdomain.keywords)

    def detect(self, query: str) -> bool:
        query_lower = query.lower()
        for keyword in self._keywords:
            if keyword.lower() in query_lower:
                return True
        return False
