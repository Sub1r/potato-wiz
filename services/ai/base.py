"""
services/ai/base.py
~~~~~~~~~~~~~~~~~~~
Abstract base class for all AI providers.

Adding a new provider means subclassing ``AIProvider`` and implementing
``complete()``.  Nothing else in the codebase needs to change.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AIProviderError(Exception):
    """Raised when a provider call fails unrecoverably."""


class AIProvider(ABC):
    """
    Minimal interface that every AI provider must implement.

    The provider is responsible for:
    - Sending a structured prompt to the underlying model.
    - Returning the raw response text (JSON or otherwise).
    - Handling its own network errors and timeouts.
    - NEVER logging API keys or credentials.

    Higher-level concerns (prompt construction, JSON parsing, validation)
    live in ``services/ai_optimizer.py``, not here.
    """

    #: Human-readable name used in logging and result metadata.
    provider_name: str = "base"

    #: The model identifier sent to the API (e.g. "llama3.2").
    model_name: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if this provider can be used right now.

        This should be a lightweight check (env-var presence, not a live
        network ping) so it does not slow down every request.
        """

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Send a chat-completion request and return the assistant's response text.

        Parameters
        ----------
        system_prompt:
            The system instructions establishing the assistant's role and rules.
        user_prompt:
            The user-facing request with all context (hardware, game, etc.).
        tools:
            Optional list of tool/function definitions in the provider's
            native format (e.g. OpenRouter server-tools).

        Returns
        -------
        str
            Raw response text from the model.  May be JSON, may be prose.
            The caller is responsible for parsing.

        Raises
        ------
        AIProviderError
            On any unrecoverable failure.  The caller (ai_optimizer) catches
            this and falls back to the deterministic optimizer.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
