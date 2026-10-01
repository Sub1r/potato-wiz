"""
services/ai
~~~~~~~~~~~
AI provider abstraction layer for Potato Wiz.

Exports the public surface that the rest of the application uses:

    from services.ai import get_provider, AIProvider

The concrete provider (Ollama or OpenRouter) is chosen by
``services.ai_optimizer`` based on configuration; callers never need to
import provider classes directly.
"""
from services.ai.base import AIProvider, AIProviderError  # noqa: F401

__all__ = ["AIProvider", "AIProviderError"]
