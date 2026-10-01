"""
services/ai/vision.py
~~~~~~~~~~~~~~~~~~~~~
Vision provider abstraction for screenshot-based settings extraction.

Architecture
------------
The optimizer never knows provider-specific image request details.  It calls::

    provider.analyze_image(image_bytes=data_url, prompt=..., game_context=...)

and receives raw model text.  Request shapes live in the provider:

- ``OpenRouterVisionProvider``  — OpenAI-compatible multimodal chat API
  (``content: [{"type": "text"}, {"type": "image_url"}]``)
- ``OllamaVisionProvider``     — local Ollama multimodal chat format
  (``messages: [{"role": "user", "content": ..., "images": [base64]}]``)

Vision capabilities are verified before an image is sent.  For OpenRouter the
model catalog must list ``image`` in ``input_modalities``; a text-only model
produces a clear error instead of silently receiving an image.

Privacy: screenshots are only ever held in memory, are never written to disk,
and no image or base64 data is logged.  With Ollama configured, images stay local.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from config import Config
from services.ai.base import AIProviderError

logger = logging.getLogger(__name__)

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Friendly, user-facing errors (never contain image data).
ERR_NO_VISION_PROVIDER = (
    "No vision-capable AI provider is configured. Set "
    "OPENROUTER_VISION_MODEL or OLLAMA_VISION_MODEL to analyze a screenshot."
)
ERR_MODEL_NOT_VISION = (
    "Your configured AI model does not support image analysis. "
    "Configure a vision-capable model."
)
ERR_OLLAMA_UNAVAILABLE = (
    "Local Ollama vision is unavailable. Make sure Ollama is running and "
    "OLLAMA_VISION_MODEL names an installed vision model."
)


class VisionError(AIProviderError):
    """Raised when a screenshot cannot be analyzed."""


class VisionProvider(ABC):
    """Minimal interface every vision provider implements."""

    provider_name: str = "vision"

    #: Model identifier used for image analysis.
    model_name: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """True when this provider could run a vision request right now."""

    @abstractmethod
    def analyze_image(
        self,
        image_data_url: str,
        prompt: str,
        game_context: Optional[Dict[str, Any]] = None,
        system_prompt: str = "",
    ) -> str:
        """
        Analyze one screenshot and return the model's raw text response.

        Parameters
        ----------
        image_data_url:
            ``data:image/png;base64,...`` URL produced by
            ``services.screenshot.preprocess_image``.
        prompt:
            The user-turn instruction.
        game_context:
            Optional game dict, forwarded by the provider only for logging or
            provider-specific templating.
        system_prompt:
            System instructions; the provider has a sensible default.

        Raises
        ------
        VisionError
            When the provider is unavailable, cannot accept images, or fails.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"


# ── OpenRouter vision ─────────────────────────────────────────────────────────

class OpenRouterVisionProvider(VisionProvider):
    """Multimodal chat requests against OpenRouter's chat-completions API."""

    provider_name = "openrouter_vision"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self._api_key = Config.OPENROUTER_API_KEY if api_key is None else api_key
        # Text and vision models are configured independently.
        self.model_name = model or Config.OPENROUTER_VISION_MODEL
        self.timeout = timeout or Config.OPENROUTER_VISION_TIMEOUT

    def is_available(self) -> bool:
        return bool(
            self._api_key and self._api_key.strip()
            and self.model_name and self.model_name.strip()
        )

    def verify_vision_support(self) -> Dict[str, Any]:
        """
        Confirm the configured model accepts image input.

        Uses the same model-catalog mechanism as the text provider.  A model is
        vision-capable only when ``image`` is present in its input modalities.

        Raises
        ------
        VisionError
            When the catalog positively reports that the model cannot read
            images.
        """
        from services.ai.openrouter_provider import fetch_model_info

        info = fetch_model_info(self.model_name, api_key=self._api_key,
                                timeout=self.timeout)
        if info is None:
            # Capability unknown (catalog unreachable).  Allowed through, but
            # logged so a text-only model is easy to spot.
            logger.warning(
                "Vision capability of model %s could not be confirmed from the "
                "OpenRouter catalog; attempting the request anyway.",
                self.model_name,
            )
            return {"model": self.model_name, "vision": None, "source": "unknown"}

        modalities = [
            str(m).lower()
            for m in (info.get("architecture", {}) or {}).get("input_modalities", [])
        ]
        supports_image = "image" in modalities
        logger.debug("Vision model %s input_modalities=%s",
                     self.model_name, modalities)
        if not supports_image:
            raise VisionError(
                f"{ERR_MODEL_NOT_VISION} "
                f"(OPENROUTER_VISION_MODEL={self.model_name})"
            )
        return {"model": self.model_name, "vision": True,
                "input_modalities": modalities, "source": "catalog"}

    def analyze_image(
        self,
        image_data_url: str,
        prompt: str,
        game_context: Optional[Dict[str, Any]] = None,
        system_prompt: str = "",
    ) -> str:
        if not self.is_available():
            raise VisionError(ERR_NO_VISION_PROVIDER)

        # Never send an image to a model that cannot read one.
        self.verify_vision_support()

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "max_tokens": int(Config.OPENROUTER_VISION_MAX_TOKENS),
        }

        # API key goes in the headers and is never logged.
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.OPENROUTER_SITE_URL,
            "X-Title": Config.OPENROUTER_SITE_NAME,
        }

        logger.info("OpenRouter vision request: model=%s (image omitted from log)",
                    self.model_name)

        try:
            req = urllib.request.Request(
                _CHAT_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:200]
            except Exception:  # pylint: disable=broad-except
                pass
            if exc.code == 404:
                raise VisionError(
                    f"OpenRouter vision model '{self.model_name}' was not found. "
                    f"Set OPENROUTER_VISION_MODEL to a vision-capable model."
                ) from exc
            # Short API detail only: the request body (image data) is never
            # included in an error message.
            suffix = f" ({detail.strip()[:160]})" if detail.strip() else ""
            raise VisionError(
                f"OpenRouter vision request failed (HTTP {exc.code}){suffix}."
            ) from exc
        except urllib.error.URLError as exc:
            raise VisionError("OpenRouter vision request failed: connection error."
                              ) from exc
        except TimeoutError as exc:
            raise VisionError(
                f"OpenRouter vision request timed out after {self.timeout}s."
            ) from exc
        except VisionError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            raise VisionError("OpenRouter vision request failed.") from exc

        return _extract_openrouter_content(raw)

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": Config.OPENROUTER_SITE_URL,
            "X-Title": Config.OPENROUTER_SITE_NAME,
        }


def _extract_openrouter_content(raw: str) -> str:
    """Pull the assistant text out of an OpenRouter response, or raise."""
    from services.ai.openrouter_provider import extract_text, summarize_response

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionError("OpenRouter vision response was not JSON.") from exc

    diagnosis = summarize_response(data)
    logger.debug("OpenRouter vision response metadata: %s", diagnosis)

    if isinstance(data, dict) and data.get("error"):
        err = data.get("error") or {}
        code = err.get("code", "") if isinstance(err, dict) else ""
        raise VisionError(f"OpenRouter vision API error [{code}].")

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise VisionError("OpenRouter vision response had no choices.")

    message = choices[0].get("message", {}) or {}
    content = extract_text(message.get("content"))
    if not content:
        if diagnosis.get("has_refusal"):
            raise VisionError("The vision model refused to analyze the image.")
        raise VisionError(
            "The vision model returned an empty response for this screenshot.")
    return content


# ── Ollama vision ─────────────────────────────────────────────────────────────

class OllamaVisionProvider(VisionProvider):
    """Local multimodal requests via Ollama's native chat format."""

    provider_name = "ollama_vision"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or Config.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model or Config.OLLAMA_VISION_MODEL
        self.timeout = timeout or Config.OLLAMA_VISION_TIMEOUT

    def is_available(self) -> bool:
        return bool(self.base_url and self.model_name and self.model_name.strip())

    def analyze_image(
        self,
        image_data_url: str,
        prompt: str,
        game_context: Optional[Dict[str, Any]] = None,
        system_prompt: str = "",
    ) -> str:
        if not self.is_available():
            raise VisionError(ERR_NO_VISION_PROVIDER)

        # Ollama expects raw base64 in an "images" array on the user message.
        base64_image = _base64_from_data_url(image_data_url)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image],
                },
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }

        url = f"{self.base_url}/api/chat"
        logger.info("Ollama vision request: model=%s (image omitted from log)",
                    self.model_name)

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise VisionError(
                    f"Ollama vision model '{self.model_name}' is not installed. "
                    f"Install it or set OLLAMA_VISION_MODEL."
                ) from exc
            raise VisionError(
                f"Ollama vision request failed (HTTP {exc.code}).") from exc
        except urllib.error.URLError as exc:
            raise VisionError(ERR_OLLAMA_UNAVAILABLE) from exc
        except TimeoutError as exc:
            raise VisionError(
                f"Ollama vision request timed out after {self.timeout}s."
            ) from exc
        except VisionError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            raise VisionError("Ollama vision request failed.") from exc

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VisionError("Ollama vision response was not JSON.") from exc

        message = (envelope.get("message", {}) or {}) if isinstance(envelope, dict) else {}
        content = message.get("content") or (
            envelope.get("response", "") if isinstance(envelope, dict) else ""
        )
        if not content:
            raise VisionError("Ollama vision response had no content.")
        return str(content)


def _base64_from_data_url(data_url: str) -> str:
    """Strip the ``data:...;base64,`` prefix for providers that need raw b64."""
    if "," in data_url and data_url.strip().startswith("data:"):
        return data_url.split(",", 1)[1]
    return data_url


# ── Selection ─────────────────────────────────────────────────────────────────

def select_vision_provider(prefer: Optional[str] = None) -> Optional[VisionProvider]:
    """
    Return the first configured, available vision provider.

    Order follows ``Config.AI_PROVIDER`` semantics: explicit ``prefer``, then
    OpenRouter (cloud), then Ollama (local).  Returns ``None`` when nothing is
    configured — callers then report ``ERR_NO_VISION_PROVIDER``.
    """
    candidates = {
        "openrouter": lambda: OpenRouterVisionProvider(),
        "ollama": lambda: OllamaVisionProvider(),
    }
    order: list = []
    configured = (prefer or "").strip().lower()
    if configured in candidates:
        order.append(configured)
    order += [name for name in ("openrouter", "ollama") if name != configured]

    for name in order:
        try:
            provider = candidates[name]()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Vision provider %s could not be created: %s",
                         name, type(exc).__name__)
            continue
        if provider.is_available():
            logger.debug("Vision provider selected: %s (%s)",
                         provider.provider_name, provider.model_name)
            return provider
    return None