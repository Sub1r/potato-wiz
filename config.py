import os

# Load .env file if python-dotenv is available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on real environment variables


def _env_int(name, default, minimum=None, maximum=None):
    """Read an integer environment variable with a safe fallback.

    An unparseable value falls back to ``default`` instead of crashing at
    import time, and the result is clamped to ``minimum``/``maximum`` when
    those bounds are supplied.
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            value = default
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'potato-wiz-dev-key-2024')
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

    # App metadata
    APP_NAME = 'Potato Wiz'
    APP_TAGLINE = 'Play Smarter, Not Harder.'
    APP_VERSION = '1.1.0'

    # Default optimizer settings
    DEFAULT_TARGET_FPS = 60
    DEFAULT_RESOLUTION = '1920x1080'
    DEFAULT_PRIORITY = 'balanced'

    # ── AI Provider ──────────────────────────────────────────────────────────
    # "auto" | "ollama" | "openrouter"
    AI_PROVIDER = os.environ.get('AI_PROVIDER', 'auto')

    # ── Ollama ───────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')
    OLLAMA_TIMEOUT = int(os.environ.get('OLLAMA_TIMEOUT', '60'))
    OLLAMA_VISION_MODEL = os.environ.get('OLLAMA_VISION_MODEL', '').strip()
    OLLAMA_VISION_TIMEOUT = int(os.environ.get('OLLAMA_VISION_TIMEOUT', '120'))

    # ── OpenRouter ───────────────────────────────────────────────────────────
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
    OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'anthropic/claude-3.5-haiku')
    OPENROUTER_TIMEOUT = int(os.environ.get('OPENROUTER_TIMEOUT', '90'))
    OPENROUTER_SITE_URL = os.environ.get('OPENROUTER_SITE_URL', 'http://localhost:5000')
    OPENROUTER_SITE_NAME = os.environ.get('OPENROUTER_SITE_NAME', 'Potato Wiz')
    OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

    # Cost controls.  OpenRouter rejects requests whose max_tokens exceeds the
    # account credit balance (HTTP 402), so the provider sends a small explicit
    # budget instead of inheriting the model's full 65536-token output window.
    # Bounds are enforced in code as well as here, so a large environment value
    # can never request more than OPENROUTER_MAX_TOKENS_LIMIT.
    OPENROUTER_MAX_TOKENS = _env_int('OPENROUTER_MAX_TOKENS', 2048, minimum=512, maximum=4096)
    OPENROUTER_MAX_TOKENS_FLOOR = 512
    OPENROUTER_MAX_TOKENS_CEILING = 4096

    # Web-research budget (per search / per optimization request).
    OPENROUTER_MAX_SEARCH_RESULTS = _env_int('OPENROUTER_MAX_SEARCH_RESULTS', 3, minimum=1, maximum=10)
    OPENROUTER_MAX_TOTAL_RESULTS = _env_int('OPENROUTER_MAX_TOTAL_RESULTS', 6, minimum=1, maximum=40)
    OPENROUTER_MAX_SEARCH_CALLS = _env_int('OPENROUTER_MAX_SEARCH_CALLS', 2, minimum=1, maximum=8)
    OPENROUTER_MAX_FETCH_PAGES = _env_int('OPENROUTER_MAX_FETCH_PAGES', 1, minimum=0, maximum=4)
    # Top-level server-tool call budget sent to OpenRouter (OpenRouter stops
    # executing server tools after this many calls).
    OPENROUTER_MAX_TOOL_CALLS = _env_int('OPENROUTER_MAX_TOOL_CALLS', 4, minimum=1, maximum=8)
    # openrouter:web_fetch is held back while the server-tool configuration is
    # validated; the first live request carries web_search only.
    OPENROUTER_WEB_FETCH_ENABLED = os.environ.get('OPENROUTER_WEB_FETCH_ENABLED', 'False') == 'True'
    # Consult the OpenRouter model catalog to confirm the configured model
    # supports tool calling before sending server tools.
    OPENROUTER_VERIFY_TOOL_SUPPORT = os.environ.get('OPENROUTER_VERIFY_TOOL_SUPPORT', 'True') == 'True'
    # Optional: 'low' | 'medium' | 'high'. Reasoning models can spend the whole
    # max_tokens budget on reasoning and return no/truncated content, so a low
    # effort keeps the answer inside the budget. Empty = not sent.
    OPENROUTER_REASONING_EFFORT = os.environ.get('OPENROUTER_REASONING_EFFORT', '').strip()
    # Vision model, configured independently of the text model.  Must be a
    # vision-capable model (catalog entry with "image" in input_modalities).
    OPENROUTER_VISION_MODEL = os.environ.get('OPENROUTER_VISION_MODEL', '').strip()
    OPENROUTER_VISION_TIMEOUT = int(os.environ.get('OPENROUTER_VISION_TIMEOUT', '120'))
    OPENROUTER_VISION_MAX_TOKENS = _env_int('OPENROUTER_VISION_MAX_TOKENS', 2048,
                                             minimum=512, maximum=4096)

    # ── Screenshot vision input ───────────────────────────────────────────────
    # Uploads are processed in memory and never stored permanently.
    SCREENSHOT_MAX_UPLOAD_BYTES = _env_int('SCREENSHOT_MAX_UPLOAD_MB', 10,
                                           minimum=1, maximum=20) * 1024 * 1024
    # Longest edge sent to the model.  Settings screenshots are text heavy, so
    # the cap is generous and small images are never upscaled.
    SCREENSHOT_MAX_DIMENSION = _env_int('SCREENSHOT_MAX_DIMENSION', 2000,
                                        minimum=800, maximum=4000)
    SCREENSHOT_JPEG_QUALITY = _env_int('SCREENSHOT_JPEG_QUALITY', 88,
                                       minimum=60, maximum=100)
    SCREENSHOT_ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')
    SCREENSHOT_ALLOWED_MIME_TYPES = ('image/png', 'image/jpeg', 'image/jpg',
                                     'image/webp', 'image/pjpeg')
    SCREENSHOT_INVALID_MESSAGE = (
        'Please upload a valid PNG, JPG, JPEG, or WebP image.'
    )
    SCREENSHOT_TOO_LARGE_MESSAGE = (
        'That image is too large. Please upload an image under 10 MB.'
    )

    @classmethod
    def has_openrouter(cls) -> bool:
        """Return True when an OpenRouter API key is configured."""
        return bool(cls.OPENROUTER_API_KEY and cls.OPENROUTER_API_KEY.strip())

    @classmethod
    def has_ollama_config(cls) -> bool:
        """Return True when Ollama is configured (URL present)."""
        return bool(cls.OLLAMA_BASE_URL and cls.OLLAMA_BASE_URL.strip())
