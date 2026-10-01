import os

# Load .env file if python-dotenv is available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on real environment variables


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

    # ── OpenRouter ───────────────────────────────────────────────────────────
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
    OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'anthropic/claude-3.5-haiku')
    OPENROUTER_TIMEOUT = int(os.environ.get('OPENROUTER_TIMEOUT', '90'))
    OPENROUTER_SITE_URL = os.environ.get('OPENROUTER_SITE_URL', 'http://localhost:5000')
    OPENROUTER_SITE_NAME = os.environ.get('OPENROUTER_SITE_NAME', 'Potato Wiz')
    OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

    @classmethod
    def has_openrouter(cls) -> bool:
        """Return True when an OpenRouter API key is configured."""
        return bool(cls.OPENROUTER_API_KEY and cls.OPENROUTER_API_KEY.strip())

    @classmethod
    def has_ollama_config(cls) -> bool:
        """Return True when Ollama is configured (URL present)."""
        return bool(cls.OLLAMA_BASE_URL and cls.OLLAMA_BASE_URL.strip())
