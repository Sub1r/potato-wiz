import os

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
    APP_VERSION = '1.0.0'

    # Default optimizer settings
    DEFAULT_TARGET_FPS = 60
    DEFAULT_RESOLUTION = '1920x1080'
    DEFAULT_PRIORITY = 'balanced'
