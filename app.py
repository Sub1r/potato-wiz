"""
Potato Wiz - Gaming Optimization Website
Run: python app.py
"""

import os
from flask import Flask, render_template, jsonify
from config import Config

# Import blueprints
from routes.home import home_bp
from routes.games import games_bp
from routes.optimizer import optimizer_bp
from routes.pc import pc_bp


def create_app():
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    app.config.from_object(Config)

    # Register blueprints
    app.register_blueprint(home_bp)
    app.register_blueprint(games_bp)
    app.register_blueprint(optimizer_bp)
    app.register_blueprint(pc_bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500

    # Inject global template variables
    @app.context_processor
    def inject_globals():
        return {
            'app_name': Config.APP_NAME,
            'app_tagline': Config.APP_TAGLINE,
            'app_version': Config.APP_VERSION,
        }

    return app


if __name__ == '__main__':
    app = create_app()
    print("=" * 50)
    print("  🥔 Potato Wiz — Gaming Optimization Website")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)
