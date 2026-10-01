from flask import Blueprint, render_template, request, jsonify, session
from services.ai.vision import VisionError
from services.game_service import get_all_games, get_game_by_slug
from services.hardware_detector import detect_hardware
from services.optimizer import generate_recommendations
from services.screenshot import ScreenshotError

optimizer_bp = Blueprint('optimizer', __name__)


@optimizer_bp.route('/optimizer')
def optimizer():
    games = get_all_games()
    game_slug = request.args.get('game', '')
    selected_game = get_game_by_slug(game_slug) if game_slug else None
    return render_template('optimizer.html',
                           games=games,
                           selected_game=selected_game)


@optimizer_bp.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.get_json() or {}
    game_slug = data.get('game', '')
    priority = data.get('priority', 'balanced')
    resolution = data.get('resolution', '1920x1080')
    target_fps = int(data.get('target_fps', 60))

    game = get_game_by_slug(game_slug)
    if not game:
        return jsonify({'error': 'Game not found'}), 404

    try:
        hardware = detect_hardware()
        result = generate_recommendations(
            hardware=hardware,
            game=game,
            target_fps=target_fps,
            priority=priority,
            resolution=resolution,
        )
        return jsonify({
            'success': True,
            'result': result.to_dict(),
            'hardware': hardware.to_dict(),
        })
    except Exception as e:
        return jsonify({'error': 'Optimization failed', 'detail': str(e)}), 500


@optimizer_bp.route('/api/ai-optimize', methods=['POST'])
def api_ai_optimize():
    """
    AI-powered optimization endpoint.

    Accepts either JSON (Phase 1 behaviour, unchanged) or multipart form data
    with an optional ``settings_screenshot`` file.

    Form fields:
        game / game_slug   — game slug (required)
        priority           — "fps" | "balanced" | "quality"
        resolution         — e.g. "1920x1080"
        target_fps         — integer
        current_settings   — optional free-text of current in-game settings
        settings_screenshot — optional image file

    When a screenshot is supplied it is analyzed first and its extracted
    settings become the current settings; manual text is kept as supplementary
    context.

    Returns a JSON AIOptimizationResult.
    Always succeeds (falls back to deterministic optimizer if AI unavailable).
    """
    from services.ai_optimizer import optimize_game

    is_json = request.is_json
    if is_json:
        data = request.get_json(silent=True) or {}
        game_slug = data.get('game', '') or data.get('game_slug', '')
        priority = data.get('priority', 'balanced')
        resolution = data.get('resolution', '1920x1080')
        target_fps = int(data.get('target_fps', 60) or 60)
        current_settings = str(data.get('current_settings', ''))
        upload = None
    else:
        form = request.form
        game_slug = form.get('game', '') or form.get('game_slug', '')
        priority = form.get('priority', 'balanced')
        resolution = form.get('resolution', '1920x1080')
        try:
            target_fps = int(form.get('target_fps', 60) or 60)
        except (TypeError, ValueError):
            target_fps = 60
        current_settings = str(form.get('current_settings', '') or '')
        upload = request.files.get('settings_screenshot')

    game = get_game_by_slug(game_slug)
    if not game:
        return jsonify({'error': 'Game not found'}), 404

    screenshot_report = None
    if upload is not None and getattr(upload, 'filename', ''):
        # Screenshot first: it is the user's own image and therefore the most
        # trustworthy description of their current settings.
        from services.ai.screenshot_flow import analyze_screenshot_upload

        try:
            screenshot_report = analyze_screenshot_upload(
                upload, game=game, manual_context=current_settings)
        except ScreenshotError as exc:
            return jsonify({
                'error': exc.message,
                'code': exc.code,
            }), 400

        current_settings = _merge_current_settings(
            screenshot_report.get('current_settings_text', ''), current_settings)

    try:
        hardware = detect_hardware()
        result = optimize_game(
            hardware=hardware,
            game=game,
            target_fps=target_fps,
            priority=priority,
            resolution=resolution,
            current_settings=current_settings,
        )
        payload = {
            'success': True,
            'result': result.to_dict(),
            'hardware': hardware.to_dict(),
        }
        if screenshot_report is not None:
            payload['screenshot'] = screenshot_report
        return jsonify(payload)
    except Exception as e:
        # Belt-and-suspenders: optimize_game() should never raise, but just in case
        return jsonify({'error': 'AI optimization failed', 'detail': str(e)}), 500


def _merge_current_settings(screenshot_text: str, manual_text: str) -> str:
    """
    Screenshot settings first, manual text as supplementary context.

    Both are labelled so Phase 1 can tell the two sources apart; neither is
    mixed with benchmark evidence.
    """
    screenshot_text = (screenshot_text or '').strip()
    manual_text = (manual_text or '').strip()
    parts = []
    if screenshot_text:
        parts.append("Settings detected from the user's screenshot:\n"
                     + screenshot_text)
    if manual_text:
        parts.append(
            "User's own notes about their settings (supplementary):\n"
            + manual_text)
    return "\n\n".join(parts)


@optimizer_bp.route('/api/ai/screenshot-settings', methods=['POST'])
def api_screenshot_settings():
    """
    Analyze a settings screenshot and return the detected settings.

    This is the screenshot step of the Phase 2 pipeline.  It never computes a
    recommendation: the user confirms the detected settings first, then the
    normal ``/api/ai-optimize`` call performs the web research.

    Request: multipart/form-data with ``game`` (or ``game_slug``) and the file
    field ``settings_screenshot``.
    """
    from services.ai.screenshot_flow import analyze_screenshot_upload

    if request.is_json:
        return jsonify({
            'error': 'Please upload a screenshot image file.',
            'code': 'no_file',
        }), 400

    game_slug = (request.form.get('game', '')
                 or request.form.get('game_slug', ''))
    game = get_game_by_slug(game_slug) if game_slug else None
    if game_slug and not game:
        return jsonify({'error': 'Game not found'}), 404

    manual_context = str(request.form.get('current_settings', '') or '')
    upload = request.files.get('settings_screenshot')

    try:
        report = analyze_screenshot_upload(
            upload, game=game, manual_context=manual_context)
    except ScreenshotError as exc:
        return jsonify({'error': exc.message, 'code': exc.code}), 400
    except VisionError as exc:
        return jsonify({'error': str(exc), 'code': 'vision_error'}), 400

    if request.form.get('include_preview', 'true').lower() != 'false':
        report['preview'] = _preview_for(upload)

    return jsonify({'success': True, 'screenshot': report})


def _preview_for(upload):
    """Build the small browser preview data URL (never persisted)."""
    try:
        from services.screenshot import build_preview_data_url, read_upload
        # The upload stream was consumed by the analysis step: rewind it.
        stream = getattr(upload, "stream", None)
        if stream is not None and hasattr(stream, "seek"):
            stream.seek(0)
        return build_preview_data_url(read_upload(upload))
    except ScreenshotError:
        return None
    except Exception:  # pylint: disable=broad-except
        return None


@optimizer_bp.route('/guides')
def guides():
    return render_template('guides.html')


@optimizer_bp.route('/settings')
def settings():
    return render_template('settings.html')


@optimizer_bp.route('/about')
def about():
    return render_template('about.html')
