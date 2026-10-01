from flask import Blueprint, render_template, jsonify
from services.hardware_detector import detect_hardware

pc_bp = Blueprint('pc', __name__)


@pc_bp.route('/my-pc')
def my_pc():
    detection_error = None
    hardware = None
    try:
        hardware = detect_hardware()
    except Exception as e:
        detection_error = "Hardware detection unavailable on this system."

    return render_template('my_pc.html',
                           hardware=hardware,
                           detection_error=detection_error)


@pc_bp.route('/api/hardware')
def api_hardware():
    try:
        hardware = detect_hardware()
        return jsonify({'success': True, 'hardware': hardware.to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
