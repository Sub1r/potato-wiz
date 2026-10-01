"""
verify_routes.py
Quick route verification script.
Run: python verify_routes.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from app import create_app

app = create_app()
client = app.test_client()

routes = [
    ('GET', '/'),
    ('GET', '/games'),
    ('GET', '/games/palworld'),
    ('GET', '/games/elden-ring'),
    ('GET', '/games/cyberpunk-2077'),
    ('GET', '/my-pc'),
    ('GET', '/optimizer'),
    ('GET', '/guides'),
    ('GET', '/settings'),
    ('GET', '/about'),
    ('GET', '/api/search?q=palworld'),
    ('GET', '/api/hardware'),
]

print("=" * 60)
print("  Potato Wiz — Route Verification")
print("=" * 60)

all_ok = True
for method, path in routes:
    r = client.get(path)
    ok = r.status_code == 200
    status = 'OK' if ok else f'FAIL ({r.status_code})'
    if not ok:
        all_ok = False
    print(f"  {method:4} {path:40} {status}")

# Test POST optimize
payload = {'game': 'palworld', 'priority': 'balanced', 'resolution': '1920x1080', 'target_fps': 60}
r = client.post('/api/optimize',
                data=json.dumps(payload),
                content_type='application/json')
if r.status_code == 200:
    data = r.get_json()
    preset = data['result']['preset']
    fps    = data['result']['estimated_fps']
    print(f"  POST /api/optimize                        OK | preset={preset}, fps={fps}")
else:
    all_ok = False
    print(f"  POST /api/optimize                        FAIL ({r.status_code})")

# Test 404 handler
r = client.get('/nonexistent-route-xyz-404')
status = 'OK (404)' if r.status_code == 404 else f'UNEXPECTED ({r.status_code})'
print(f"  GET  /nonexistent                         {status}")

print("=" * 60)
if all_ok:
    print("  ALL ROUTES OK")
else:
    print("  SOME ROUTES FAILED")
print("=" * 60)
