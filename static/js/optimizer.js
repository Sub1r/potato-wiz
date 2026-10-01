/**
 * Potato Wiz — Settings Optimizer JS
 * Handles: priority toggle, FPS slider, form submission, results rendering
 */

(function () {
  'use strict';

  // ── Elements ──────────────────────────────────────────
  const form            = document.getElementById('optimizerForm');
  const gameSelect      = document.getElementById('gameSelect');
  const priorityInput   = document.getElementById('priorityInput');
  const priorityBtns    = document.querySelectorAll('.priority-btn');
  const fpsRange        = document.getElementById('fpsRange');
  const fpsDisplay      = document.getElementById('fpsDisplay');
  const resSelect       = document.getElementById('resolutionSelect');
  const resultsContent  = document.getElementById('resultsContent');
  const placeholder     = document.getElementById('resultsPlaceholder');
  const loadingEl       = document.getElementById('optimizerLoading');

  if (!form) return;

  // ── FPS slider ────────────────────────────────────────
  if (fpsRange && fpsDisplay) {
    fpsRange.addEventListener('input', () => {
      fpsDisplay.textContent = fpsRange.value;
      fpsRange.setAttribute('aria-valuenow', fpsRange.value);
      fpsRange.setAttribute('aria-valuetext', fpsRange.value + ' fps');
    });
  }

  // ── Priority buttons ──────────────────────────────────
  priorityBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      priorityBtns.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      priorityInput.value = btn.dataset.value;
    });
  });

  // ── Pre-select game from URL param ────────────────────
  const urlParams = new URLSearchParams(window.location.search);
  const gameParam = urlParams.get('game');
  if (gameParam && gameSelect) {
    gameSelect.value = gameParam;
  }

  // Load settings from localStorage defaults
  try {
    const stored = JSON.parse(localStorage.getItem('potatoWizSettings') || '{}');
    if (stored.defaultFps && fpsRange) {
      fpsRange.value = stored.defaultFps;
      fpsDisplay.textContent = stored.defaultFps;
    }
    if (stored.defaultPriority && priorityInput) {
      priorityInput.value = stored.defaultPriority;
      priorityBtns.forEach(btn => {
        const isActive = btn.dataset.value === stored.defaultPriority;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-pressed', isActive);
      });
    }
    if (stored.defaultResolution && resSelect) {
      resSelect.value = stored.defaultResolution;
    }
  } catch(e) {}

  // ── Form submit ───────────────────────────────────────
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const game = gameSelect.value;
    if (!game) {
      gameSelect.focus();
      gameSelect.style.borderColor = 'var(--red)';
      setTimeout(() => { gameSelect.style.borderColor = ''; }, 1500);
      return;
    }

    const payload = {
      game:       game,
      priority:   priorityInput.value,
      resolution: resSelect ? resSelect.value : '1920x1080',
      target_fps: fpsRange ? parseInt(fpsRange.value) : 60,
    };

    showLoading();

    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        showError(err.error || 'Optimization failed.');
        return;
      }

      const data = await res.json();
      if (data.success) {
        renderResults(data.result, data.hardware, gameSelect.options[gameSelect.selectedIndex].text);
      } else {
        showError('Could not generate settings. Please try again.');
      }
    } catch (err) {
      showError('Network error. Please check your connection and try again.');
    }
  });

  // ── UI states ─────────────────────────────────────────
  function showLoading() {
    placeholder.style.display = 'none';
    loadingEl.classList.add('visible');
    resultsContent.style.display = 'none';
  }

  function hideLoading() {
    loadingEl.classList.remove('visible');
  }

  function showError(msg) {
    hideLoading();
    placeholder.style.display = 'none';
    resultsContent.style.display = 'block';
    resultsContent.innerHTML = `
      <div style="background:var(--red-dim);border:1px solid rgba(255,92,107,0.25);
                  border-radius:var(--radius-lg);padding:var(--space-8);text-align:center;">
        <div style="font-size:40px;margin-bottom:var(--space-3);" aria-hidden="true">⚠️</div>
        <h3 style="font-size:var(--text-xl);font-weight:700;color:var(--red);margin-bottom:var(--space-2);">
          Optimization Failed
        </h3>
        <p style="color:var(--text-muted);">${escHtml(msg)}</p>
      </div>`;
  }

  // ── Results renderer ──────────────────────────────────
  function renderResults(result, hardware, gameName) {
    hideLoading();
    placeholder.style.display = 'none';
    resultsContent.style.display = 'block';

    const statusColor = {
      'Excellent': 'var(--green)',
      'Very Good': 'var(--green)',
      'Playable':  'var(--green)',
      'Low':       'var(--yellow)',
      'Unplayable':'var(--red)',
    }[result.status] || 'var(--yellow)';

    const settingsListHtml = (result.settings_list || []).map(s => `
      <div class="result-setting-row">
        <div class="result-setting-name">${escHtml(s.name)}</div>
        <div class="result-setting-right">
          <span class="result-setting-val">${escHtml(String(s.value))}</span>
          ${s.why ? `<span class="result-setting-why">${escHtml(s.why)}</span>` : ''}
        </div>
      </div>`).join('');

    const hwHtml = hardware ? `
      <div style="background:var(--bg-secondary);border:1px solid var(--border);
                  border-radius:var(--radius-md);padding:var(--space-4) var(--space-5);
                  margin-bottom:var(--space-5);display:flex;flex-wrap:wrap;gap:var(--space-5);">
        <div>
          <div style="font-size:10px;color:var(--text-dim);font-weight:600;
                      text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px;">GPU</div>
          <div style="font-size:var(--text-sm);font-weight:600;">${escHtml(hardware.gpu)}</div>
        </div>
        <div>
          <div style="font-size:10px;color:var(--text-dim);font-weight:600;
                      text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px;">RAM</div>
          <div style="font-size:var(--text-sm);font-weight:600;">${hardware.ram_gb} GB</div>
        </div>
        <div>
          <div style="font-size:10px;color:var(--text-dim);font-weight:600;
                      text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px;">VRAM</div>
          <div style="font-size:var(--text-sm);font-weight:600;">${hardware.vram_gb} GB</div>
        </div>
      </div>` : '';

    resultsContent.innerHTML = `
      <!-- Header card -->
      <div class="results-card">
        <div class="results-card-header">
          <div>
            <div style="font-size:var(--text-xs);color:var(--text-dim);margin-bottom:4px;">
              Optimized for
            </div>
            <div class="results-card-title">${escHtml(gameName)}</div>
          </div>
          <div class="results-fps-badge">
            <div class="results-fps-number" style="color:${statusColor};">
              ~${escHtml(result.estimated_fps)}
            </div>
            <div class="results-fps-label">est. FPS</div>
          </div>
        </div>

        <div class="results-card-body">
          ${hwHtml}

          <!-- Notes -->
          ${result.notes ? `
          <div class="results-notes" style="margin-bottom:var(--space-5);" role="note">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="2" class="results-notes-icon"
                 aria-hidden="true">
              <circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>
            </svg>
            <p class="results-notes-text">${escHtml(result.notes)}</p>
          </div>` : ''}

          <!-- Settings list -->
          <div class="results-settings-list" role="list">
            ${settingsListHtml}
          </div>
        </div>
      </div>

      <!-- Status badge row -->
      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:var(--space-3);">
        <span class="badge badge-green">✓ ${escHtml(result.status)}</span>
        <a href="/games/${escHtml(gameSelect.value)}" class="btn btn-secondary btn-sm">
          View Full Game Details →
        </a>
      </div>`;

    // Scroll into view smoothly
    resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Escape HTML helper ────────────────────────────────
  function escHtml(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

})();
