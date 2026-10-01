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



// ─────────────────────────────────────────────────────────────────────────────
// AI Optimizer — appended section
// Handles the "Optimize with AI" button and renders AI results.
// ─────────────────────────────────────────────────────────────────────────────

(function () {
  'use strict';

  const aiBtn              = document.getElementById('aiOptimizeBtn');
  const currentSettingsEl  = document.getElementById('currentSettingsInput');
  const gameSelectEl       = document.getElementById('gameSelect');
  const priorityInputEl    = document.getElementById('priorityInput');
  const resSelectEl        = document.getElementById('resolutionSelect');
  const fpsRangeEl         = document.getElementById('fpsRange');
  const resultsContentEl   = document.getElementById('resultsContent');
  const placeholderEl      = document.getElementById('resultsPlaceholder');
  const loadingEl          = document.getElementById('optimizerLoading');

  if (!aiBtn) return;

  // ── Evidence type labels and colours ────────────────────────────────────
  const EVIDENCE_LABELS = {
    'MEASURED_BENCHMARK':  { label: 'Measured Benchmark', color: '#22c55e' },
    'PUBLISHED_BENCHMARK': { label: 'Published Benchmark', color: '#60a5fa' },
    'COMMUNITY_REPORT':    { label: 'Community Report',    color: '#f59e0b' },
    'OFFICIAL_INFORMATION':{ label: 'Official Info',       color: '#818cf8' },
    'GUIDE':               { label: 'Guide',               color: '#34d399' },
    'AI_INFERENCE':        { label: 'AI Reasoning',        color: '#9ca3af' },
    'FALLBACK_ESTIMATE':   { label: 'Math Estimate',       color: '#6b7280' },
  };

  function evidenceBadge(type) {
    const info = EVIDENCE_LABELS[type] || { label: type, color: '#6b7280' };
    return `<span style="font-size:9px;font-weight:700;letter-spacing:.05em;
                         text-transform:uppercase;padding:2px 6px;border-radius:4px;
                         background:${info.color}22;color:${info.color};
                         border:1px solid ${info.color}44;">${esc(info.label)}</span>`;
  }

  // ── AI status colours ─────────────────────────────────────────────────────
  function statusPill(status) {
    if (status === 'ok') {
      return `<span style="font-size:10px;font-weight:700;color:#22c55e;
                           background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.25);
                           border-radius:4px;padding:2px 8px;">✦ AI Optimized</span>`;
    }
    return `<span style="font-size:10px;font-weight:700;color:#f59e0b;
                         background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.25);
                         border-radius:4px;padding:2px 8px;">⚙ Deterministic</span>`;
  }

  // ── Button click ──────────────────────────────────────────────────────────
  aiBtn.addEventListener('click', async () => {
    const game = gameSelectEl ? gameSelectEl.value : '';
    if (!game) {
      if (gameSelectEl) {
        gameSelectEl.focus();
        gameSelectEl.style.borderColor = 'var(--red)';
        setTimeout(() => { gameSelectEl.style.borderColor = ''; }, 1500);
      }
      return;
    }

    // Show loading
    if (placeholderEl) placeholderEl.style.display = 'none';
    if (loadingEl) {
      loadingEl.classList.add('visible');
      // Update loading message for AI
      const loadingP = loadingEl.querySelector('p');
      if (loadingP) loadingP.textContent = 'AI is researching benchmarks and analyzing your hardware…';
    }
    if (resultsContentEl) resultsContentEl.style.display = 'none';

    const payload = {
      game:             game,
      priority:         priorityInputEl ? priorityInputEl.value : 'balanced',
      resolution:       resSelectEl ? resSelectEl.value : '1920x1080',
      target_fps:       fpsRangeEl ? parseInt(fpsRangeEl.value, 10) : 60,
      current_settings: currentSettingsEl ? currentSettingsEl.value.trim() : '',
    };

    try {
      const res = await fetch('/api/ai-optimize', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      if (loadingEl) loadingEl.classList.remove('visible');

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showAiError(err.error || 'AI optimization failed.');
        return;
      }

      const data = await res.json();
      if (data.success) {
        const gameName = gameSelectEl
          ? (gameSelectEl.options[gameSelectEl.selectedIndex] || {}).text || game
          : game;
        renderAiResults(data.result, data.hardware, gameName);
      } else {
        showAiError('Could not generate AI settings. Please try again.');
      }
    } catch (err) {
      if (loadingEl) loadingEl.classList.remove('visible');
      showAiError('Network error. Please check your connection.');
    }
  });

  // ── Error display ─────────────────────────────────────────────────────────
  function showAiError(msg) {
    if (placeholderEl) placeholderEl.style.display = 'none';
    if (resultsContentEl) {
      resultsContentEl.style.display = 'block';
      resultsContentEl.innerHTML = `
        <div style="background:var(--red-dim,rgba(239,68,68,.08));
                    border:1px solid rgba(255,92,107,.25);
                    border-radius:var(--radius-lg);padding:var(--space-8);text-align:center;">
          <div style="font-size:40px;margin-bottom:var(--space-3);" aria-hidden="true">⚠️</div>
          <h3 style="font-size:var(--text-xl);font-weight:700;color:var(--red);margin-bottom:var(--space-2);">
            AI Optimization Unavailable
          </h3>
          <p style="color:var(--text-muted);">${esc(msg)}</p>
          <p style="color:var(--text-dim);font-size:var(--text-xs);margin-top:var(--space-3);">
            Use "Generate Settings" for the standard rule-based recommendation.
          </p>
        </div>`;
    }
  }

  // ── Main renderer ─────────────────────────────────────────────────────────
  function renderAiResults(result, hardware, gameName) {
    if (!resultsContentEl) return;
    resultsContentEl.style.display = 'block';

    const s = result.recommended_settings || {};
    const fallback = result.fallback || {};

    // ── Hardware chip ──────────────────────────────────────────────────────
    const hwHtml = hardware ? `
      <div style="background:var(--bg-secondary);border:1px solid var(--border);
                  border-radius:var(--radius-md);padding:var(--space-4) var(--space-5);
                  margin-bottom:var(--space-5);display:flex;flex-wrap:wrap;gap:var(--space-5);">
        ${hwChip('GPU', hardware.gpu)}
        ${hwChip('CPU', hardware.cpu)}
        ${hwChip('RAM', hardware.ram_gb + ' GB')}
        ${hwChip('VRAM', hardware.vram_gb + ' GB')}
      </div>` : '';

    // ── Summary ────────────────────────────────────────────────────────────
    const summaryHtml = result.summary ? `
      <div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.2);
                  border-radius:var(--radius-md);padding:var(--space-4) var(--space-5);
                  margin-bottom:var(--space-5);">
        <p style="color:var(--text);font-size:var(--text-sm);line-height:1.6;margin:0;">
          ${esc(result.summary)}
        </p>
      </div>` : '';

    // ── FPS badge ──────────────────────────────────────────────────────────
    const fpsColor = fpsStatusColor(result.estimated_fps);
    const fpsBadgeHtml = result.estimated_fps ? `
      <div style="display:flex;align-items:center;gap:var(--space-3);margin-bottom:var(--space-5);">
        <div style="text-align:center;background:var(--panel);border:1px solid var(--border);
                    border-radius:var(--radius-md);padding:var(--space-3) var(--space-5);">
          <div style="font-size:var(--text-2xl);font-weight:800;color:${fpsColor};">
            ~${esc(result.estimated_fps)}
          </div>
          <div style="font-size:10px;color:var(--text-dim);font-weight:600;text-transform:uppercase;
                      letter-spacing:.06em;">est. FPS</div>
        </div>
        <div style="flex:1;">
          <div style="font-size:var(--text-xs);color:var(--text-dim);margin-bottom:4px;">Source</div>
          ${evidenceBadge(result.fps_source)}
          <div style="font-size:var(--text-xs);color:var(--text-dim);margin-top:4px;">
            Confidence: <strong>${esc(result.confidence || 'low')}</strong>
          </div>
        </div>
      </div>` : '';

    // ── Settings changes table ─────────────────────────────────────────────
    const changesHtml = buildChangesHtml(result.changes || [], s, fallback);

    // ── Reasoning ─────────────────────────────────────────────────────────
    const reasoningHtml = result.reasoning ? `
      <details style="margin-bottom:var(--space-5);">
        <summary style="cursor:pointer;font-size:var(--text-sm);font-weight:700;
                        color:var(--text);padding:var(--space-3) 0;user-select:none;">
          💡 Why these settings?
        </summary>
        <div style="margin-top:var(--space-3);padding:var(--space-4);
                    background:var(--bg-secondary);border-radius:var(--radius-md);
                    font-size:var(--text-xs);color:var(--text-muted);line-height:1.7;">
          ${esc(result.reasoning)}
        </div>
      </details>` : '';

    // ── Evidence / sources ─────────────────────────────────────────────────
    const evidenceHtml = buildEvidenceHtml(result.evidence || []);

    // ── Warnings ──────────────────────────────────────────────────────────
    const warningsHtml = buildWarningsHtml(result.warnings || []);

    resultsContentEl.innerHTML = `
      <div class="results-card">
        <div class="results-card-header">
          <div>
            <div style="font-size:var(--text-xs);color:var(--text-dim);margin-bottom:4px;">
              AI-Optimized for
            </div>
            <div class="results-card-title">${esc(gameName)}</div>
          </div>
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
            ${statusPill(result.status)}
            ${result.provider && result.provider !== 'none'
              ? `<span style="font-size:10px;color:var(--text-dim);">${esc(result.provider)}</span>`
              : ''}
          </div>
        </div>

        <div class="results-card-body">
          ${hwHtml}
          ${summaryHtml}
          ${fpsBadgeHtml}
          ${warningsHtml}
          ${changesHtml}
          ${reasoningHtml}
          ${evidenceHtml}
        </div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:var(--space-3);margin-top:var(--space-4);">
        <span class="badge badge-green">✓ AI Analysis Complete</span>
        <a href="/games/${esc(gameSelectEl ? gameSelectEl.value : '')}"
           class="btn btn-secondary btn-sm">
          View Full Game Details →
        </a>
      </div>`;

    resultsContentEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Settings changes ──────────────────────────────────────────────────────
  function buildChangesHtml(changes, recommended, fallback) {
    if (!changes.length && !Object.keys(recommended).length) return '';

    // Build full settings list from recommended_settings
    const settingRows = [
      { key: 'graphics_preset', label: 'Graphics Preset' },
      { key: 'resolution',      label: 'Resolution'      },
      { key: 'upscaling',       label: 'Upscaling'       },
      { key: 'view_distance',   label: 'View Distance'   },
      { key: 'shadows',         label: 'Shadows'         },
      { key: 'effects',         label: 'Effects'         },
      { key: 'textures',        label: 'Textures'        },
      { key: 'anti_aliasing',   label: 'Anti-Aliasing'   },
      { key: 'motion_blur',     label: 'Motion Blur'     },
      { key: 'vsync',           label: 'V-Sync'          },
    ];

    // Index changes by setting name for quick lookup
    const changeMap = {};
    for (const ch of changes) {
      changeMap[ch.setting.toLowerCase()] = ch;
    }

    let html = `<div style="margin-bottom:var(--space-5);">
      <div style="font-size:var(--text-sm);font-weight:700;margin-bottom:var(--space-3);">
        Recommended Settings
      </div>`;

    for (const row of settingRows) {
      const val = recommended[row.key];
      if (!val) continue;
      const ch = changeMap[row.label.toLowerCase()];
      const hasChange = ch && ch.from && ch.from !== 'Unknown' && ch.from !== ch.to;

      html += `<div class="result-setting-row">
        <div class="result-setting-name">${esc(row.label)}</div>
        <div class="result-setting-right">
          ${hasChange
            ? `<span style="color:var(--text-dim);text-decoration:line-through;
                            font-size:var(--text-xs);margin-right:4px;">${esc(ch.from)}</span>
               <span style="color:var(--green);">→ ${esc(val)}</span>`
            : `<span class="result-setting-val">${esc(String(val))}</span>`}
          ${ch && ch.reason
            ? `<span class="result-setting-why">${esc(ch.reason)}</span>`
            : ''}
        </div>
      </div>`;
    }

    html += `</div>`;
    return html;
  }

  // ── Evidence list ─────────────────────────────────────────────────────────
  function buildEvidenceHtml(evidence) {
    if (!evidence.length) return '';

    const items = evidence.map(e => {
      const link = e.url
        ? `<a href="${esc(e.url)}" target="_blank" rel="noopener noreferrer"
              style="color:var(--blue);font-size:var(--text-xs);word-break:break-all;"
              aria-label="Open source: ${esc(e.title || e.domain)}">${esc(e.title || e.domain || e.url)}</a>`
        : `<span style="font-size:var(--text-xs);color:var(--text-muted);">${esc(e.title || 'Source')}</span>`;

      return `<div style="padding:var(--space-3) 0;border-bottom:1px solid var(--border);">
        <div style="display:flex;align-items:center;gap:var(--space-2);margin-bottom:4px;flex-wrap:wrap;">
          ${evidenceBadge(e.type)}
          ${e.domain ? `<span style="font-size:10px;color:var(--text-dim);">${esc(e.domain)}</span>` : ''}
        </div>
        ${link}
        ${e.claim
          ? `<p style="font-size:var(--text-xs);color:var(--text-muted);margin:4px 0 0;
                       line-height:1.5;">${esc(e.claim)}</p>`
          : ''}
      </div>`;
    }).join('');

    return `<details style="margin-bottom:var(--space-5);" open>
      <summary style="cursor:pointer;font-size:var(--text-sm);font-weight:700;
                      color:var(--text);padding:var(--space-3) 0;user-select:none;">
        🔍 Sources & Evidence (${evidence.length})
      </summary>
      <div style="margin-top:var(--space-2);">${items}</div>
    </details>`;
  }

  // ── Warnings ──────────────────────────────────────────────────────────────
  function buildWarningsHtml(warnings) {
    if (!warnings.length) return '';
    const items = warnings.map(w =>
      `<li style="font-size:var(--text-xs);color:var(--yellow);margin-bottom:4px;">⚠ ${esc(w)}</li>`
    ).join('');
    return `<ul style="list-style:none;padding:0;background:rgba(245,158,11,.06);
                       border:1px solid rgba(245,158,11,.2);border-radius:var(--radius-md);
                       padding:var(--space-3) var(--space-4);margin-bottom:var(--space-5);">
      ${items}
    </ul>`;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function hwChip(label, value) {
    return `<div>
      <div style="font-size:10px;color:var(--text-dim);font-weight:600;
                  text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px;">${esc(label)}</div>
      <div style="font-size:var(--text-sm);font-weight:600;">${esc(String(value || '—'))}</div>
    </div>`;
  }

  function fpsStatusColor(fpsStr) {
    if (!fpsStr) return 'var(--text-dim)';
    const nums = String(fpsStr).match(/\d+/g);
    if (!nums) return 'var(--text-dim)';
    const avg = nums.reduce((a, b) => a + parseInt(b, 10), 0) / nums.length;
    if (avg >= 75) return 'var(--green)';
    if (avg >= 50) return 'var(--green)';
    if (avg >= 35) return 'var(--yellow)';
    return 'var(--red)';
  }

  function esc(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

})();
