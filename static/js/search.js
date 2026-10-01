/**
 * Potato Wiz — Navigation Search
 * Handles the nav search bar with dropdown results.
 */

(function () {
  'use strict';

  const searchInput    = document.getElementById('navSearchInput');
  const searchDropdown = document.getElementById('searchDropdown');

  if (!searchInput || !searchDropdown) return;

  let debounceTimer = null;
  let currentResults = [];
  let highlightedIndex = -1;

  // ── Open / close dropdown ─────────────────────────────
  function showDropdown() {
    searchDropdown.classList.add('visible');
    searchInput.setAttribute('aria-expanded', 'true');
  }

  function hideDropdown() {
    searchDropdown.classList.remove('visible');
    searchInput.setAttribute('aria-expanded', 'false');
    highlightedIndex = -1;
  }

  // ── Render results ────────────────────────────────────
  function renderResults(results) {
    currentResults = results;
    highlightedIndex = -1;

    if (results.length === 0) {
      searchDropdown.innerHTML = `
        <div class="search-no-results">
          No games found for "<strong>${escHtml(searchInput.value)}</strong>"
        </div>`;
      showDropdown();
      return;
    }

    const html = results.map((game, i) => {
      const genres = Array.isArray(game.genre) ? game.genre.slice(0, 2).join(' · ') : '';
      const imgSrc = `/static/images/games/${game.image || ''}`;
      const bg     = game.image_placeholder || '#1a3a5c';

      return `
        <a href="/games/${game.slug}"
           class="search-result-item"
           role="option"
           id="search-result-${i}"
           data-index="${i}">
          <img
            src="${escHtml(imgSrc)}"
            alt="${escHtml(game.name)}"
            class="search-result-thumb"
            onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"
            loading="lazy"
          >
          <div class="search-result-thumb-placeholder"
               style="background:${escHtml(bg)};display:none;">🎮</div>
          <div class="search-result-info">
            <div class="search-result-name">${escHtml(game.name)}</div>
            <div class="search-result-genre">${escHtml(genres)}</div>
          </div>
        </a>`;
    }).join('');

    searchDropdown.innerHTML = html;
    showDropdown();
  }

  function renderLoading() {
    searchDropdown.innerHTML = `
      <div class="search-loading">
        <div class="spinner" aria-hidden="true"></div>
        Searching…
      </div>`;
    showDropdown();
  }

  // ── Fetch from API ────────────────────────────────────
  async function fetchResults(query) {
    try {
      renderLoading();
      const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      searchDropdown.innerHTML = `
        <div class="search-no-results">Search unavailable. Try again.</div>`;
      showDropdown();
    }
  }

  // ── Input handling ────────────────────────────────────
  searchInput.addEventListener('input', () => {
    const q = searchInput.value.trim();

    clearTimeout(debounceTimer);

    if (!q) {
      hideDropdown();
      return;
    }

    debounceTimer = setTimeout(() => fetchResults(q), 220);
  });

  searchInput.addEventListener('focus', () => {
    if (searchInput.value.trim()) {
      fetchResults(searchInput.value.trim());
    }
  });

  // ── Keyboard navigation ───────────────────────────────
  searchInput.addEventListener('keydown', (e) => {
    const items = searchDropdown.querySelectorAll('.search-result-item');

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      highlightedIndex = Math.min(highlightedIndex + 1, items.length - 1);
      updateHighlight(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      highlightedIndex = Math.max(highlightedIndex - 1, -1);
      updateHighlight(items);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightedIndex >= 0 && items[highlightedIndex]) {
        items[highlightedIndex].click();
      } else if (searchInput.value.trim()) {
        window.location.href = `/games?q=${encodeURIComponent(searchInput.value.trim())}`;
      }
    } else if (e.key === 'Escape') {
      hideDropdown();
      searchInput.blur();
    }
  });

  function updateHighlight(items) {
    items.forEach((item, i) => {
      item.classList.toggle('highlighted', i === highlightedIndex);
    });
  }

  // ── Close on outside click ────────────────────────────
  document.addEventListener('click', (e) => {
    const container = searchInput.closest('.nav-search');
    if (container && !container.contains(e.target)) {
      hideDropdown();
    }
  });

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
