/**
 * Potato Wiz — Global App JavaScript
 * Handles: nav scroll, mobile menu, image fallbacks
 */

(function () {
  'use strict';

  // ── Navigation scroll effect ─────────────────────────
  const nav = document.querySelector('.site-nav');
  if (nav) {
    const onScroll = () => {
      nav.classList.toggle('scrolled', window.scrollY > 20);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll(); // initial check
  }

  // ── Mobile hamburger menu ─────────────────────────────
  const hamburger = document.getElementById('hamburgerBtn');
  const mobileMenu = document.getElementById('mobileMenu');

  if (hamburger && mobileMenu) {
    hamburger.addEventListener('click', () => {
      const isOpen = mobileMenu.classList.toggle('open');
      hamburger.classList.toggle('open', isOpen);
      hamburger.setAttribute('aria-expanded', isOpen);
      document.body.style.overflow = isOpen ? 'hidden' : '';
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (mobileMenu.classList.contains('open') &&
          !mobileMenu.contains(e.target) &&
          !hamburger.contains(e.target)) {
        mobileMenu.classList.remove('open');
        hamburger.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
        document.body.style.overflow = '';
      }
    });

    // Close on escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && mobileMenu.classList.contains('open')) {
        mobileMenu.classList.remove('open');
        hamburger.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
        document.body.style.overflow = '';
        hamburger.focus();
      }
    });
  }

  // ── Mobile search → navigate to games page ────────────
  const mobileSearch = document.getElementById('mobileSearchInput');
  if (mobileSearch) {
    mobileSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && mobileSearch.value.trim()) {
        window.location.href = `/games?q=${encodeURIComponent(mobileSearch.value.trim())}`;
      }
    });
  }

  // ── Image error fallback ──────────────────────────────
  // Already handled inline with onerror attributes in templates.

  // ── Active nav link highlight ─────────────────────────
  // Already handled server-side with Jinja2 conditionals.

  // ── Lazy image loading fallback ───────────────────────
  if ('IntersectionObserver' in window) {
    document.querySelectorAll('img[loading="lazy"]').forEach(img => {
      img.addEventListener('error', () => {
        // Hide broken image, show sibling placeholder if available
        img.style.display = 'none';
        const placeholder = img.nextElementSibling;
        if (placeholder && (placeholder.classList.contains('game-card-placeholder') ||
                            placeholder.classList.contains('lib-card-placeholder') ||
                            placeholder.classList.contains('panel-game-thumb-placeholder'))) {
          placeholder.style.display = 'flex';
        }
      });
    });
  }

  // ── Reduced motion preference ─────────────────────────
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion) {
    document.documentElement.style.setProperty('--t-fast',   '0ms');
    document.documentElement.style.setProperty('--t-base',   '0ms');
    document.documentElement.style.setProperty('--t-slow',   '0ms');
    document.documentElement.style.setProperty('--t-smooth', '0ms');
  }

})();
