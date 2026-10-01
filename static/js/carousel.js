/**
 * Potato Wiz — Game Card Carousel
 * Desktop: JS-driven transform carousel
 * Mobile: native scroll
 */

(function () {
  'use strict';

  const track      = document.getElementById('carouselTrack');
  const prevBtn    = document.getElementById('carouselPrev');
  const nextBtn    = document.getElementById('carouselNext');
  const container  = document.querySelector('.carousel-track-container');

  if (!track || !prevBtn || !nextBtn || !container) return;

  const CARD_GAP   = 16; // matches CSS gap
  let currentIndex = 0;

  function getCardWidth() {
    const firstCard = track.querySelector('.game-card');
    if (!firstCard) return 188;
    return firstCard.offsetWidth + CARD_GAP;
  }

  function getVisibleCount() {
    if (!container) return 4;
    return Math.floor(container.offsetWidth / getCardWidth());
  }

  function getTotalCards() {
    return track.querySelectorAll('.game-card').length;
  }

  function getMaxIndex() {
    const visible = getVisibleCount();
    const total   = getTotalCards();
    return Math.max(0, total - visible);
  }

  function updateCarousel(animate = true) {
    if (!animate) {
      track.style.transition = 'none';
    } else {
      track.style.transition = 'transform 380ms cubic-bezier(0.25, 0.46, 0.45, 0.94)';
    }

    const offset = currentIndex * getCardWidth();
    track.style.transform = `translateX(-${offset}px)`;

    // Update button states
    prevBtn.classList.toggle('disabled', currentIndex === 0);
    nextBtn.classList.toggle('disabled', currentIndex >= getMaxIndex());

    prevBtn.setAttribute('aria-disabled', currentIndex === 0);
    nextBtn.setAttribute('aria-disabled', currentIndex >= getMaxIndex());
  }

  prevBtn.addEventListener('click', () => {
    if (currentIndex > 0) {
      currentIndex = Math.max(0, currentIndex - 1);
      updateCarousel();
    }
  });

  nextBtn.addEventListener('click', () => {
    const max = getMaxIndex();
    if (currentIndex < max) {
      currentIndex = Math.min(max, currentIndex + 1);
      updateCarousel();
    }
  });

  // Keyboard support
  prevBtn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      prevBtn.click();
    }
  });
  nextBtn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      nextBtn.click();
    }
  });

  // Swipe support (touch)
  let touchStartX = 0;
  let touchEndX   = 0;

  track.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });

  track.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    const diff = touchStartX - touchEndX;
    if (Math.abs(diff) > 40) {
      if (diff > 0) nextBtn.click();
      else prevBtn.click();
    }
  }, { passive: true });

  // Resize handler
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      currentIndex = Math.min(currentIndex, getMaxIndex());
      updateCarousel(false);
    }, 150);
  });

  // Disable JS carousel on mobile (use native scroll)
  function isMobile() {
    return window.innerWidth < 768;
  }

  function init() {
    if (!isMobile()) {
      updateCarousel(false);
    } else {
      // Reset transform on mobile, let CSS handle scrolling
      track.style.transform = '';
      prevBtn.style.display = 'none';
      nextBtn.style.display = 'none';
    }
  }

  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(init, 150);
  });

  init();

})();
