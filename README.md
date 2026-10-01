# 🥔 Potato Wiz

**Play Smarter, Not Harder.**

A production-quality gaming optimization website that helps PC gamers find the perfect graphics settings for their hardware and games.

---

## Features

- **Hardware Detection** — Automatically reads your GPU, CPU, RAM, and monitor specs (Windows, Linux, macOS)
- **Game Library** — Browse 10+ popular games with full optimization profiles
- **Settings Optimizer** — Generate personalized settings based on your hardware and performance goals
- **FPS Estimation** — See expected framerates before you change any settings
- **Guides** — Performance tips and guides for better PC gaming
- **Full Search** — Live search with keyboard navigation
- **Responsive** — Works on desktop, tablet, and mobile

---

## Tech Stack

| Layer     | Technology |
|-----------|------------|
| Backend   | Python 3 + Flask |
| Templates | Jinja2 |
| Frontend  | HTML5 + CSS3 + Vanilla JS |
| Data      | JSON (data/games.json) |
| Icons     | Lucide Icons (CDN) |
| Font      | Inter (Google Fonts CDN) |
| HW Detection | psutil + PowerShell/subprocess |

---

## Quick Start

### 1. Clone / Navigate to the project

```bash
cd "potato-wiz"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## Project Structure

```
potato-wiz/
│
├── app.py                    # Flask app entry point
├── config.py                 # App configuration
├── requirements.txt          # Python dependencies
│
├── routes/
│   ├── home.py               # Home page
│   ├── games.py              # Games library + game detail + search API
│   ├── optimizer.py          # Optimizer + guides + settings + about
│   └── pc.py                 # My PC + hardware API
│
├── services/
│   ├── hardware_detector.py  # Safe hardware detection
│   ├── optimizer.py          # Recommendation engine
│   ├── fps_estimator.py      # FPS calculation logic
│   └── game_service.py       # Game data loading + search
│
├── models/
│   └── models.py             # HardwareSpecs, OptimizationResult dataclasses
│
├── data/
│   └── games.json            # All game data (10 games)
│
├── templates/
│   ├── base.html             # Base layout + nav + footer
│   ├── home.html             # Landing page
│   ├── games.html            # Game library
│   ├── game_detail.html      # Individual game optimization
│   ├── my_pc.html            # Hardware detection
│   ├── optimizer.html        # Settings optimizer
│   ├── guides.html           # Performance guides
│   ├── settings.html         # User settings
│   ├── about.html            # About page
│   ├── 404.html              # Not found
│   └── 500.html              # Server error
│
├── static/
│   ├── css/
│   │   ├── variables.css     # Design tokens
│   │   ├── base.css          # Base + typography + utility
│   │   ├── navbar.css        # Navigation
│   │   ├── hero.css          # Hero section + panel
│   │   ├── cards.css         # Game cards + features
│   │   ├── optimizer.css     # Optimizer + settings + guides
│   │   └── responsive.css    # Responsive breakpoints
│   │
│   ├── js/
│   │   ├── app.js            # Global app (nav, mobile menu)
│   │   ├── search.js         # Nav search with dropdown
│   │   ├── carousel.js       # Game card carousel
│   │   └── optimizer.js      # Optimizer form + results
│   │
│   └── images/
│       ├── hero/             # Hero background image(s)
│       └── games/            # Game cover images
│
└── tests/
    ├── test_optimizer.py
    ├── test_hardware.py
    └── test_games.py
```

---

## Game Images

The app will work without game images — placeholder colors are shown instead.

To add real game images, place them in `static/images/games/` using these filenames:

| Game | Filename |
|------|----------|
| Palworld | palworld.jpg |
| Elden Ring | elden-ring.jpg |
| Cyberpunk 2077 | cyberpunk-2077.jpg |
| Baldur's Gate 3 | baldurs-gate-3.jpg |
| Hogwarts Legacy | hogwarts-legacy.jpg |
| Red Dead Redemption 2 | rdr2.jpg |
| GTA V | gta-v.jpg |
| Forza Horizon 5 | forza-horizon-5.jpg |
| The Witcher 3 | witcher-3.jpg |
| Starfield | starfield.jpg |

For the hero background, add: `static/images/hero/hero-bg.jpg`

---

## Adding a New Game

Edit `data/games.json` and add a new entry following the existing structure.

Required fields:

```json
{
  "id": 11,
  "name": "Game Name",
  "slug": "game-name",
  "genre": ["Genre1", "Genre2"],
  "description": "...",
  "image": "game-name.jpg",
  "image_placeholder": "#1a3a5c",
  "release_year": 2024,
  "developer": "Studio Name",
  "engine": "Engine Name",
  "popular": false,
  "featured": false,
  "resolutions": ["1920x1080", "2560x1440"],
  "upscaling_support": ["FSR 2.0", "DLSS 3.0"],
  "minimum_specs": { ... },
  "recommended_specs": { ... },
  "fps_profiles": { "high-end": {...}, "mid-high": {...}, ... },
  "recommended_settings": { ... },
  "settings_detail": [ ... ]
}
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Home page |
| GET | `/games` | Game library (with `?q=` search, `?genre=` filter) |
| GET | `/games/<slug>` | Individual game detail |
| GET | `/my-pc` | Hardware detection page |
| GET | `/optimizer` | Settings optimizer |
| GET | `/guides` | Performance guides |
| GET | `/settings` | User preferences |
| GET | `/about` | About page |
| GET | `/api/search?q=` | JSON game search results |
| POST | `/api/optimize` | JSON optimization result |
| GET | `/api/hardware` | JSON hardware detection result |

---

## License

MIT — build something cool.
