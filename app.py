"""
Akilli Gastronomi Rehberi — Streamlit App
Kullanim: streamlit run app.py
"""

from __future__ import annotations

import json
import urllib.parse
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import streamlit as st
from src.search.semantic_search import search
from src.search.query_parser import parse_query

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEMTLER = {"Kadıköy": "kadikoy", "Bostanlı": "bostanli"}
SEMT_REVERSE = {v: k for k, v in SEMTLER.items()}

POPULAR_QUERIES = [
    ("date-mekanlari", "Mum ışığında, sessiz date akşamı", "En iyi date mekanları"),
    ("lahmacuncular", "Mahallenin en iyi lahmacuncusu", "En iyi lahmacuncular"),
    ("meze-keyfi", "Mezesi bol, rakı yanı", "En iyi mezeye sahip mekanlar"),
    ("romantik-balik", "Sahilde rakı-balık keyfi", "Deniz kenarında romantik balık"),
    ("kahvalti", "Çoluk çocuklu pazar kahvaltısı", "Çoluk çocuklu kahvaltı"),
    ("donerci", "Hızlı ve ucuz dönerci", "Hızlı ve ucuz dönerci"),
    ("sac-kavurma", "Sac kavurma & tandır", "Sac kavurma yapan yer"),
    ("manzarali-pizza", "Manzaralı, makul pizza", "Pahalı olmayan manzaralı pizza"),
]

POPULAR_CARD_META = {
    "date-mekanlari":    {"tag": "Romantik",  "sub": "Düşük ışık, çiftlere uygun, rezervasyonlu", "glyph": "♥", "bg": ("oklch(0.52 0.14 30)", "oklch(0.38 0.10 25)")},
    "lahmacuncular":     {"tag": "Yerel",     "sub": "Odun ateşi, taş tezgah, klasik",           "glyph": "◐", "bg": ("oklch(0.50 0.13 28)", "oklch(0.35 0.11 22)")},
    "meze-keyfi":        {"tag": "Meyhane",   "sub": "Mezesi bol, rakı yanı, yaşayan meyhane",   "glyph": "✦", "bg": ("oklch(0.55 0.09 225)", "oklch(0.40 0.08 240)")},
    "romantik-balik":    {"tag": "Manzaralı", "sub": "Deniz kenarı, taze balık, mezeli",          "glyph": "≈", "bg": ("oklch(0.58 0.10 220)", "oklch(0.42 0.09 235)")},
    "kahvalti":          {"tag": "Aile",      "sub": "Geniş alan, oyun köşeli, sakin",            "glyph": "☼", "bg": ("oklch(0.75 0.12 70)", "oklch(0.62 0.13 50)")},
    "donerci":           {"tag": "Hızlı",     "sub": "15 dk altı, ucuz, lezzetli",                "glyph": "↻", "bg": ("oklch(0.62 0.14 38)", "oklch(0.48 0.13 32)")},
    "sac-kavurma":       {"tag": "Et",        "sub": "Et severlere, kanıtlanmış adres",           "glyph": "✱", "bg": ("oklch(0.45 0.08 50)", "oklch(0.32 0.07 45)")},
    "manzarali-pizza":   {"tag": "Manzaralı", "sub": "Roof-top, kalabalık değil, makul fiyat",   "glyph": "△", "bg": ("oklch(0.55 0.11 180)", "oklch(0.40 0.10 195)")},
}

MIN_SCORE = 0.30
MAX_RESULTS = 7


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_image_map(neighborhood: str) -> dict[str, str]:
    raw_file = BASE_DIR / "data" / "raw" / f"{neighborhood}_raw.json"
    if not raw_file.exists():
        return {}
    with open(raw_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        (r.get("placeId") or r.get("title", "")): r.get("imageUrl", "")
        for r in raw
        if r.get("imageUrl")
    }


@st.cache_data
def load_popular_results(neighborhood: str) -> list[dict]:
    popular_file = BASE_DIR / "data" / f"popular_{neighborhood}.json"
    if not popular_file.exists():
        return []
    with open(popular_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
def popular_card_grid(popular_data: list[dict], image_map: dict[str, str], neighborhood: str) -> str:
    """Populer sorgulari 4-sutun grid kart olarak render eder."""
    slug_results: dict[str, list[dict]] = {}
    for section in popular_data:
        slug_results[section["slug"]] = section["results"]

    cards_html = ""
    for slug, title, query_text in POPULAR_QUERIES:
        meta = POPULAR_CARD_META.get(slug, {})
        tag = meta.get("tag", "")
        sub = meta.get("sub", "")
        glyph = meta.get("glyph", "•")
        bg_a, bg_b = meta.get("bg", ("oklch(0.64 0.16 38)", "oklch(0.48 0.13 32)"))

        results = slug_results.get(slug, [])
        venue_count = len(results)
        avg_rating = sum(r.get("rating", 0) for r in results) / max(venue_count, 1)

        # Top result's image as card photo
        img_url = ""
        if results:
            top_pid = results[0].get("_place_id", "")
            img_url = image_map.get(top_pid, "")

        if img_url:
            photo_html = f'<img src="{img_url}" alt="{title}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;">'
        else:
            photo_html = f'''
            <div style="position:absolute;inset:0;background:linear-gradient(135deg,{bg_a},{bg_b});"></div>
            <div class="stripes"></div>
            <div class="placeholder-glyph">{glyph}</div>
            '''

        query_encoded = urllib.parse.quote(query_text)

        cards_html += f'''
        <a href="?q={query_encoded}&semt={neighborhood}" style="text-decoration:none;color:inherit;display:block;">
          <div class="qcard">
            <div class="qcard-photo">
              {photo_html}
              <div class="qcard-tag">{tag}</div>
              <div class="qcard-overlay">
                <span>★ {avg_rating:.1f}</span>
                <span>{venue_count} mekan</span>
              </div>
            </div>
            <div class="qcard-meta">
              <span class="qcard-title">{title}</span>
              <span class="qcard-sub">{sub}</span>
            </div>
          </div>
        </a>
        '''

    return f'''
    <div class="card-grid">
      {cards_html}
    </div>
    '''


def result_cards_grid(results: list[dict], image_map: dict[str, str]) -> str:
    """Restoran sonuc kartlarini 3-sutun grid olarak render eder."""
    cards_html = ""
    for i, r in enumerate(results):
        pid = r.get("_place_id", "")
        img_url = image_map.get(pid, "")
        name = r.get("name", "?")
        rating = r.get("rating", 0)
        score = r.get("score", 0)
        cuisine = r.get("cuisine_style", "-")
        cuisine_title = cuisine.title() if cuisine else "-"
        address = r.get("address", "-")
        price_scale = r.get("price_scale", "-")
        ambiance = r.get("ambiance", "-")
        dishes = r.get("signature_dishes", [])
        best_for = r.get("best_for", [])
        liked = r.get("liked_items", [])
        disliked = r.get("disliked_items", [])
        pf = r.get("physical_features", {})
        # Format physical features as readable Turkish strings
        feature_lines = []
        if pf:
            view = pf.get("view", "")
            if view and "yok" not in str(view).lower():
                feature_lines.append(str(view).capitalize())
            decor = pf.get("decor", "")
            if decor and "yok" not in str(decor).lower():
                feature_lines.append(f"{str(decor).capitalize()} dekora sahip")
            seating = pf.get("seating", "")
            if seating:
                feature_lines.append(f"{str(seating).capitalize()} oturma")
            outdoor = pf.get("outdoor", "")
            if outdoor and "yok" not in str(outdoor).lower():
                feature_lines.append("Açık hava bölümü var" if outdoor == "var" else f"Açık hava: {outdoor}")
            parking = pf.get("parking", "")
            if parking:
                if parking in ("var", "otopark"):
                    feature_lines.append("Otoparkı var")
                elif parking == "yok":
                    feature_lines.append("Otoparkı yok")
                elif parking == "vale":
                    feature_lines.append("Vale park")
                else:
                    feature_lines.append(f"Park: {parking}")
            max_grp = pf.get("max_group_size")
            if max_grp and str(max_grp).isdigit():
                feature_lines.append(f"En fazla {max_grp} kişilik gruplar")
        features_str = " · ".join(feature_lines) if feature_lines else ""
        is_fallback = r.get("_is_fallback", False)
        fallback_note = r.get("_fallback_note", "")
        failed_kws = r.get("_failed_keywords", [])
        review_evidence = r.get("_review_evidence", [])
        leftover_kws = r.get("_leftover_keywords", [])

        if img_url:
            photo_html = f'<img src="{img_url}" alt="{name}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;">'
        else:
            photo_html = f'<div style="position:absolute;inset:0;background:linear-gradient(135deg,oklch(0.55 0.05 80),oklch(0.40 0.04 70));display:grid;place-items:center;font-family:var(--serif);font-style:italic;color:rgba(255,255,255,0.7);font-size:48px;">🍽️</div>'

        # Dishes / liked
        dishes_str = ", ".join(dishes[:5]) if dishes else ""
        liked_str = ", ".join(
            li["name"] if isinstance(li, dict) else li for li in liked[:5]
        ) if liked else ""
        best_for_str = ", ".join(best_for) if best_for else ""

        cards_html += f'''
        <div class="rcard" id="rcard-{i}">
          <div class="rcard-photo">
            {photo_html}
            <div class="rcard-badge">★ {rating}</div>
          </div>
          <div class="rcard-body">
            <h3><span>{name}</span><span class="rcard-score">AGR {score:.2f}</span></h3>
            <p class="rcard-cuisine">{cuisine_title}</p>
            <p class="rcard-address">{address}</p>
            <p class="rcard-price">{ambiance} · <strong>{price_scale}</strong></p>
            {f'<p class="rcard-dishes"><strong>İmza yemekler:</strong> {dishes_str}</p>' if dishes_str else ""}
            {f'<p class="rcard-liked"><strong>Beğenilen lezzetler:</strong> {liked_str}</p>' if liked_str else ""}
            {f'<p class="rcard-bestfor"><strong>Neye gidilir:</strong> {best_for_str}</p>' if best_for_str else ""}
            {f'<p class="rcard-fallback">⚠️ {fallback_note}<br>🔍 Aranan ama bulunamayan: <b>{", ".join(failed_kws)}</b></p>' if is_fallback else ""}
            {f'<details class="rcard-details"><summary>📝 Yorum kanıtı ({", ".join(leftover_kws)})</summary>{"<br>".join("…"+s.strip()+"…" for s in review_evidence[:3])}</details>' if review_evidence and leftover_kws else ""}
            {f'<details class="rcard-details"><summary>👎 Şikayetler</summary>{"<br>".join("• "+ (di["name"]+" — "+di.get("reason","")) if isinstance(di, dict) else "• "+di for di in disliked[:5])}</details>' if disliked else ""}
            {f'<p class="rcard-features">{features_str}</p>' if features_str else ""}
          </div>
        </div>
        '''

    return f'<div class="results-grid">{cards_html}</div>'


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="akıllı·sofra",
    page_icon="",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS + Fonts
# ---------------------------------------------------------------------------
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ---- Design Tokens ---- */
:root {
  --bg: oklch(0.985 0.008 75);
  --bg-2: oklch(0.965 0.012 75);
  --ink: oklch(0.22 0.012 60);
  --ink-2: oklch(0.42 0.012 60);
  --ink-3: oklch(0.62 0.012 60);
  --line: oklch(0.90 0.012 75);
  --line-strong: oklch(0.82 0.014 75);
  --accent: #E26B41;
  --accent-ink: oklch(0.98 0.012 75);
  --accent-soft: oklch(0.93 0.04 38);
  --serif: "Instrument Serif", "Times New Roman", serif;
  --sans: "Geist", ui-sans-serif, system-ui, sans-serif;
  --mono: "Geist Mono", ui-monospace, monospace;
}

/* ---- Streamlit overrides ---- */
.stApp {
  background: var(--bg);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit default header/footer chrome */
.stMainBlockContainer {
  max-width: 1280px !important;
  padding: 0 32px !important;
}

header[data-testid="stHeader"] { display: none; }
#MainMenu { display: none; }
footer { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stStatusWidget"] { display: none; }

/* Remove default Streamlit element margins */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

/* ---- TopBar ---- */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 20px 0;
  margin-bottom: 8px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: inherit;
}
.brand-mark {
  width: 30px; height: 30px;
  border-radius: 50%;
  background: var(--ink);
  color: var(--bg);
  display: grid; place-items: center;
  font-family: var(--serif);
  font-style: italic;
  font-size: 19px;
  line-height: 1;
  padding-bottom: 2px;
}
.brand-name {
  font-family: var(--serif);
  font-size: 22px;
  letter-spacing: -0.01em;
  line-height: 1;
}
.brand-name em {
  font-style: italic;
  color: var(--accent);
}
.nav-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}
.nav-link {
  padding: 10px 14px;
  font-size: 14px;
  color: var(--ink-2);
  border-radius: 999px;
  text-decoration: none;
  transition: background .15s ease, color .15s ease;
}
.nav-link:hover { background: var(--bg-2); color: var(--ink); }
.nav-pill {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 6px 6px 14px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: white;
  font-size: 14px;
  color: var(--ink);
  cursor: pointer;
  transition: box-shadow .15s ease, border-color .15s ease;
}
.nav-pill:hover { box-shadow: 0 2px 10px oklch(0.22 0.012 60 / 0.08); border-color: var(--line-strong); }
.nav-pill .avatar {
  width: 26px; height: 26px;
  border-radius: 50%;
  background: linear-gradient(135deg, oklch(0.68 0.10 50), oklch(0.55 0.14 30));
  display: grid; place-items: center;
  color: white;
  font-size: 12px;
  font-weight: 600;
}

/* ---- Hero ---- */
.hero {
  padding: 40px 0 24px;
}
.eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin-bottom: 18px;
}
.eyebrow::before {
  content: "";
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
}
.hero-title {
  font-family: var(--serif);
  font-weight: 400;
  font-size: clamp(48px, 6.4vw, 84px);
  line-height: 0.98;
  letter-spacing: -0.02em;
  margin: 0 0 14px;
  color: var(--ink);
  max-width: 1000px;
}
.hero-title em {
  font-style: italic;
  color: var(--accent);
}
.hero-sub {
  font-size: 17px;
  line-height: 1.5;
  color: var(--ink-2);
  max-width: 620px;
  margin: 0 0 28px;
}

/* ---- Search Row ---- */
.search-row {
  display: flex;
  align-items: stretch;
  background: white;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 8px;
  gap: 0;
  box-shadow: 0 1px 2px oklch(0.22 0.012 60 / 0.04), 0 8px 28px oklch(0.22 0.012 60 / 0.06);
  max-width: 980px;
  transition: box-shadow .2s ease, border-color .2s ease;
}
.search-row:focus-within {
  border-color: var(--ink);
  box-shadow: 0 1px 2px oklch(0.22 0.012 60 / 0.04), 0 12px 36px oklch(0.22 0.012 60 / 0.10);
}
.search-field {
  flex: 1;
  display: flex; flex-direction: column;
  padding: 10px 18px;
  min-width: 0;
}
.search-label {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin-bottom: 2px;
}
.search-input {
  border: none;
  outline: none;
  background: transparent;
  font-family: inherit;
  font-size: 16px;
  color: var(--ink);
  width: 100%;
  padding: 0;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
}
.search-input::placeholder { color: var(--ink-3); }
.search-divider {
  width: 1px;
  background: var(--line);
  margin: 6px 0;
  align-self: stretch;
}
.search-btn {
  align-self: stretch;
  display: flex; align-items: center; gap: 8px;
  padding: 0 22px;
  border-radius: 12px;
  background: var(--accent);
  color: var(--accent-ink);
  border: none;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.005em;
  cursor: pointer;
  transition: filter .15s ease, transform .05s ease;
  white-space: nowrap;
}
.search-btn:hover { filter: brightness(1.05); }
.search-btn:active { transform: translateY(1px); }

/* ---- Example chips ---- */
.examples {
  margin-top: 18px;
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px;
  font-size: 13.5px;
  color: var(--ink-2);
  max-width: 980px;
}
.examples-label {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.chip {
  padding: 6px 12px;
  border-radius: 999px;
  background: white;
  border: 1px solid var(--line);
  font-size: 13px;
  color: var(--ink-2);
  cursor: pointer;
  text-decoration: none;
  transition: border-color .15s ease, color .15s ease, background .15s ease;
  font-family: var(--sans);
}
.chip:hover { border-color: var(--ink); color: var(--ink); }

/* ---- Popular Section ---- */
.section {
  padding: 56px 0 32px;
}
.section-head {
  display: flex; align-items: flex-end; justify-content: space-between;
  margin-bottom: 28px;
  gap: 24px;
}
.section-title {
  font-family: var(--serif);
  font-weight: 400;
  font-size: clamp(28px, 3vw, 38px);
  letter-spacing: -0.015em;
  line-height: 1.05;
  margin: 0;
  max-width: 720px;
}
.section-title em { font-style: italic; color: var(--accent); }
.section-sub {
  font-size: 14px;
  color: var(--ink-3);
  margin: 8px 0 0;
}

/* ---- Card Grid ---- */
.card-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 20px 18px;
}
@media (max-width: 1100px) { .card-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 800px)  { .card-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 520px)  { .card-grid { grid-template-columns: 1fr; } }

/* ---- QCard ---- */
.qcard {
  cursor: pointer;
  display: flex; flex-direction: column;
  gap: 12px;
  background: transparent;
  border: none;
  padding: 0;
  text-align: left;
  transition: transform .2s ease;
}
.qcard:hover { transform: translateY(-2px); }
.qcard:hover .qcard-photo .qcard-overlay { opacity: 1; }
.qcard-photo {
  position: relative;
  aspect-ratio: 4 / 5;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--line);
}
.stripes {
  position: absolute; inset: 0;
  background-image: repeating-linear-gradient(
    135deg,
    transparent 0,
    transparent 14px,
    rgba(255,255,255,0.18) 14px,
    rgba(255,255,255,0.18) 15px
  );
}
.placeholder-glyph {
  position: absolute; inset: 0;
  display: grid; place-items: center;
  font-family: var(--serif);
  font-style: italic;
  color: rgba(255,255,255,0.85);
  font-size: 56px;
  line-height: 1;
  letter-spacing: -0.01em;
}
.qcard-overlay {
  position: absolute; left: 0; right: 0; bottom: 0;
  padding: 12px 14px;
  background: linear-gradient(to top, rgba(0,0,0,0.45), transparent);
  color: white;
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.85;
  transition: opacity .2s ease;
  display: flex; justify-content: space-between; align-items: center;
}
.qcard-tag {
  position: absolute;
  top: 12px; left: 12px;
  padding: 5px 9px;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(8px);
  border-radius: 6px;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink);
}
.qcard-meta { display: flex; flex-direction: column; gap: 4px; padding: 0 2px; }
.qcard-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--ink);
  line-height: 1.3;
  letter-spacing: -0.005em;
}
.qcard-sub {
  font-size: 13px;
  color: var(--ink-3);
  line-height: 1.35;
}

/* ---- Results View ---- */
.results-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 0 20px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 24px;
}
.results-summary {
  font-family: var(--serif);
  font-size: 22px;
  letter-spacing: -0.01em;
}
.results-summary em { font-style: italic; color: var(--accent); }
.results-meta { font-size: 13px; color: var(--ink-3); }
.back-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 14px;
  border-radius: 999px;
  background: white;
  border: 1px solid var(--line);
  font-size: 13px;
  color: var(--ink);
  cursor: pointer;
  text-decoration: none;
  font-family: var(--sans);
}
.back-btn:hover { border-color: var(--ink); }

/* ---- AI Summary ---- */
.ai-summary {
  background: white;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 22px 24px;
  margin-bottom: 32px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 18px;
  align-items: start;
}
.ai-summary .ai-icon {
  width: 36px; height: 36px;
  border-radius: 50%;
  background: var(--ink);
  color: var(--bg);
  display: grid; place-items: center;
  font-family: var(--serif);
  font-style: italic;
  font-size: 20px;
  line-height: 1;
}
.ai-summary .ai-label {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin-bottom: 6px;
}
.ai-summary p {
  margin: 0;
  font-size: 15px;
  line-height: 1.55;
  color: var(--ink);
  max-width: 720px;
}
.ai-summary mark {
  background: var(--accent-soft);
  color: var(--ink);
  padding: 1px 4px;
  border-radius: 4px;
}

/* ---- Results Grid ---- */
.results-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px 20px;
}
@media (max-width: 1000px) { .results-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 640px)  { .results-grid { grid-template-columns: 1fr; } }

/* ---- RCard (restaurant result card) ---- */
.rcard {
  display: flex; flex-direction: column;
  gap: 12px;
  background: white;
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  transition: transform .2s ease, box-shadow .2s ease;
}
.rcard:hover { transform: translateY(-2px); box-shadow: 0 8px 28px oklch(0.22 0.012 60 / 0.08); }
.rcard-photo {
  position: relative;
  aspect-ratio: 4 / 3;
  overflow: hidden;
}
.rcard-badge {
  position: absolute; top: 12px; left: 12px;
  padding: 5px 10px;
  background: white;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  color: var(--ink);
  box-shadow: 0 1px 3px oklch(0.22 0.012 60 / 0.12);
  font-family: var(--sans);
}
.rcard-body {
  padding: 0 14px 14px;
  display: flex; flex-direction: column; gap: 6px;
}
.rcard-body h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--ink);
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 10px;
}
.rcard-score {
  font-size: 12px;
  font-family: var(--mono);
  color: var(--accent);
  flex-shrink: 0;
}
.rcard-cuisine {
  margin: 0;
  font-size: 13px;
  color: var(--ink-2);
}
.rcard-address {
  margin: 0;
  font-size: 12px;
  color: var(--ink-3);
}
.rcard-price {
  margin: 0;
  font-size: 13px;
  color: var(--ink);
}
.rcard-price strong { font-weight: 600; }
.rcard-note {
  margin: 4px 0 0;
  font-size: 12.5px;
  color: var(--ink-2);
  line-height: 1.45;
  font-style: italic;
}
.rcard-dishes, .rcard-liked, .rcard-bestfor, .rcard-features {
  margin: 2px 0 0;
  font-size: 12px;
  color: var(--ink-2);
}
.rcard-fallback {
  margin: 4px 0 0;
  font-size: 11.5px;
  color: var(--accent);
  background: var(--accent-soft);
  padding: 8px 10px;
  border-radius: 8px;
}
.rcard-details {
  margin-top: 4px;
  font-size: 11.5px;
  color: var(--ink-3);
}
.rcard-details summary {
  cursor: pointer;
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--ink-3);
  padding: 4px 0;
}
.rcard-details summary:hover { color: var(--ink-2); }
.rcard-details p {
  margin: 4px 0 0;
  font-size: 11px;
  color: var(--ink-3);
  line-height: 1.5;
}

/* ---- Trust Strip ---- */
.trust {
  margin: 32px 0 0;
  padding: 28px 0;
  border-top: 1px solid var(--line);
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}
@media (max-width: 780px) { .trust { grid-template-columns: 1fr 1fr; } }
.trust-item .n {
  font-family: var(--serif);
  font-size: 28px;
  letter-spacing: -0.015em;
  line-height: 1;
  margin-bottom: 6px;
}
.trust-item .n em { font-style: italic; color: var(--accent); }
.trust-item .l {
  font-size: 12.5px;
  color: var(--ink-3);
  line-height: 1.4;
}

/* ---- Footer ---- */
.app-footer {
  margin: 24px 0 0;
  padding: 22px 0 44px;
  border-top: 1px solid var(--line);
  display: flex; justify-content: space-between; align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 12.5px;
  color: var(--ink-3);
}
.footer-links { display: flex; gap: 18px; flex-wrap: wrap; }
.footer-links a { color: var(--ink-3); text-decoration: none; }
.footer-links a:hover { color: var(--ink); }

/* ---- Query parse expander restyle ---- */
section[data-testid="stExpander"] {
  background: white;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  margin-bottom: 8px !important;
}

/* ---- Streamlit widget overrides ---- */
div.stSpinner { text-align: center; padding: 24px 0; }
div[data-testid="stNotification"] { display: none; }

/* Smooth animations */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
.anim { animation: fadeUp .35s ease both; }
</style>
""")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "neighborhood" not in st.session_state:
    st.session_state.neighborhood = "kadikoy"
if "parsed_query" not in st.session_state:
    st.session_state.parsed_query = None
if "_url_processed" not in st.session_state:
    st.session_state._url_processed = ""

# ---------------------------------------------------------------------------
# URL query param handling (for custom HTML search bar)
# ---------------------------------------------------------------------------
try:
    url_q = st.query_params.get("q", "")
    url_semt = st.query_params.get("semt", "")
except Exception:
    url_q = ""
    url_semt = ""

if url_q and url_q != st.session_state._url_processed:
    st.session_state._url_processed = url_q
    st.session_state.query = url_q
    if url_semt and url_semt in SEMT_REVERSE:
        st.session_state.neighborhood = url_semt

    with st.spinner(f"🔍 **{url_q}** için {SEMT_REVERSE.get(st.session_state.neighborhood, st.session_state.neighborhood)} restoranları taranıyor..."):
        parsed = parse_query(url_q)
        st.session_state.parsed_query = parsed
        results = search(
            url_q,
            top_k=30,
            neighborhood=st.session_state.neighborhood,
            parsed=parsed,
        )
        st.session_state.results = results

# ---------------------------------------------------------------------------
# TopBar
# ---------------------------------------------------------------------------
st.html("""
<div class="topbar">
  <a href="/" class="brand" aria-label="akilli sofra">
    <span class="brand-mark">a</span>
    <span class="brand-name">akıllı<em>·</em>sofra</span>
  </a>
  <div class="nav-actions">
    <a class="nav-link" href="#">Rehber</a>
    <a class="nav-link" href="#">Mekan ekle</a>
    <a class="nav-link" href="#">Yardım</a>
    <button class="nav-pill" type="button" aria-label="Hesap">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M2 4h10M2 7h10M2 10h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      </svg>
      <span class="avatar">M</span>
    </button>
  </div>
</div>
""")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
current_neighborhood = st.session_state.neighborhood
current_semt_name = SEMT_REVERSE.get(current_neighborhood, "Kadıköy")
current_query = st.session_state.query

today_str = datetime.now().strftime("%d %B")
st.html(f"""
<div class="hero">
  <div class="eyebrow">{current_semt_name} · Beta · {today_str}</div>
  <h1 class="hero-title">
    İstediğini söyle, <em>en uygun mekanı</em> bulalım.
  </h1>
  <p class="hero-sub">
    Yıldız sayısına değil; ruh haline, fiyatına, yani isteklerine göre öneri. İsteğini yaz, semtini seç, gerisini bize bırak.
  </p>
</div>
""")

# ---------------------------------------------------------------------------
# Search bar (custom HTML form → URL query params)
# ---------------------------------------------------------------------------
selected_kadikoy = "selected" if current_neighborhood == "kadikoy" else ""
selected_bostanli = "selected" if current_neighborhood == "bostanli" else ""

st.html(f"""
<form id="search-form" method="GET" action="/" style="max-width:980px;">
  <div class="search-row">
    <div class="search-field" style="flex:1">
      <span class="search-label">Ne arıyorsun?</span>
      <input name="q" class="search-input" type="text"
             placeholder="Örn: sahilde manzaralı rakı-balık akşamı…"
             value="{html_escape(current_query)}"
             autocomplete="off">
    </div>
    <div class="search-divider"></div>
    <div class="search-field" style="min-width:180px;">
      <span class="search-label">Semt</span>
      <select name="semt" class="search-input" style="cursor:pointer;">
        <option value="kadikoy" {selected_kadikoy}>Kadıköy</option>
        <option value="bostanli" {selected_bostanli}>Bostanlı</option>
      </select>
    </div>
    <button class="search-btn" type="submit" aria-label="Ara">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.6" />
        <path d="M11 11l3 3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <span>Ara</span>
    </button>
  </div>
</form>

<div class="examples">
  <span class="examples-label">Şunları deneyebilirsin →</span>
  <a class="chip" href="?q={urllib.parse.quote('deniz kenarında romantik balık')}&semt={current_neighborhood}">deniz kenarında romantik balık</a>
  <a class="chip" href="?q={urllib.parse.quote('çoluk çocuklu pazar kahvaltısı')}&semt={current_neighborhood}">çoluk çocuklu pazar kahvaltısı</a>
  <a class="chip" href="?q={urllib.parse.quote('hızlı ve ucuz dönerci')}&semt={current_neighborhood}">hızlı ve ucuz dönerci</a>
  <a class="chip" href="?q={urllib.parse.quote('geç saatte açık mekan')}&semt={current_neighborhood}">geç saatte açık mekan</a>
</div>
""")

# ---------------------------------------------------------------------------
# Results view
# ---------------------------------------------------------------------------
all_results = st.session_state.results
results = [r for r in all_results if r["score"] >= MIN_SCORE][:MAX_RESULTS]
has_results = bool(results) and bool(st.session_state._url_processed)

if has_results:
    st.html('<div class="results">')

    # Results head
    st.html(f"""
    <div class="results-head">
      <div>
        <div class="results-summary">
          "<em>{st.session_state.query}</em>" için {len(results)} mekan · {current_semt_name}
        </div>
        <div class="results-meta">AI değerlendirmesi · son güncelleme: birkaç dakika önce</div>
      </div>
      <a href="/" class="back-btn">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M9 3l-4 4 4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Aramayı değiştir
      </a>
    </div>
    """)

    # Restaurant result cards
    image_map = load_image_map(st.session_state.neighborhood)
    cards_html = result_cards_grid(results, image_map)
    st.html(cards_html)

    st.html('</div>')  # close .results

elif st.session_state._url_processed and not results:
    # No results found
    st.html(f"""
    <div class="results">
      <div class="results-head">
        <div>
          <div class="results-summary">"<em>{st.session_state.query}</em>" · {current_semt_name}</div>
        </div>
        <a href="/" class="back-btn">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M9 3l-4 4 4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Aramayı değiştir
        </a>
      </div>
      <p style="color:var(--ink-2);font-size:17px;padding:24px 0;">
        Bu kriterlere uygun mekan bulunamadı. Farklı bir arama deneyin.
      </p>
    </div>
    """)

# ---------------------------------------------------------------------------
# Popular section (only show when no active search)
# ---------------------------------------------------------------------------
if not st.session_state._url_processed:
    st.html("""
    <div class="section">
      <div class="section-head">
        <div>
          <h2 class="section-title">
            İnsanların en çok <em>sorduğu</em> şeyler
          </h2>
          <p class="section-sub">Sade cümlelerle yazılmış gerçek aramalar — birine tıkla, sonuçları gör.</p>
        </div>
      </div>
    </div>
    """)

    popular_data = load_popular_results(current_neighborhood)
    image_map = load_image_map(current_neighborhood)

    if not popular_data:
        st.caption("Popüler sonuçlar henüz oluşturulmadı. `python -m src.search.generate_popular all` çalıştırın.")
    else:
        grid_html = popular_card_grid(popular_data, image_map, current_neighborhood)
        st.html(grid_html)

# ---------------------------------------------------------------------------
# Trust strip + Footer
# ---------------------------------------------------------------------------
st.html(f"""
<div class="trust">
  <div class="trust-item">
    <div class="n">2.412<em>+</em></div>
    <div class="l">{current_semt_name} ve çevresinde listelenen mekan</div>
  </div>
  <div class="trust-item">
    <div class="n">186<em>K</em></div>
    <div class="l">Yorum analiz edildi</div>
  </div>
  <div class="trust-item">
    <div class="n">37<em>sn</em></div>
    <div class="l">Ortalama yanıt süresi</div>
  </div>
  <div class="trust-item">
    <div class="n">94<em>%</em></div>
    <div class="l">Kullanıcı önerisini beğendi</div>
  </div>
</div>

<footer class="app-footer">
  <div>© 2026 akıllı·sofra — Bağımsız gastronomi rehberi</div>
  <nav class="footer-links">
    <a href="#">Hakkında</a>
    <a href="#">SSS</a>
    <a href="#">İletişim</a>
    <a href="#">KVKK</a>
    <a href="#">Mekan sahipleri</a>
  </nav>
</footer>
""")
