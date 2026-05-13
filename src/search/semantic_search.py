"""
Semantik arama: Dogal dil sorgusunu embedding'e cevirir,
cosine similarity ile en uygun restoranlari bulur.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Kategori yakinlik katsayilari: sorgu terimi → (kategori_terimi → boost)
# 1.30 = birebir eslesme, 1.15 = yakin komsu, 1.00 = notr,
# 0.85 = uzak akraba, 0.70 = alakasiz
CATEGORY_PROXIMITY = {
    "kafe": {
        "kafe": 1.30, "kahve": 1.30, "coffee": 1.30, "kafeterya": 1.30,
        "kahvaltı": 1.15, "brunch": 1.15, "pastane": 1.15, "çikolata": 1.15,
        "tatlı": 1.10, "dondurma": 1.10, "şekerleme": 1.10,
        "restoran": 1.00, "pizza": 0.85, "bar": 0.85,
        "kebap": 0.70, "döner": 0.70, "balık": 0.70, "meyhane": 0.70,
        "çorbacı": 0.70, "fast food": 0.85, "tantuni": 0.70, "çiğ köfte": 0.70,
    },
    "kahvaltı": {
        "kahvaltı": 1.30, "brunch": 1.30, "breakfast": 1.30,
        "kafe": 1.15, "kahve": 1.15, "pastane": 1.15, "coffee": 1.15,
        "restoran": 0.90, "tatlı": 0.90,
        "kebap": 0.70, "döner": 0.70, "balık": 0.70, "meyhane": 0.70,
        "pizza": 0.70, "çorbacı": 0.85,
    },
    "meyhane": {
        "meyhane": 1.30, "içki": 1.30,
        "balık": 1.15, "deniz mahsülleri": 1.15, "fish": 1.15,
        "restoran": 1.00, "kebap": 0.90, "ocakbaşı": 0.90,
        "kahvaltı": 0.70, "kafe": 0.70, "kahve": 0.70, "döner": 0.70,
    },
    "balık": {
        "balık": 1.30, "deniz mahsülleri": 1.30, "fish": 1.30,
        "meyhane": 1.15, "restoran": 1.00,
        "kebap": 0.70, "döner": 0.70, "kahvaltı": 0.70, "kafe": 0.70,
        "pizza": 0.70, "tatlı": 0.70,
    },
    "lahmacun": {
        "pideci": 1.30, "lahmacun": 1.30,
        "kebap": 1.15, "ocakbaşı": 1.15, "restoran": 1.00,
        "çorbacı": 1.00, "pide": 1.30,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "meyhane": 0.70,
    },
    "pide": {
        "pideci": 1.30, "pide": 1.30, "lahmacun": 1.30,
        "kebap": 1.15, "restoran": 1.00, "çorbacı": 1.00,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "meyhane": 0.70,
    },
    "kebap": {
        "kebap": 1.30, "ocakbaşı": 1.30, "kebapçı": 1.30,
        "pideci": 1.15, "restoran": 1.00, "dönerci": 1.10,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "meyhane": 0.85,
    },
    "döner": {
        "dönerci": 1.30, "döner": 1.30,
        "fast food": 1.10, "restoran": 1.00, "kebap": 1.15,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "meyhane": 0.70,
    },
    "pizza": {
        "pizza": 1.30, "italyan": 1.30,
        "restoran": 1.00, "fast food": 1.10,
        "kahvaltı": 0.70, "balık": 0.70, "meyhane": 0.70, "döner": 0.70,
    },
    "tatlı": {
        "tatlı": 1.30, "dondurma": 1.30, "çikolata": 1.30,
        "şekerleme": 1.30, "pastane": 1.30,
        "kafe": 1.15, "kahve": 1.15, "coffee": 1.15,
        "restoran": 0.90, "kahvaltı": 1.10,
        "kebap": 0.70, "balık": 0.70, "meyhane": 0.70, "döner": 0.70,
    },
    "köfte": {
        "köfte": 1.30, "restoran": 1.00, "kebap": 1.10, "fast food": 1.05,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "tatlı": 0.70,
    },
    "çiğ köfte": {
        "çiğ köfte": 1.30, "çiğköfte": 1.30,
        "restoran": 0.90, "fast food": 1.05,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "meyhane": 0.70,
    },
    "tantuni": {
        "tantuni": 1.30, "restoran": 1.00, "dürüm": 1.10,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "tatlı": 0.70,
    },
    "burger": {
        "hamburger": 1.30, "burger": 1.30, "cheeseburger": 1.30,
        "fast food": 1.10, "restoran": 1.00,
        "kahvaltı": 0.70, "kafe": 0.70, "balık": 0.70, "meyhane": 0.70,
    },
}


def category_boost(query: str, result: dict) -> float:
    """Sorgudaki terimin restoran kategorisine yakinligina gore boost.
    1.30 = birebir, 1.15 = yakin, 1.00 = notr, 0.85 = uzak, 0.70 = alakasiz."""
    query_lower = query.lower()
    categories = " ".join(result.get("categories", [])).lower()
    dishes = " ".join(result.get("signature_dishes", [])).lower()
    name = result.get("name", "").lower()

    search_pool = f"{categories} {dishes} {name}"

    best_boost = 1.0
    matched = False

    for keyword, proximity_map in CATEGORY_PROXIMITY.items():
        if keyword in query_lower:
            matched = True
            for cat_term, boost_value in proximity_map.items():
                if cat_term in search_pool:
                    best_boost = max(best_boost, boost_value)

    return best_boost if matched else 1.0


def rating_confidence_boost(rating: float, review_count: int) -> float:
    """Bayesian rating guveni: az yorumlu yuksek puanlari dengeler.
    review_count=50 → guven=0.5, review_count=200 → guven=0.8"""
    confidence = min(review_count / 100, 1.0)
    # 4.0 uzeri her 0.1 puan icin %2, en fazla %20 bonus
    rating_bonus = max(0, (rating - 4.0)) * 0.2
    return 1.0 + rating_bonus * confidence


def mention_boost(query: str, result: dict) -> float:
    """Query'de gecen terimler liked_items/disliked_items icinde kac kisi
    tarafindan bahsedilmis? Ne kadar cok kisi soylemis = o kadar guvenilir."""
    query_lower = query.lower()
    liked = result.get("liked_items", [])
    disliked = result.get("disliked_items", [])

    max_mentions = 0
    for item in liked + disliked:
        name = item["name"].lower() if isinstance(item, dict) else str(item).lower()
        count = item.get("mention_count", 0) if isinstance(item, dict) else 0
        if name in query_lower or any(
            kw in name for kw in query_lower.split() if len(kw) > 4
        ):
            max_mentions = max(max_mentions, count)

    if max_mentions == 0:
        return 1.0
    # 0 mention → 1.0, 10 mention → 1.12, 20 mention → 1.15 (cap)
    return 1.0 + min(max_mentions / 100, 0.15)


def price_boost(query: str, result: dict) -> float:
    """Sorguda 'ucuz/uygun/butce/ekonomik' gecince fiyati dusuk olana boost.
    Price scale: ₺=0, ₺₺=1, ₺₺₺=2, ₺₺₺₺=3 → daha dusuk = daha yuksek boost."""
    query_lower = query.lower()
    # "uygun" tek basina "musait" anlamina da gelir → sadece fiyat baglaminda tetikle
    cheap_triggers = ["ucuz", "bütçe", "ekonomik", "hesaplı"]
    price_context = ["uygun fiyat", "fiyatı uygun", "uygun fiyatlı", "fiyatlar uygun"]
    if not any(t in query_lower for t in cheap_triggers) and not any(p in query_lower for p in price_context):
        return 1.0

    scale = result.get("price_scale", "")
    price_perception = result.get("price_perception", "").lower()

    # ₺ sayisina gore boost: ne kadar az ₺ o kadar iyi
    tl_count = scale.count("₺")
    if tl_count == 0:
        tl_count = 1  # bilinmiyorsa orta varsay

    # ₺ = 1.15, ₺₺ = 1.05, ₺₺₺ = 0.95, ₺₺₺₺ = 0.85
    boost_map = {1: 1.15, 2: 1.05, 3: 0.95, 4: 0.85}
    base = boost_map.get(tl_count, 1.0)

    # price_perception "ucuz" veya "uygun" ise ekstra
    if price_perception in ("ucuz", "uygun"):
        base += 0.05

    return base


def group_size_boost(query: str, result: dict) -> float:
    """Sorguda 'kalabalik/buyuk grup' gecince max_group_size buyuk olana boost."""
    query_lower = query.lower()
    group_triggers = ["kalabalık", "büyük grup", "kalabalik grup",
                      "geniş grup", "grup yemek", "toplantı"]
    if not any(t in query_lower for t in group_triggers):
        return 1.0

    pf = result.get("physical_features", {})
    if not pf.get("group_friendly"):
        return 0.85  # grup dostu degilse penalty

    max_size = pf.get("max_group_size", 8)
    # 8 ve alti → 1.0, 10-12 → 1.05, 15+ → 1.10, 30+ → 1.15
    if max_size >= 30:
        return 1.15
    elif max_size >= 15:
        return 1.10
    elif max_size >= 10:
        return 1.05
    return 1.0


# Hard filter: binary (var/yok) fiziksel ozellikler.
# Sorguda trigger terim gecerse → o field'in degeri listede yoksa restoran elenir.
HARD_FILTERS = {
    "view": {
        "deniz kenarı": ["deniz manzaralı"],
        "deniz manzaralı": ["deniz manzaralı"],
        "sahil": ["deniz manzaralı"],
        "manzaralı": ["deniz manzaralı", "bahçe manzaralı", "şehir manzaralı", "park manzaralı"],
        "bahçe manzara": ["bahçe manzaralı"],
        "park manzara": ["park manzaralı"],
        "şehir manzara": ["şehir manzaralı"],
    },
    "outdoor": {
        "açık alan": ["var"],
        "bahçede": ["var"],
        "açık hava": ["var"],
        "teras": ["var"],
    },
    "parking": {
        "otopark": ["var", "otopark", "vale"],
        "park yeri": ["var", "otopark", "sokak", "vale"],
        "vale": ["var", "vale", "otopark"],
    },
}


def apply_hard_filters(query: str, physical_features: dict) -> bool:
    """Sorgudaki tetikleyicilere gore restorani filtreler.
    False → elenir, True → gecer."""
    query_lower = query.lower()
    for field, triggers in HARD_FILTERS.items():
        for term, allowed in triggers.items():
            if term in query_lower:
                value = physical_features.get(field, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                if str(value).lower() not in [str(a).lower() for a in allowed]:
                    return False
    return True


def parse_negative_exclusions(query: str) -> tuple[list[str], list[str]]:
    """Sorguda 'X olmayan', 'X olmasin', 'X haric', 'Xsiz/sız' gibi dislama kaliplarini bulur.
    Returns: (excluded_terms, food_contexts) — food_contexts dislanan terimin
    hangi yemek turuyle sinirli oldugunu belirtir (burger, pide vs).
    Context yoksa → genel dislama (restoran adi, kategorileri, mutfagi dahil her yerde aranir)."""
    import re

    query_lower = query.lower()
    excluded = []
    contexts = []

    # Baglac ve anlamsiz kelimeler — context olarak kullanilmaz
    NOISE_CONTEXT = {"ama", "lakin", "fakat", "ancak", "ve", "veya", "ya", "da", "de",
                     "bir", "bu", "su", "şu", "o", "ise", "icin", "için", "gibi",
                     "daha", "en", "cok", "çok", "biraz", "olsun", "yerler", "yer"}

    # "X olmayan" / "X olmasin" / "X olmasın" kaliplari
    for m in re.finditer(r"(\w+)\s+(olmayan|olmasın|olmasin|degil|değil)", query_lower):
        term = m.group(1)
        if term not in NOISE_CONTEXT:
            excluded.append(term)
            rest = query_lower[m.end():].strip().split()
            ctx = ""
            if rest:
                w = rest[0]
                if w not in NOISE_CONTEXT and len(w) >= 3:
                    ctx = w
            contexts.append(ctx)

    # "X haric" / "X dışında" / "X disinda" kalibi
    for m in re.finditer(r"(\w+)\s+(haric|hariç|dışında|dısında|disinda)", query_lower):
        term = m.group(1)
        if term not in NOISE_CONTEXT and term not in excluded:
            excluded.append(term)
            contexts.append("")  # genel dislama

    # "X istemiyorum" / "X istemem" kalibi
    for m in re.finditer(r"(\w+)\s+(istemiyorum|istemem|istemez|istemiyoruz)", query_lower):
        term = m.group(1)
        if term not in NOISE_CONTEXT and term not in excluded:
            excluded.append(term)
            contexts.append("")

    # "Xsiz/sız/suz/süz" kalibi
    for word in query_lower.split():
        if word.endswith(("siz", "sız", "suz", "süz")) and len(word) > 4:
            stem = word[:-3]
            if stem not in excluded:
                excluded.append(stem)
                contexts.append("")

    # Sadece suffix bazli dislamalar (sogansiz gibi) icin bos context'i doldur.
    # "X olmayan" kalibinda context zaten belirlendi (bos veya dolu), elleme.
    for i, (exc_term, ctx) in enumerate(zip(excluded, contexts)):
        if ctx != "":
            continue
        # Bu dislama suffix kaynakli mi kontrol et: sorguda "exc_term + siz/sız" var mi?
        for sfx in ["siz", "sız", "suz", "süz"]:
            if exc_term + sfx in query_lower:
                for kw in CATEGORY_PROXIMITY:
                    if kw in query_lower and kw not in excluded:
                        contexts[i] = kw
                        break
                break

    return excluded, contexts


def _fuzzy_match_term(term: str, text: str) -> bool:
    """Terimin metinde tam veya kok haliyle gecip gecmedigini kontrol eder.
    'kebapçı' → 'kebap' kokunu de arar (Türkçe meslek/yer ekleri icin)."""
    term_lower = term.lower()
    if term_lower in text:
        return True
    # Turkce meslek/yer eklerini kirp: -çı -ci -cu -cü -çu -hane -evi
    for suffix in ["çı", "ci", "cu", "cü", "çu", "hane", "evi", "cisi", "çısı"]:
        if term_lower.endswith(suffix):
            stem = term_lower[:-len(suffix)]
            if len(stem) >= 4 and stem in text:
                return True
    return False


def negative_filter(excluded_terms: list[str], food_contexts: list[str], result: dict) -> bool:
    """Dislanacak terimleri restoranin tum alanlarinda kontrol eder.
    food_context varsa → sadece context'le ayni item'da gecen dislama gecerli.
    food_context yoksa → genel dislama: ad, kategori, mutfak stili, yemekler dahil aranir."""
    if not excluded_terms:
        return True

    all_items = (
        [(d, "sig") for d in result.get("signature_dishes", [])]
        + [(li["name"] if isinstance(li, dict) else li, "lik") for li in result.get("liked_items", [])]
        + [(di["name"] if isinstance(di, dict) else di, "dis") for di in result.get("disliked_items", [])]
    )
    name_lower = result.get("name", "").lower()
    categories_str = " ".join(result.get("categories", [])).lower()
    cuisine_lower = result.get("cuisine_style", "").lower()

    for i, exclude_term in enumerate(excluded_terms):
        ctx = food_contexts[i] if i < len(food_contexts) else ""
        ctx = ctx.lower()

        found_violation = False
        if ctx:
            for item_name, _source in all_items:
                item_lower = item_name.lower()
                if _fuzzy_match_term(ctx, item_lower) and _fuzzy_match_term(exclude_term, item_lower):
                    found_violation = True
                    break
        else:
            if _fuzzy_match_term(exclude_term, name_lower):
                found_violation = True
            elif _fuzzy_match_term(exclude_term, categories_str):
                found_violation = True
            elif _fuzzy_match_term(exclude_term, cuisine_lower):
                found_violation = True
            else:
                for item_name, _source in all_items:
                    if _fuzzy_match_term(exclude_term, item_name.lower()):
                        found_violation = True
                        break

        if found_violation:
            return False

    return True


def _llm_negative_filter(excluded_terms: list[str], exclude_categories: list[str], result: dict) -> bool:
    """LLM tarafindan cikarilan dislama terimlerini ve kategorileri kontrol eder.
    Daha basit ve direkt: terim -> ad, kategori, mutfak, yemekler. Kategori -> kategoriler."""
    if not excluded_terms and not exclude_categories:
        return True

    name_lower = result.get("name", "").lower()
    categories_str = " ".join(result.get("categories", [])).lower()
    cuisine_lower = result.get("cuisine_style", "").lower()

    # Kategori dislamasi
    for exc_cat in exclude_categories:
        if exc_cat.lower() in categories_str:
            return False

    # Terim dislamasi
    all_items = (
        [d.lower() for d in result.get("signature_dishes", [])]
        + [(li["name"] if isinstance(li, dict) else li).lower() for li in result.get("liked_items", [])]
        + [(di["name"] if isinstance(di, dict) else di).lower() for di in result.get("disliked_items", [])]
    )
    for exc_term in excluded_terms:
        term_lower = exc_term.lower()
        if _fuzzy_match_term(term_lower, name_lower):
            return False
        if _fuzzy_match_term(term_lower, categories_str):
            return False
        if _fuzzy_match_term(term_lower, cuisine_lower):
            return False
        for item in all_items:
            if _fuzzy_match_term(term_lower, item):
                return False

    return True


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _contains_word(text: str, word: str) -> bool:
    """Kelimenin metin icinde tam kelime olarak gecmesini kontrol eder.
    \\b yerine (?<!\\w)...(?!\\w) kullanir cunku \\b Turkce karakterlerde tutarsiz.
    'sac' → 'sac kavurma'✓, 'kisacasi'✗, 'Saca Marin'✗"""
    import re
    pattern = r"(?<!\w)" + re.escape(word) + r"(?!\w)"
    return bool(re.search(pattern, text))


def keyword_match_boost(
    leftover_keywords: list[str],
    scored: list[dict],
    neighborhood: str,
) -> dict[str, float]:
    """Leftover keyword'lerin ham yorumlarda gecme sayisina gore boost carpani.
    Tum keyword'ler ayni review'da gecmelidir (false positive azaltma).
    Returns: {pid_or_name: boost_carpani} — 1.0 notr, 0.85 hic eslesme yok, ~2.5 guclu eslesme."""
    if not leftover_keywords:
        return {}

    raw_file = BASE_DIR / "data" / "raw" / f"{neighborhood}_raw.json"
    if not raw_file.exists():
        return {}

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # pid → review texts
    reviews_by_place: dict[str, list[str]] = {}
    for r in raw_data:
        pid = r.get("placeId") or r.get("title", "")
        all_reviews = r.get("reviews") or r.get("reviewsData") or []
        texts = []
        for rev in all_reviews:
            t = (rev.get("text") or rev.get("reviewText") or "").strip()
            if t:
                texts.append(t)
        if texts:
            reviews_by_place[pid] = texts

    kws = [kw.lower() for kw in leftover_keywords]
    # Multi-word keyword'leri ayir: "sac kavurma" → hem butun hem ["sac","kavurma"]
    # olarak dene. Butun phrase eslesmezse kelimeleri tek tek dene (daha dusuk boost).
    kws_split = []
    for kw in kws:
        words = kw.split()
        kws_split.append((kw, words))
    # Tum tekil kelimeler (multi-word'lerin parcalari dahil)
    all_single_words = list(set(w for _, words in kws_split for w in words))

    boosts: dict[str, float] = {}

    for r in scored:
        pid = r.get("_place_id", "")
        name = r.get("name", "")

        reviews = reviews_by_place.get(pid, [])
        if not reviews:
            for rpid, rreviews in reviews_by_place.items():
                if name and name in rpid:
                    reviews = rreviews
                    break

        match_count = 0
        for review in reviews:
            review_lower = review.lower()
            # Once tum keyword'leri butun phrase olarak dene
            phrase_match = all(_contains_word(review_lower, kw) for kw in kws)
            if phrase_match:
                match_count += 1
            elif len(kws) == 1 and len(kws_split[0][1]) > 1:
                # Tek bir multi-word phrase var ve butun olarak eslesmedi.
                # Kelimelere bolup dene: tum kelimeler ayni yorumda var mi?
                if all(_contains_word(review_lower, w) for w in kws_split[0][1]):
                    match_count += 0.5  # yarim say (daha dusuk guven)

        if match_count > 0:
            boosts[pid or name] = 1.0 + min(match_count * 0.35, 1.5)
        else:
            boosts[pid or name] = 0.85

    return boosts


def _find_dead_keywords(
    leftover_keywords: list[str],
    neighborhood: str,
) -> list[str]:
    """Hicbir restoranin hicbir yorumunda gecmeyen 'olu' kelimeleri doner.
    Bunlar veride olmadigi icin sorgudan cikarilip fallback yapilabilir."""
    if not leftover_keywords:
        return []

    raw_file = BASE_DIR / "data" / "raw" / f"{neighborhood}_raw.json"
    if not raw_file.exists():
        return []

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dead = []
    for kw in leftover_keywords:
        found = False
        for r in raw_data:
            reviews = r.get("reviews") or r.get("reviewsData") or []
            for rev in reviews:
                text = (rev.get("text") or rev.get("reviewText") or "").lower()
                if _contains_word(text, kw):
                    found = True
                    break
            if found:
                break
        if not found:
            dead.append(kw)

    return dead


def _count_keyword_frequency(
    leftover_keywords: list[str],
    scored: list[dict],
) -> dict[str, int]:
    """Her leftover keyword'un kac restoranin yorumlarinda gectigini sayar.
    En sik gecen keyword'u belirlemek icin kullanilir."""
    freq: dict[str, int] = {}
    for kw in leftover_keywords:
        freq[kw] = 0
        kw_lower = kw.lower()
        for r in scored:
            # Hizli kontrol: restoranin identity card'indaki text'lerde var mi?
            # (Tam yorum taramasi yavas olur, burada scored icindeki
            # liked/disliked items ve signature dishes ile yetinelim)
            all_text = " ".join([
                r.get("name", ""),
                " ".join(r.get("signature_dishes", [])),
                " ".join(li["name"] if isinstance(li, dict) else li for li in r.get("liked_items", [])),
                " ".join(di["name"] if isinstance(di, dict) else di for di in r.get("disliked_items", [])),
            ]).lower()
            if _contains_word(all_text, kw_lower):
                freq[kw] += 1
    return freq


def _score_restaurants(
    query: str,
    query_emb: list[float],
    embeddings_data: list[dict],
    identity_cards: dict[str, dict],
    filter_query: str,
    excluded_terms: list[str],
    food_contexts: list[str],
    exclude_categories: list[str] | None = None,
    use_llm_filter: bool = False,
) -> list[dict]:
    """Restoranlari verilen query embedding'e gore skorla.
    use_llm_filter=True ise _llm_negative_filter kullanir (LLM'den gelen yapili dislama).
    Degilse regex-tabanli negative_filter kullanir (fallback)."""
    scored = []
    for item in embeddings_data:
        sim = cosine_similarity(query_emb, item["embedding"])
        card = identity_cards.get(item["place_id"], {})
        ic = card.get("identity_card", {})
        pf = ic.get("physical_features", {})

        if not apply_hard_filters(filter_query, pf):
            continue

        result_preview = {
            "name": item["name"],
            "signature_dishes": ic.get("signature_dishes", []),
            "liked_items": ic.get("liked_items", []),
            "disliked_items": ic.get("disliked_items", []),
            "categories": card.get("categories", []),
            "cuisine_style": ic.get("cuisine_style", ""),
        }

        if use_llm_filter:
            if not _llm_negative_filter(excluded_terms, exclude_categories or [], result_preview):
                continue
        else:
            if not negative_filter(excluded_terms, food_contexts, result_preview):
                continue

        result = {
            "name": item["name"],
            "score": round(sim, 4),
            "_place_id": card.get("place_id", item.get("place_id", "")),
            "address": card.get("address", ""),
            "rating": card.get("rating", 0),
            "categories": card.get("categories", []),
            "price_range": card.get("price_range", ""),
            "price_scale": ic.get("price_scale", ""),
            "price_perception": ic.get("price_perception", ""),
            "signature_dishes": ic.get("signature_dishes", []),
            "liked_items": ic.get("liked_items", []),
            "disliked_items": ic.get("disliked_items", []),
            "ambiance": ic.get("ambiance", ""),
            "best_for": ic.get("best_for", []),
            "cuisine_style": ic.get("cuisine_style", ""),
            "physical_features": pf,
            "vibe_summary": ic.get("vibe_summary", ""),
            "_review_count": card.get("reviews_count", 0),
        }

        cat_b = category_boost(query, result)
        rat_b = rating_confidence_boost(
            card.get("rating", 0), card.get("reviews_count", 0)
        )
        men_b = mention_boost(query, result)
        pri_b = price_boost(query, result)
        grp_b = group_size_boost(query, result)

        result["raw_semantic"] = round(sim, 4)
        result["score"] = round(sim * cat_b * rat_b * men_b * pri_b * grp_b, 4)
        result["_cat_boost"] = cat_b
        result["_rat_boost"] = rat_b
        result["_men_boost"] = men_b
        result["_pri_boost"] = pri_b
        result["_grp_boost"] = grp_b
        scored.append(result)

    return scored


def search(query: str, top_k: int = 3, neighborhood: str = "bostanli",
           parsed: dict | None = None) -> list[dict]:
    """Semantik restoran aramasi.

    parsed verilirse → LLM tarafindan ayrismis yapili parametreler kullanilir
    (negasyon, keyword, kategori filtreleme). Verilmezse → regex tabanli fallback.
    """
    embeddings_file = BASE_DIR / "data" / "embeddings" / f"{neighborhood}_embeddings.json"
    identity_file = BASE_DIR / "data" / "processed" / f"{neighborhood}_identity_cards.json"

    client = OpenAI()

    # LLM'den gelen semantic_query varsa embedding'i onunla al,
    # yoksa ham sorguyla
    search_text = parsed["semantic_query"] if parsed else query

    query_emb = (
        client.embeddings.create(
            model="text-embedding-3-small",
            input=[search_text],
        )
        .data[0]
        .embedding
    )

    with open(embeddings_file, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)

    with open(identity_file, "r", encoding="utf-8") as f:
        identity_cards = {c["place_id"]: c for c in json.load(f)}

    # ---------------------------------------------------------------
    # Dislama ve filtreleme: LLM ya da regex
    # ---------------------------------------------------------------
    if parsed:
        excluded_terms = parsed.get("must_exclude_terms", [])
        exclude_categories = parsed.get("exclude_categories", [])
        food_contexts: list[str] = []
        use_llm = True
        leftover_kws = parsed.get("must_include_keywords", [])
    else:
        excluded_terms, food_contexts = parse_negative_exclusions(query)
        exclude_categories = None
        use_llm = False
        leftover_kws = extract_leftover_keywords(query)

    scored = _score_restaurants(
        query=query,
        query_emb=query_emb,
        embeddings_data=embeddings_data,
        identity_cards=identity_cards,
        filter_query=query,
        excluded_terms=excluded_terms,
        food_contexts=food_contexts,
        exclude_categories=exclude_categories,
        use_llm_filter=use_llm,
    )

    # ---------------------------------------------------------------
    # Keyword boost: LLM'den must_include_keywords, regex'ten leftover
    # ---------------------------------------------------------------
    if leftover_kws:
        kw_boosts = keyword_match_boost(leftover_kws, scored, neighborhood)
        for r in scored:
            pid = r.get("_place_id", "") or r.get("name", "")
            kw_b = kw_boosts.get(pid, 0.85)
            r["_kw_boost"] = round(kw_b, 2)
            r["score"] = round(r["score"] * kw_b, 4)

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    # Fallback: tum keyword'ler bir arada hicbir restoranda eslesmediyse,
    # nadir kelimeleri cikarip sadece en sik gecenle boost'u yeniden dene.
    # Iki durum: (a) coklu keyword → en sik olani tut, (b) tek multi-word
    # keyword → kelimelere bolup en sik olani dene.
    if leftover_kws and all(r.get("_kw_boost", 1.0) <= 0.85 for r in scored):
        # Fallback icin aday keyword'ler: mevcut + multi-word'lerin parcalari
        fallback_candidates = list(leftover_kws)
        for kw in leftover_kws:
            parts = kw.split()
            if len(parts) > 1:
                for p in parts:
                    if p not in fallback_candidates:
                        fallback_candidates.append(p)

        # Sadece farkli keyword varsa fallback yap (en az 1 yeni aday)
        if len(fallback_candidates) > len(leftover_kws) or len(leftover_kws) > 1:
            kw_freq = _count_keyword_frequency(fallback_candidates, scored)
            sorted_candidates = sorted(fallback_candidates, key=lambda kw: kw_freq.get(kw, 0))
            best_kw = sorted_candidates[-1] if sorted_candidates else None

            if best_kw and kw_freq.get(best_kw, 0) > 0:
                drop_kws = [kw for kw in leftover_kws if kw != best_kw]
                # Revert eski 0.85 penalty
                for r in scored:
                    old_kw = r.get("_kw_boost", 1.0)
                    if old_kw <= 0.85:
                        r["score"] = round(r["score"] / old_kw, 4)

                best_kw_boosts = keyword_match_boost([best_kw], scored, neighborhood)
                for r in scored:
                    pid = r.get("_place_id", "") or r.get("name", "")
                    kw_b = best_kw_boosts.get(pid, 0.85)
                    r["_kw_boost"] = round(kw_b, 2)
                    r["score"] = round(r["score"] * kw_b, 4)
                    r["_is_fallback"] = True
                    r["_fallback_note"] = (
                        f"'{' '.join(drop_kws)}' icin sonuc bulunamadi. "
                        f"Alternatif olarak '{best_kw}' eslesmeleri gosteriliyor."
                    )
                    r["_failed_keywords"] = drop_kws

                scored.sort(key=lambda x: x["score"], reverse=True)
                top = scored[:top_k]

    # Yorum kaniti
    if leftover_kws:
        review_evidence = search_raw_reviews(leftover_kws, top, neighborhood)
        for r in top:
            name = r["name"]
            if name in review_evidence:
                r["_review_evidence"] = review_evidence[name]
                r["_leftover_keywords"] = leftover_kws

    notes = customer_notes(query, top)
    for r, note in zip(top, notes):
        r["_customer_note"] = note
    return top


# ---------------------------------------------------------------
# Artik kelime tespiti: Sorguda mevcut sistemler tarafindan
# "tuketilmeyen" kelimeleri bulup ham yorumlarda arar.
# Ornek: "icinde domates olan donerci" → doner → CATEGORY_PROXIMITY'de var
# ama "domates" hicbir sistemde yok → ham yorumda aranacak.
# ---------------------------------------------------------------

# Mevcut sistemlerin tukettigi tum kelimeler/ifadeler
def _build_consumed_words() -> set[str]:
    consumed = set()

    # CATEGORY_PROXIMITY: sadece alt terimleri (trigger'leri) tuket,
    # key'leri (lahmacun, pide, kofte gibi spesifik yemek isimleri) degil.
    # Onlar keyword bazli boost'ta kullanilabilmeli.
    # NOT: Bazi key'ler kendi dict'lerinde value olarak da gorunur (self-boost).
    for proximity_map in CATEGORY_PROXIMITY.values():
        for term in proximity_map:
            if term not in CATEGORY_PROXIMITY:  # key olanlari atla
                consumed.add(term.lower())

    # HARD_FILTERS trigger kelimeleri
    for triggers in HARD_FILTERS.values():
        for term in triggers:
            for w in term.split():
                consumed.add(w.lower())

    # Price trigger kelimeleri
    for w in ["ucuz", "bütçe", "ekonomik", "hesaplı", "uygun", "fiyat", "fiyatı",
              "fiyatlı", "fiyatlar"]:
        consumed.add(w)

    # Group trigger kelimeleri
    for w in ["kalabalık", "kalabalik", "büyük", "geniş", "grup", "yemek", "toplantı"]:
        consumed.add(w)

    # Negatif filtre kalip kelimeleri (olmayan, -siz, degil)
    for w in ["olmayan", "değil", "degil"]:
        consumed.add(w)

    # "en iyi / iyi / guzel / var mi / ariyorum / bul / goster / listele" gibi genel kaliplar
    for w in ["iyi", "güzel", "guzel", "var", "mı", "mi", "mu", "mü",
              "arıyorum", "ariyorum", "bul", "göster", "goster", "listeler",
              "listele", "liste", "nerede", "nerden", "hangi", "tavsiye",
              "öner", "oner", "oneri", "öneri", "önerir", "onerir",
              "önerirsin", "onerirsin", "önersene", "onersene",
              "bana", "bize", "için", "icin",
              "çok", "cok", "daha", "en", "bir", "bu", "şu", "su",
              "ve", "ile", "veya", "gibi", "kadar", "ama", "fakat",
              "yapan", "yapiyor", "yapıyor", "yapıyormuş", "eden",
              "koyan", "koyuyor", "kullanan", "olan", "olsun",
              "nasıl", "nasil", "neden", "niye", "ne", "kim",
              "da", "de", "ta", "te", "ki", "ya", "la", "le",
              "restoran", "restorant", "mekan", "yer", "yeri",
              "mekanı", "mekanları", "mekanda", "mekana", "mekandan",
              "içinde", "icinde", "içine", "icine", "içerisinde", "icerisinde",
              "olsun", "yemek", "yemekler", "yemekleri",
              "misin", "mısın", "musun", "müsün", "midir", "mıdır",
              "mudur", "müdür"]:
        consumed.add(w)

    return consumed


CONSUMED_WORDS = _build_consumed_words()


def _is_suffixed_consumed(word: str) -> bool:
    """Turkce ek almis bir kelimenin koku CONSUMED_WORDS'ta mi kontrol eder.
    'mekanı' → mekan ✓, 'mekanları' → mekan ✓, 'mekanda' → mekan ✓"""
    # Yaygin Turkce ekler (uzun→kisa, false positive'i azaltmak icin)
    _SUFFIXES = [
        "ları", "leri", "larıdır", "leridir",
        "ında", "inde", "unda", "ünde",
        "ından", "inden", "undan", "ünden",
        "ına", "ine", "una", "üne",
        "ını", "ini", "unu", "ünü",
        "nın", "nin", "nun", "nün",
        "yla", "yle",
        "lar", "ler", "dır", "dir", "dur", "dür",
        "dan", "den", "tan", "ten",
        "da", "de", "ta", "te",
        "na", "ne", "nı", "ni", "nu", "nü",
        "a", "e", "ı", "i", "u", "ü",
        "ki",
    ]
    w = word.lower()
    for suffix in _SUFFIXES:
        if w.endswith(suffix):
            stem = w[:-len(suffix)]
            if len(stem) >= 4 and stem in CONSUMED_WORDS:
                return True
    return False


def extract_leftover_keywords(query: str) -> list[str]:
    """Sorgudaki mevcut sistemler tarafindan tuketilmemis anlamli kelimeleri doner.
    Bunlar genellikle spesifik malzeme/urun adlaridir (domates, sogan, kusbasi vs)."""
    import re
    query_lower = query.lower()

    # Negatif kaliplardaki kelimeleri cikar (onlar dislama, leftover degil)
    excluded_terms, _food_contexts = parse_negative_exclusions(query)
    excluded_lower = [e.lower() for e in excluded_terms]

    # Multi-word ifadeleri once kontrol et (deniz kenari, acik alan vs.)
    multi_word_consumed = set()
    for triggers in HARD_FILTERS.values():
        for term in triggers:
            if term in query_lower:
                for w in term.split():
                    multi_word_consumed.add(w.lower())

    for phrase in ["uygun fiyat", "fiyatı uygun", "uygun fiyatlı", "fiyatlar uygun",
                   "büyük grup", "buyuk grup", "kalabalık grup", "kalabalik grup",
                   "geniş grup", "genis grup", "açık alan", "acik alan", "açık hava",
                   "acik hava", "deniz kenarı", "deniz kenari", "park yeri",
                   "deniz manzaralı", "deniz manzarali"]:
        if phrase in query_lower:
            for w in phrase.split():
                multi_word_consumed.add(w.lower())

    words = re.findall(r"\w+", query_lower)
    leftover = []
    for w in words:
        w_clean = w.strip()
        if len(w_clean) < 3:
            continue
        if w_clean in excluded_lower:
            continue
        if w_clean in CONSUMED_WORDS:
            continue
        if w_clean in multi_word_consumed:
            continue
        # Turkce ek almis hali CONSUMED kokten turemisse atla (mekanı/mekanları→mekan)
        if _is_suffixed_consumed(w_clean):
            continue
        # -siz/-sız suffix: sogansiz → sogan (negatif eksiltme yakalar)
        for suffix in ["siz", "sız", "suz", "süz"]:
            if w_clean.endswith(suffix) and len(w_clean) > len(suffix) + 2:
                s_stem = w_clean[:-len(suffix)]
                if s_stem not in CONSUMED_WORDS:
                    leftover.append(w_clean)
                break
        else:
            # Isim cekim eklerini kirp: lahmacunu→lahmacun, corbaya→corba
            # Tum ekleri dene, en uzun gecerli koku sec (CONSUMED'da olmayan)
            _NOUN_SUFFIXES = ["nın", "nin", "nun", "nün", "nı", "ni", "nu", "nü",
                            "ının", "inin", "unun", "ünün", "ını", "ini", "unu", "ünü",
                            "ya", "ye", "yı", "yi", "yu", "yü",
                            "ı", "i", "u", "ü"]
            best_stem = w_clean
            best_stem_len = 0  # 0 → henüz geçerli kök bulunamadı
            for sfx in _NOUN_SUFFIXES:
                if w_clean.endswith(sfx):
                    s = w_clean[:-len(sfx)]
                    if len(s) >= 4 and s not in CONSUMED_WORDS and len(s) > best_stem_len:
                        best_stem = s
                        best_stem_len = len(s)
            leftover.append(best_stem)

    # Tekrarlari kaldir, siralamayi koru
    seen = set()
    unique = []
    for w in leftover:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def search_raw_reviews(
    leftover_keywords: list[str],
    candidates: list[dict],
    neighborhood: str = "bostanli",
    max_snippets: int = 3,
) -> dict[str, list[str]]:
    """Ham yorumlarda artik kelimeleri ara, restoran bazinda ilgili snippet'leri don.

    Returns: {restaurant_name: [snippet1, snippet2, ...]}
    """
    if not leftover_keywords:
        return {}

    raw_file = BASE_DIR / "data" / "raw" / f"{neighborhood}_raw.json"
    if not raw_file.exists():
        return {}

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # place_id → (name, reviews) mapping
    reviews_by_place: dict[str, tuple[str, list[str]]] = {}
    for r in raw_data:
        pid = r.get("placeId") or r.get("title", "")
        rname = r.get("title", "")
        all_reviews = r.get("reviews") or r.get("reviewsData") or []
        review_texts = []
        for rev in all_reviews:
            text = rev.get("text") or rev.get("reviewText") or ""
            if text.strip():
                review_texts.append(text.strip())
        if review_texts:
            reviews_by_place[pid] = (rname, review_texts)

    # Her aday restoranin yorumlarinda kelimeleri ara
    results: dict[str, list[str]] = {}
    for cand in candidates:
        name = cand.get("name", "")
        pid = cand.get("_place_id", "")

        # Once place_id ile dene
        reviews: list[str] = []
        if pid and pid in reviews_by_place:
            reviews = reviews_by_place[pid][1]
        else:
            # Isimle eslestir (place_id yoksa veya eslesmediyse)
            for rpid, (rname, rreviews) in reviews_by_place.items():
                if rname == name:
                    reviews = rreviews
                    break
        if not reviews:
            continue

        snippets: list[str] = []
        for review_text in reviews:
            review_lower = review_text.lower()
            # Tum leftover keyword'ler ayni review'da tam kelime olarak gecmeli
            if all(_contains_word(review_lower, kw) for kw in leftover_keywords):
                import re
                sentences = re.split(r"[.!?;]\s*", review_text)
                for sent in sentences:
                    sent_lower = sent.lower()
                    if any(_contains_word(sent_lower, kw) for kw in leftover_keywords):
                        sent_clean = sent.strip()
                        if len(sent_clean) > 10 and sent_clean not in snippets:
                            snippets.append(sent_clean)
                        if len(snippets) >= max_snippets:
                            break
                if len(snippets) >= max_snippets:
                    break

        if snippets:
            results[name] = snippets

    return results


def customer_notes(query: str, results: list[dict]) -> list[str]:
    """Her sonucun neden o sirada oldugunu karsilastirmali aciklar."""
    if not results:
        return []

    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 3]

    notes = []
    for i, r in enumerate(results):
        name = r["name"]
        rating = r.get("rating", 0)
        dishes = r.get("signature_dishes", [])
        liked = [li["name"] if isinstance(li, dict) else li for li in r.get("liked_items", [])]
        disliked = [di["name"] if isinstance(di, dict) else di for di in r.get("disliked_items", [])]
        pf = r.get("physical_features", {})
        view = pf.get("view", "")
        outdoor = pf.get("outdoor", "")
        parking = pf.get("parking", "")
        max_group = pf.get("max_group_size", 0)
        price_scale = r.get("price_scale", "")
        price_perception = r.get("price_perception", "")
        score = r.get("score", 0)
        men_b = r.get("_men_boost", 1.0)

        # Sorguyla eslesen spesifik item'lari bul
        matching_items = []
        for item in liked + disliked:
            item_lower = item.lower()
            for qw in query_words:
                if qw in item_lower and item not in matching_items:
                    matching_items.append(item)
                    break

        # Guclu yonleri topla
        strengths = []

        # Spesifik eslesen urunler
        if matching_items:
            strengths.append(f"\"{', '.join(matching_items[:3])}\" ovulmus")
        elif dishes:
            strengths.append(f"imza yemekleri: {', '.join(dishes[:3])}")

        # Puan
        if rating >= 4.5:
            strengths.append(f"{rating} puanla yuksek")
        elif rating >= 4.0:
            strengths.append(f"{rating} puan")

        # Fiyat (sadece sorgu fiyat odakliysa)
        if any(t in query_lower for t in ["ucuz", "bütçe", "ekonomik", "hesaplı"]):
            strengths.append(f"{price_perception} ({price_scale})")

        # Grup (sadece sorgu grup odakliysa)
        if any(t in query_lower for t in ["kalabalık", "büyük grup", "grup"]):
            strengths.append(f"max {max_group} kisi")

        # Fiziksel (her zaman goster)
        if view != "manzara yok" and view:
            strengths.append(view)
        if outdoor == "var":
            strengths.append("acik alan var")
        if parking not in ("yok", ""):
            strengths.append("otopark var")

        # Onceki sonucla karsilastirma
        if i == 0:
            note = f"#{i+1} \"{name}\": {', '.join(strengths)}."
        else:
            prev = results[i-1]
            prev_men = prev.get("_men_boost", 1.0)
            prev_rat = prev.get("_rat_boost", 1.0)
            prev_rating = prev.get("rating", 0)
            prev_name = prev["name"]

            # Farki aciklayan faktorler
            diff_reasons = []

            # Bahsedilme farki
            if men_b < prev_men:
                prev_liked = [li["name"] if isinstance(li, dict) else li for li in prev.get("liked_items", [])]
                prev_matches = [item for item in prev_liked if any(
                    qw in item.lower() for qw in query_words
                )]
                if prev_matches:
                    diff_reasons.append(
                        f"\"{prev_name}\"da aynı lezzet daha cok ovulmus "
                        f"({', '.join(prev_matches[:2])})"
                    )
                else:
                    diff_reasons.append(
                        f"\"{prev_name}\"da aranan lezzet yorumlarda daha cok gecmis"
                    )

            # Puan farki
            if rating < prev_rating:
                diff_reasons.append(
                    f"\"{prev_name}\" daha yuksek puanlı ({prev_rating} vs {rating})"
                )

            if not strengths:
                strengths.append("aradıgın kriterlere uyuyor")

            note = f"#{i+1} \"{name}\": {', '.join(strengths)}."
            if diff_reasons:
                note += f" Geride kalma nedeni: {'; '.join(diff_reasons)}."
            elif score < prev.get("score", 0):
                note += " Sıralamada bir alt sırada."

        notes.append(note)

    return notes


def format_result(result: dict, idx: int) -> str:
    rating = result.get("rating", 0)
    stars_str = f"{rating}★" if isinstance(rating, (int, float)) else "?★"
    dishes = ", ".join(result.get("signature_dishes", []))
    best_for = ", ".join(result.get("best_for", []))
    pf = result.get("physical_features", {})

    liked = result.get("liked_items", [])
    liked_str = ", ".join(f"{li['name']}" for li in liked[:5]) if liked else "-"

    disliked = result.get("disliked_items", [])
    disliked_str = ", ".join(f"{di['name']}({di.get('reason','')})" for di in disliked[:3]) if disliked else "-"

    group_info = ""
    if pf:
        group_ok = "✅" if pf.get("group_friendly") else "❌"
        max_g = pf.get("max_group_size", "?")
        group_info = f"\n  Grup:       {group_ok} (max {max_g} kisi)"

    lines = [
        f"\n{'='*60}",
        f"  #{idx}  {result['name']}  ⭐{stars_str}  (benzerlik: {result['score']})",
        f"{'='*60}",
    ]

    # Fallback: spesifik terim bulunamadi mesaji
    if result.get("_is_fallback"):
        lines.append(f"  ⚠️  {result.get('_fallback_note', '')}")

    lines += [
        f"  Adres:      {result.get('address', '-')}",
        f"  Kategori:   {', '.join(result.get('categories', []))}",
        f"  Mutfak:     {result.get('cuisine_style', '-')}",
        f"  Fiyat:      {result.get('price_range', '-')} | {result.get('price_scale', '-')} ({result.get('price_perception', '-')})",
        f"  Ambiyans:   {result.get('ambiance', '-')}",
        f"  Imza yemek: {dishes if dishes else '-'}",
        f"  👍 Begendi: {liked_str}",
        f"  👎 Sikayet: {disliked_str}",
        f"  En iyi:     {best_for if best_for else '-'}",
    ]
    if pf:
        lines += [
            f"  Manzara:    {pf.get('view', '-')}",
            f"  Dekor:      {pf.get('decor', '-')}",
            f"  Ses:        {pf.get('noise_level', '-')}",
            f"  Oturma:     {pf.get('seating', '-')} | Acik alan: {pf.get('outdoor', '-')} | Park: {pf.get('parking', '-')}",
            f"  Temizlik:   {pf.get('cleanliness', '-')}",
        ]
    lines += [
        group_info,
        f"  Neden:      {result.get('_customer_note', result.get('vibe_summary', '-'))}",
    ]
    # Artik kelimeler icin ham yorum kanitlari
    review_evidence = result.get("_review_evidence", [])
    if review_evidence:
        lines.append(f"  📝 Yorum detayi:")
        for snippet in review_evidence[:3]:
            # Highlight edilen kelimeleri isaretle
            lines.append(f"     \"...{snippet.strip()}...\"")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python -m src.search.semantic_search <sorgu> [--semt=kadikoy] [--top=5]")
        print("Ornek: python -m src.search.semantic_search \"Sahilde romantik aksam yemegi\" --semt=kadikoy --top=5")
        return

    neighborhood = "bostanli"
    top_k = 3
    query_parts = []
    for arg in sys.argv[1:]:
        if arg.startswith("--semt="):
            neighborhood = arg.split("=")[1]
        elif arg.startswith("--top="):
            top_k = int(arg.split("=")[1])
        else:
            query_parts.append(arg)

    query = " ".join(query_parts)
    embeddings_file = BASE_DIR / "data" / "embeddings" / f"{neighborhood}_embeddings.json"

    if not embeddings_file.exists():
        print(f"HATA: Embedding dosyasi bulunamadi: {embeddings_file}")
        print("Once vektorlestirme script'ini calistirin: python -m src.vectorizer.embed")
        return

    print(f"\n🔍 [{neighborhood}] Arama: \"{query}\"\n")

    results = search(query, top_k=top_k, neighborhood=neighborhood)

    if not results:
        print("Sonuc bulunamadi.")
        return

    for i, r in enumerate(results, 1):
        print(format_result(r, i))

    print()


if __name__ == "__main__":
    main()
