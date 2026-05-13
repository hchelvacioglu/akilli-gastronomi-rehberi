"""
Restoran yorumlarini LLM ile analiz ederek her mekan icin "kimlik karti" olusturur.
Kullanim: python -m src.analyzer.llm_identity [kadikoy|bostanli]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SYSTEM_PROMPT = """Sen tecrubeli bir gastronomi elestirmeni ve yemek kulturleri uzmanisin.
Restoran yorumlarini analiz ederek yapisal bir kimlik karti cikariyorsun.

Kurallar:
- Sadece yorumlarda gecen bilgileri kullan, uydurma.
- liked_items ve disliked_items icin yorumlarda adi gecen spesifik urunleri/yemekleri listele.
- physical_features icin yorumlarda bahsedilen gozlemlere dayan.
- group_friendly ve max_group_size icin yorumlarda "kalabalik", "buyuk masa", "grup", "aile" gibi ifadeleri tara.
- price_scale Google Maps'in 4 seviyeli (₺ - ₺₺₺₺) sistemini kullan.
- Tum ciktilari Turkce yaz.
- JSON disinda hicbir sey yazma."""

IDENTITY_PROMPT = """Asagidaki restoran yorumlarini analiz et ve su JSON semasina uygun bir kimlik karti olustur:

{{
  "signature_dishes": ["yemek1", "yemek2", "yemek3"],
  "liked_items": [{{"name": "urun/yemek adi", "mention_count": 5, "sentiment": "pozitif"}}, ...],
  "disliked_items": [{{"name": "urun/yemek adi", "reason": "soguk gelmesi", "mention_count": 2}}, ...],
  "ambiance": "romantik / aile / casual / fine dining / sahilde / esnaf / ogrenci / nostaljik / bohem / ...",
  "strengths": ["guclu yon 1", "guclu yon 2", ...],
  "weaknesses": ["zayif yon 1", ...],
  "best_for": ["kullanici senaryosu 1", "kullanici senaryosu 2", ...],
  "price_scale": "₺ / ₺₺ / ₺₺₺ / ₺₺₺₺",
  "price_perception": "ucuz / orta / pahali / cok pahali",
  "cuisine_style": "Turk / Akdeniz / Italyan / Karadeniz / ...",
  "physical_features": {{
    "view": "deniz manzarali / sehir manzarali / manzara yok / ...",
    "decor": "modern / rustik / nostaljik / sade / luks / ...",
    "noise_level": "sessiz / orta / canli / gurultulu",
    "cleanliness": "cok temiz / temiz / orta / ...",
    "seating": "rahat / dar / genis / ...",
    "outdoor": "var / yok / kisitli",
    "parking": "var / yok / sokak / otopark / vale",
    "group_friendly": true/false,
    "max_group_size": 8
  }},
  "vibe_summary": "2-3 cumlelik mekanin ruhunu ozetleyen metin"
}}

Restoran: {name}
Kategori: {categories}
Google Puan: {rating}
Fiyat Araligi (Google Maps): {price_range}

Yorumlar ({review_count} adet):
{reviews}

Sadece JSON ciktisi ver:"""


def load_raw_data() -> list[dict]:
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_reviews_text(restaurant: dict, max_reviews: int = 50) -> tuple[str, int]:
    reviews = restaurant.get("reviews", [])
    if not reviews:
        reviews = restaurant.get("reviewsData", [])
    if not reviews:
        return "(Yorum bulunamadi)", 0

    lines = []
    for r in reviews[:max_reviews]:
        text = r.get("text") or r.get("reviewText") or ""
        stars = r.get("stars") or r.get("rating") or 0
        if text.strip():
            lines.append(f"[{stars}★] {text.strip()}")

    return "\n".join(lines) if lines else "(Yorum metni yok)", len(lines)


def get_price_info(restaurant: dict) -> str:
    price = restaurant.get("price", "")
    if price and price != "None":
        return str(price)
    return "belirtilmemis"


def analyze_restaurant(client: OpenAI, restaurant: dict) -> dict | None:
    name = restaurant.get("title", "Bilinmeyen Restoran")
    categories = ", ".join(restaurant.get("categories", [])) or "belirtilmemis"
    rating = restaurant.get("totalScore") or restaurant.get("reviewsScore") or 0
    price_range = get_price_info(restaurant)
    reviews_text, review_count = extract_reviews_text(restaurant, max_reviews=50)

    user_prompt = IDENTITY_PROMPT.format(
        name=name,
        categories=categories,
        rating=rating,
        price_range=price_range,
        reviews=reviews_text,
        review_count=review_count,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        identity_card = json.loads(response.choices[0].message.content)
        return {
            "place_id": restaurant.get("placeId") or restaurant.get("title", ""),
            "name": name,
            "rating": rating,
            "address": restaurant.get("address", ""),
            "location": restaurant.get("location", {}),
            "categories": restaurant.get("categories", []),
            "price_range": price_range,
            "reviews_count": restaurant.get("reviewsCount", 0),
            "analyzed_reviews": review_count,
            "identity_card": identity_card,
        }

    except Exception as e:
        print(f"  Hata ({name}): {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python -m src.analyzer.llm_identity <semt>")
        return

    neighborhood = sys.argv[1].lower()
    raw_file = BASE_DIR / "data" / "raw" / f"{neighborhood}_raw.json"
    output_dir = BASE_DIR / "data" / "processed"
    output_file = output_dir / f"{neighborhood}_identity_cards.json"

    if not os.environ.get("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY ortam degiskeni ayarlanmamis.")
        print("Proje ana dizininde .env dosyasi olusturun ve API anahtarini ekleyin.")
        return

    if not raw_file.exists():
        print(f"HATA: Ham veri dosyasi bulunamadi: {raw_file}")
        print("Once scraping script'ini calistirin: python -m src.scraper.apify_client")
        return

    client = OpenAI()
    with open(raw_file, "r", encoding="utf-8") as f:
        restaurants = json.load(f)

    print(f"{len(restaurants)} restoran analiz edilecek.\n")

    identity_cards = []
    for i, r in enumerate(restaurants):
        name = r.get("title", "?")
        print(f"[{i+1}/{len(restaurants)}] {name} ...", end=" ", flush=True)

        card = analyze_restaurant(client, r)
        if card:
            identity_cards.append(card)
            print("✓")
        else:
            print("✗")

        if i < len(restaurants) - 1:
            time.sleep(0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(identity_cards, f, ensure_ascii=False, indent=2)

    print(f"\n{len(identity_cards)} kimlik karti kaydedildi: {output_file}")


if __name__ == "__main__":
    main()
