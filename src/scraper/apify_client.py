"""
Apify Google Maps Scraper entegrasyonu.
Belirtilen semtteki 4.0+ puanli restoranlari ceker.
Kullanim: python -m src.scraper.apify_client [kadikoy|bostanli]
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()

MIN_RATING = 4.0
MAX_REVIEWS_PER_PLACE = 50

NEIGHBORHOODS = {
    "bostanli": {
        "center": {"lat": 38.455, "lng": 27.105},
        "location_name": "Bostanli, Izmir, Turkiye",
        "queries": [
            "donerci Bostanli Izmir",
            "kofteci Bostanli Izmir",
            "tatlici Bostanli Izmir",
            "ev yemegi Bostanli Izmir",
            "iskender Bostanli Izmir",
            "cig kofte Bostanli Izmir",
            "corba Bostanli Izmir",
            "kokorec Bostanli Izmir",
            "midye Bostanli Izmir",
            "Bostanli Izmir",
            "sokak lezzeti Bostanli Izmir",
            "tantuni Bostanli Izmir",
            "kumpir Bostanli Izmir",
            "et restorani Bostanli Izmir",
            "pizza Bostanli Izmir",
        ],
    },
    "kadikoy": {
        "center": {"lat": 40.990, "lng": 29.025},
        "location_name": "Kadikoy, Istanbul, Turkiye",
        "queries": [
            "donerci Kadikoy Istanbul",
            "kofteci Kadikoy Istanbul",
            "tatlici Kadikoy Istanbul",
            "ev yemegi Kadikoy Istanbul",
            "iskender Kadikoy Istanbul",
            "cig kofte Kadikoy Istanbul",
            "corba Kadikoy Istanbul",
            "kokorec Kadikoy Istanbul",
            "midye Kadikoy Istanbul",
            "Kadikoy Istanbul",
            "sokak lezzeti Kadikoy Istanbul",
            "tantuni Kadikoy Istanbul",
            "kumpir Kadikoy Istanbul",
            "et restorani Kadikoy Istanbul",
            "pizza Kadikoy Istanbul",
            "balikci Kadikoy Istanbul",
            "meyhane Kadikoy Istanbul",
            "kahvalti Kadikoy Istanbul",
            "burger Kadikoy Istanbul",
            "kebapci Kadikoy Istanbul",
        ],
    },
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def scrape_restaurants(neighborhood: str) -> list[dict]:
    cfg = NEIGHBORHOODS[neighborhood]
    client = ApifyClient(os.environ["APIFY_API_KEY"])

    all_places = {}
    queries = cfg["queries"]

    for query in queries:
        print(f"Araniyor: {query}")

        run_input = {
            "searchStringsArray": [query],
            "location": cfg["location_name"],
            "maxCrawledPlacesPerSearch": 30,
            "language": "tr",
            "maxReviews": MAX_REVIEWS_PER_PLACE,
            "includesReviews": True,
            "reviewsSort": "mostRelevant",
            "maxImages": 0,
            "includeHistoricalPlaces": False,
            "includeClosingHours": True,
        }

        try:
            run = client.actor("compass/crawler-google-places").call(run_input=run_input)

            items = list(
                client.dataset(run["defaultDatasetId"]).iterate_items()
            )

            for item in items:
                place_id = item.get("placeId") or item.get("title", "")
                if place_id and place_id not in all_places:
                    score = item.get("totalScore") or item.get("reviewsScore") or 0
                    if isinstance(score, (int, float)) and score >= MIN_RATING:
                        all_places[place_id] = item
                        print(f"  ✓ {item.get('title')} (puan: {score})")

        except Exception as e:
            print(f"  Hata: {e}")
            continue

    places = list(all_places.values())
    places.sort(
        key=lambda p: p.get("totalScore") or p.get("reviewsScore") or 0, reverse=True
    )
    print(f"\nToplam {len(places)} restoran bulundu (>{MIN_RATING} puan).")
    return places


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python -m src.scraper.apify_client <semt>")
        print(f"Secenekler: {', '.join(NEIGHBORHOODS.keys())}")
        return

    neighborhood = sys.argv[1].lower()
    if neighborhood not in NEIGHBORHOODS:
        print(f"HATA: '{neighborhood}' bilinmiyor. Secenekler: {', '.join(NEIGHBORHOODS.keys())}")
        return

    output_file = OUTPUT_DIR / f"{neighborhood}_raw.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mevcut veriyi yukle (varsa)
    existing = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            for r in existing_data:
                pid = r.get("placeId") or r.get("title", "")
                if pid:
                    existing[pid] = r
        print(f"Mevcut {len(existing)} restoran yuklendi.")

    # Yeni restoranlari cek
    new_restaurants = scrape_restaurants(neighborhood)

    # Merge: yeni restoranlari ekle
    added = 0
    for r in new_restaurants:
        pid = r.get("placeId") or r.get("title", "")
        if pid and pid not in existing:
            score = r.get("totalScore") or r.get("reviewsScore") or 0
            if isinstance(score, (int, float)) and score >= MIN_RATING:
                existing[pid] = r
                added += 1
                print(f"  + {r.get('title')} (puan: {score})")

    merged = list(existing.values())
    merged.sort(
        key=lambda p: p.get("totalScore") or p.get("reviewsScore") or 0, reverse=True
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n{added} yeni restoran eklendi. Toplam: {len(merged)}")
    print(f"Ham veri kaydedildi: {output_file}")


if __name__ == "__main__":
    main()
