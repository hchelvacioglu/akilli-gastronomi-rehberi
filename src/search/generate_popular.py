"""
Populer sorgulari onceden hesaplayip JSON'a kaydeder.
Ayda bir guncellemek icin: python -m src.search.generate_popular [kadikoy|bostanli]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.search.semantic_search import search
from src.search.query_parser import parse_query

BASE_DIR = Path(__file__).resolve().parent.parent.parent

POPULAR_QUERIES = [
    ("date-mekanlari", "👫 Date mekanları", "En iyi date mekanları"),
    ("lahmacuncular", "🥩 Lahmacuncular", "En iyi lahmacuncular"),
    ("meze-keyfi", "🥂 Meze keyfi", "En iyi mezeye sahip mekanlar"),
    ("romantik-balik", "🐟 Deniz kenarı romantik balık", "Deniz kenarında romantik balık"),
    ("kahvalti", "🥞 Kahvaltı mekanları", "Çoluk çocuklu kahvaltı"),
    ("donerci", "🥙 Hızlı & ucuz dönerci", "Hızlı ve ucuz dönerci"),
    ("sac-kavurma", "🍳 Sac kavurma", "Sac kavurma yapan yer"),
    ("manzarali-pizza", "🍕 Manzaralı pizza", "Pahalı olmayan manzaralı pizza"),
]

MIN_SCORE = 0.30
MAX_PER_QUERY = 15


def generate(neighborhood: str):
    output = []
    for slug, title, query in POPULAR_QUERIES:
        print(f"[{title}] arama yapiliyor...", flush=True)
        parsed = parse_query(query)
        results = search(query, top_k=MAX_PER_QUERY, neighborhood=neighborhood, parsed=parsed)

        # MIN_SCORE filtresi, sadece gerekli alanlari sakla
        filtered = []
        for r in results:
            if r["score"] < MIN_SCORE:
                continue
            filtered.append({
                "name": r.get("name", ""),
                "rating": r.get("rating", 0),
                "score": r.get("score", 0),
                "cuisine_style": r.get("cuisine_style", ""),
                "categories": r.get("categories", [])[:3],
                "_place_id": r.get("_place_id", ""),
            })

        print(f"  → {len(filtered)} sonuc", flush=True)
        output.append({
            "slug": slug,
            "title": title,
            "query": query,
            "results": filtered,
        })

    out_dir = BASE_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"popular_{neighborhood}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nKaydedildi: {out_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python -m src.search.generate_popular <semt>")
        print("Secenekler: kadikoy, bostanli, all")
    else:
        nb = sys.argv[1].lower()
        if nb == "all":
            for n in ["kadikoy", "bostanli"]:
                generate(n)
        else:
            generate(nb)
