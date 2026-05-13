"""
Kimlik kartlarini OpenAI embeddings ile vektorlestirip JSON'a kaydeder.
Kullanim: python -m src.vectorizer.embed [kadikoy|bostanli]
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def identity_to_text(card: dict) -> str:
    """Kimlik kartini embedding icin tek bir metne donusturur."""
    ic = card.get("identity_card", {})
    parts = [
        f"Restoran: {card.get('name', '')}",
        f"Kategoriler: {', '.join(card.get('categories', []))}",
        f"Imza yemekler: {', '.join(ic.get('signature_dishes', []))}",
        f"Ambiyans: {ic.get('ambiance', '')}",
        f"Guclu yonler: {', '.join(ic.get('strengths', []))}",
        f"Zayif yonler: {', '.join(ic.get('weaknesses', []))}",
        f"En iyi senaryolar: {', '.join(ic.get('best_for', []))}",
        f"Mutfak stili: {ic.get('cuisine_style', '')}",
        f"Fiyat algisi: {ic.get('price_perception', '')}",
        f"Ozet: {ic.get('vibe_summary', '')}",
    ]
    return "\n".join(parts)


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python -m src.vectorizer.embed <semt>")
        return

    neighborhood = sys.argv[1].lower()
    identity_file = BASE_DIR / "data" / "processed" / f"{neighborhood}_identity_cards.json"
    output_dir = BASE_DIR / "data" / "embeddings"
    output_file = output_dir / f"{neighborhood}_embeddings.json"

    if not os.environ.get("OPENAI_API_KEY"):
        print("HATA: OPENAI_API_KEY ortam degiskeni ayarlanmamis.")
        return

    if not identity_file.exists():
        print(f"HATA: Kimlik karti dosyasi bulunamadi: {identity_file}")
        print("Once analyzer script'ini calistirin: python -m src.analyzer.llm_identity")
        return

    client = OpenAI()

    with open(identity_file, "r", encoding="utf-8") as f:
        cards = json.load(f)

    print(f"{len(cards)} restoran vektorlestirilecek.")

    texts = [identity_to_text(c) for c in cards]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    embeddings = []
    for card, emb_data in zip(cards, response.data):
        embeddings.append(
            {
                "place_id": card.get("place_id", ""),
                "name": card.get("name", ""),
                "embedding": emb_data.embedding,
                "text_used": identity_to_text(card),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    dims = len(embeddings[0]["embedding"]) if embeddings else 0
    print(f"{len(embeddings)} embedding ({dims} boyutlu) kaydedildi: {output_file}")


if __name__ == "__main__":
    main()
