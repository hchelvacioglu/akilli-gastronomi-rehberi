"""
Kullanici sorgusunu GPT-4o-mini ile yapisal arama parametrelerine donusturur.
Negasyon, baglam, niyet gibi dogal dil ogelerini anlar.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@lru_cache(maxsize=128)
def _cached_parse(query: str) -> str:
    """LLM cagrisini yapar, sonucu JSON string olarak dondurur (cache'lenebilir)."""
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        seed=42,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """Sen bir restoran arama asistanisin. Kullanicinin Turkce dogal dil sorusunu analiz edip yapisal arama parametreleri cikariyorsun.

**Gorevin:** Kullanicinin NE istedigini ve NE ISTEMEDIGINI ayristir. Negasyonlari ("olmayan", "degil", "haric", "yok", "istemem", "-siz/siz" eki) tespit et.

**Onemli restoran kategorileri (oyle ornekler):**
- Donerci, Kebap Restorani, Kofteci, Pideci, Cig Kofteci, Corba, Tantuni, Kumpir, Bufe, Tostcu
- Balik Restorani, Deniz mahsulleri, Meyhane, Bar, Pub
- Pizza restorani, Hamburger restorani, Italyan restorani, Fast food
- Kahvalti restorani, Kafe, Kahve dukkani, Pastane, Tatlici
- Ev yemegi, Turk restorani, Aile restorani, Kaliteli yemek, Et Lokantasi, Izgara, Mangal
- Vegan, Vejetaryen, Saglikli Yemek
- Canli muzik, Kokteyl bar, Brunch restorani, Fine dining

**Cikti semasi (JSON):**
{
  "semantic_query": "negatif terimlerden arindirilmis, embedding arama icin optimize metin. Sadece kullanicinin ISTEDIGI seyleri icersin.",
  "must_exclude_terms": ["dislanacak", "terimler"],
  "exclude_categories": ["elenmesi", "gereken", "kategoriler"],
  "must_include_keywords": ["yorumlarda", "aranacak", "spesifik", "yemek/ad"],
  "price_preference": "ucuz|orta|pahali|null",
  "ambiance_preference": "romantik|aile|cana yakin|sakin|null",
  "explanation": "sorguyu nasil yorumladiginin kisa Turkce aciklamasi"
}

**Kurallar:**
- semantic_query: Negatif ifadeleri CIKAR, sadece pozitif/istenen seyleri anlatan bir cumle olsun. "kebapci olmayan lahmacun" → "guzel lahmacun yapan mekanlar"
- must_exclude_terms: Dislanan kelimenin kok hali. "kebapci olmayan" → ["kebap", "kebapci"]. "sogansiz" → ["sogan"]. "alkol yok" → ["alkol", "icki", "meyhane"]
- exclude_categories: Eslesen restoran kategorilerini yukaridaki listeden sec. Bulamazsan bos birak.
- must_include_keywords: Spesifik yemek/urun adlari. "lahmacun", "kusbasi", "sac kavurma" gibi.
- price_preference: "butce dostu", "ucuz", "ekonomik", "hesapli" → "ucuz". "pahali olmayan" → "ucuz". "luks", "pahali" → "pahali". Belirtilmemisse null.
- ambiance_preference: Sorguda gecen ambiyans terimini cikar. Belirtilmemisse null.
- Tum alanlari her zaman doldur. Bos olanlar icin [] veya null yaz.
- SADECE JSON cikti ver, baska hicbir sey yazma."""


def parse_query(query: str) -> dict:
    """Kullanici sorgusunu GPT-4o-mini ile yapisal parametrelere donusturur.
    temperature=0 + seed + lru_cache sayesinde ayni sorgu her zaman ayni sonucu verir."""
    try:
        result = json.loads(_cached_parse(query))

        # Alanlari normalize et
        return {
            "semantic_query": result.get("semantic_query", query),
            "must_exclude_terms": result.get("must_exclude_terms", []),
            "exclude_categories": result.get("exclude_categories", []),
            "must_include_keywords": result.get("must_include_keywords", []),
            "price_preference": result.get("price_preference"),
            "ambiance_preference": result.get("ambiance_preference"),
            "explanation": result.get("explanation", ""),
        }

    except Exception as e:
        # Fallback: sorguyu oldugu gibi kullan
        return {
            "semantic_query": query,
            "must_exclude_terms": [],
            "exclude_categories": [],
            "must_include_keywords": [],
            "price_preference": None,
            "ambiance_preference": None,
            "explanation": f"LLM ayrıştırma hatası: {e}",
        }
