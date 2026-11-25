"""R1 step 2b-U1/U2: download and screen UEH works from OpenAlex.

Query: institutions.id:I181727666, publication_year 2016-2025 (all types), full records.
Screening (automatic, mirrors the HUB rules): (i) type article / conference-paper / review;
(ii) not retracted; (iii) English text: OpenAlex language 'en', or title without Vietnamese
diacritics; (iv) at least a title.
Outputs: ueh/raw/page_*.json, ueh/ueh_corpus.jsonl, ueh/ueh_screening.json
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

from openalex_client import abstract_text, get

R1 = Path(__file__).resolve().parent.parent
RAW = R1 / "ueh" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UEH = "https://openalex.org/I181727666"
VN_CHARS = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I)


def download() -> list[dict]:
    works, cursor, page = [], "*", 0
    while cursor:
        page += 1
        data = get("/works", {"filter": "institutions.id:I181727666,publication_year:2016-2025",
                              "per-page": 200, "cursor": cursor})
        (RAW / f"page_{page:03d}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not data.get("results"):
            break
    (RAW / "query.json").write_text(json.dumps({"filter": "institutions.id:I181727666,publication_year:2016-2025",
                                                 "date": date.today().isoformat(), "n": len(works)}), encoding="utf-8")
    return works


def english(work: dict) -> bool:
    title = work.get("display_name") or ""
    if work.get("language") == "en":
        return not VN_CHARS.search(title)
    return bool(title) and not VN_CHARS.search(title) and work.get("language") in (None, "", "en")


def main() -> None:
    works = download()
    reasons = Counter()
    corpus = []
    seen = set()
    for w in works:
        if w["id"] in seen:
            reasons["duplicate_id"] += 1
            continue
        seen.add(w["id"])
        if w.get("type") not in ("article", "conference-paper", "review"):
            reasons["type"] += 1
            continue
        if w.get("is_retracted"):
            reasons["retracted"] += 1
            continue
        if not w.get("display_name"):
            reasons["no_title"] += 1
            continue
        if not english(w):
            reasons["not_english"] += 1
            continue
        ab = abstract_text(w)
        authors = []
        for a in w.get("authorships", []):
            if any(i.get("id") == UEH for i in a.get("institutions", [])):
                aid = (a.get("author") or {}).get("id")
                if aid:
                    authors.append(aid)
        raws = [s for a in w.get("authorships", []) for s in (a.get("raw_affiliation_strings") or [])]
        corpus.append({"id": w["id"], "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                       "year": w["publication_year"], "title": w["display_name"], "abstract": ab,
                       "text": (w["display_name"] + " " + ab).strip(), "has_abstract": bool(ab.strip()),
                       "type": w.get("type"), "language": w.get("language"),
                       "ueh_author_ids": sorted(set(authors)), "n_raw_affiliation_strings": len(raws),
                       "has_any_raw_affiliation": bool(raws)})
    with (R1 / "ueh" / "ueh_corpus.jsonl").open("w", encoding="utf-8") as h:
        for r in corpus:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"query_date": date.today().isoformat(), "n_downloaded": len(works), "excluded": dict(reasons),
               "n_corpus": len(corpus), "n_with_abstract": sum(r["has_abstract"] for r in corpus),
               "by_year": dict(sorted(Counter(r["year"] for r in corpus).items())),
               "abstract_by_year": dict(sorted(Counter(r["year"] for r in corpus if r["has_abstract"]).items()))}
    (R1 / "ueh" / "ueh_screening.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
