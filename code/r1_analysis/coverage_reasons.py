"""R1 step 1c: why OpenAlex missed the 123 HUB papers added from the HUB list,
and how many OpenAlex records carry a HUB name string without the HUB ID.

Outputs: out/coverage_reasons.csv, out/coverage_reasons.json,
         out/hub_unlinked_raw_strings.csv
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from openalex_client import cursor_all, get
from r1_common import BASE, OUT, hub_rows, write_csv, write_json

HUB_ID = "https://openalex.org/I3129629310"
ORIG = BASE.parent / "topic7_original_scope_2026-10-26"
HUB_NAME = re.compile(r"banking university|university of banking|dai hoc ngan hang|truong dai hoc ngan hang|"
                      r"\bhub\b.*banking|ho chi minh university of banking", re.I)
OTHER = re.compile(r"banking academy|hoc vien ngan hang|academy of banking|university of economics|"
                   r"finance and banking university|university of finance|ueh", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def is_hub_string(s: str) -> bool:
    n = norm(s)
    return bool(HUB_NAME.search(n)) and not (OTHER.search(n) and not re.search(r"banking university|university of banking", n))


def find_work(add: dict, review: dict) -> tuple[dict | None, str]:
    ids = [x for x in (add.get("openalex_or_external_id", ""), review.get("candidate_openalex_id", "")) if "openalex.org/W" in x]
    for oid in ids:
        w = get("/works/" + oid.rsplit("/", 1)[-1])
        if w.get("id"):
            return w, "openalex_id"
    doi = (add.get("doi") or review.get("verified_doi") or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    if doi:
        w = get("/works/doi:" + doi)
        if w.get("id"):
            return w, "doi"
    title = (add.get("matched_title") or add.get("listed_title") or "").strip()
    if title:
        q = re.sub(r"[^\w\s]", " ", title)[:200]
        data = get("/works", {"filter": f"title.search:{q}", "per-page": 10,
                              "select": "id,display_name,publication_year,authorships,doi"})
        year = int(add.get("matched_year") or add.get("listed_year") or 0)
        best, score = None, 0.0
        for w in data.get("results", []):
            s = SequenceMatcher(None, norm(title), norm(w.get("display_name", ""))).ratio()
            if s > score and abs((w.get("publication_year") or 0) - year) <= 1:
                best, score = w, s
        if best and score >= 0.9:
            full = get("/works/" + best["id"].rsplit("/", 1)[-1])
            return full, f"title({score:.2f})"
    return None, "not_found"


def classify(w: dict | None) -> tuple[str, list[str]]:
    if not w:
        return "A_not_in_openalex", []
    auths = w.get("authorships", [])
    raws = [s for a in auths for s in (a.get("raw_affiliation_strings") or [])]
    linked = any(inst.get("id") == HUB_ID for a in auths for inst in a.get("institutions", []))
    if linked:
        return "E_now_linked", raws
    if not raws:
        return "B_no_affiliation_string", raws
    if any(is_hub_string(s) for s in raws):
        return "D_hub_string_not_linked", raws
    return "C_strings_without_hub_name", raws


def reasons() -> dict:
    adds = {r["review_id"]: r for r in csv.DictReader(open(ORIG / "18_usable_additions_after_vietnam_partner_review_2026-10-26.csv", encoding="utf-8-sig"))}
    reviews = {r["review_id"]: r for r in csv.DictReader(open(ORIG / "09_affiliation_review_enriched.csv", encoding="utf-8-sig"))}
    corpus = hub_rows()
    used = [r for r in corpus if r["source"] != "inherited_openalex"]
    out = []
    for r in used:
        rid = r["review_id"]
        w, how = find_work(adds[rid], reviews.get(rid, {}))
        cls, raws = classify(w)
        out.append({"review_id": rid, "year": r["year"], "evidence": r["affiliation_evidence"],
                    "class": cls, "found_by": how, "openalex_id": (w or {}).get("id", ""),
                    "openalex_type": (w or {}).get("type", ""),
                    "n_authorships": len((w or {}).get("authorships", [])),
                    "hub_like_strings": " | ".join(sorted({s for s in raws if is_hub_string(s)}))[:300],
                    "raw_strings_sample": " | ".join(sorted(set(raws)))[:300],
                    "title": r["title"][:150]})
        print(rid, cls, how, flush=True)
    write_csv("coverage_reasons.csv", out)
    cnt = Counter(x["class"] for x in out)
    by_ev = Counter((x["evidence"], x["class"]) for x in out)
    return {"n": len(out), "classes": dict(cnt), "by_evidence": {f"{a}|{b}": v for (a, b), v in by_ev.items()},
            "found_by": dict(Counter(x["found_by"].split("(")[0] for x in out))}


def unlinked_strings() -> dict:
    corpus_ids = {r["id"] for r in hub_rows()}
    corpus_dois = {r["doi"].lower() for r in hub_rows() if r.get("doi")}
    pats = ['"University of Banking"', '"Banking University"']
    seen = {}
    for p in pats:
        flt = f"raw_affiliation_strings.search:{p},publication_year:2016-2025,institutions.id:!I3129629310"
        for w in cursor_all(flt, select="id,doi,display_name,publication_year,type,language,authorships,is_retracted"):
            seen[w["id"]] = w
    rows = []
    for w in seen.values():
        raws = [s for a in w.get("authorships", []) for s in (a.get("raw_affiliation_strings") or [])]
        hubs = sorted({s for s in raws if is_hub_string(s)})
        if not hubs:
            continue
        doi = re.sub(r"^https?://doi\.org/", "", (w.get("doi") or "").lower())
        rows.append({"openalex_id": w["id"], "year": w.get("publication_year"), "type": w.get("type"),
                     "language": w.get("language"), "retracted": w.get("is_retracted"),
                     "in_corpus": w["id"] in corpus_ids or (doi and doi in corpus_dois),
                     "hub_strings": " | ".join(hubs)[:300], "title": (w.get("display_name") or "")[:150]})
    write_csv("hub_unlinked_raw_strings.csv", rows or [{"openalex_id": ""}])
    eligible = [r for r in rows if r["type"] in ("article", "conference-paper", "review") and not r["retracted"]]
    return {"candidates_searched": len(seen), "with_exact_hub_string": len(rows),
            "eligible_type": len(eligible),
            "eligible_not_in_corpus": sum(not r["in_corpus"] for r in eligible),
            "eligible_in_corpus": sum(bool(r["in_corpus"]) for r in eligible),
            "by_year_not_in_corpus": dict(sorted(Counter(r["year"] for r in eligible if not r["in_corpus"]).items())),
            "language_not_in_corpus": dict(Counter(r["language"] for r in eligible if not r["in_corpus"]))}


if __name__ == "__main__":
    res = {"query_date": __import__("datetime").date.today().isoformat(), "reasons": reasons(),
           "unlinked_hub_strings": unlinked_strings()}
    write_json("coverage_reasons.json", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
