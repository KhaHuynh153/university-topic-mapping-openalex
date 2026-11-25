"""R1 step 2a: list-based OpenAlex coverage, same automatic procedure for HUB and HVNH.

For each item in a university's own publication list:
  1. match its title against every OpenAlex record linked to the university ID
     (2016-2025, all work types); similarity >= 0.90 after lower-casing and
     removing punctuation -> "covered";
  2. otherwise search the whole of OpenAlex by title (similarity >= 0.90,
     publication year within +-1 of the list year) and classify why it is not linked:
       A not found in OpenAlex
       B found, no raw affiliation string at all
       C found, raw strings present but none names the university
       D found, a raw string names the university but the record is not linked.
No manual verification (unlike the HUB main corpus).

Outputs: out/list_coverage_<school>.csv, out/list_coverage.json
"""
from __future__ import annotations

import csv
import html
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from openalex_client import cursor_all, get
from r1_common import BASE, OUT, write_csv, write_json

R1 = Path(__file__).resolve().parent.parent
HUB_HTML = BASE.parent / "topic_7_research_plan" / "pilot" / "data" / "raw" / "official" / "hub_international_2020_2024.html"
SCHOOLS = {
    "HUB": {"id": "I3129629310",
            "name": re.compile(r"banking university|university of banking|dai hoc ngan hang"),
            "not": re.compile(r"banking academy|academy of banking|hoc vien ngan hang")},
    "HVNH": {"id": "I4403386521",
             "name": re.compile(r"banking academy|academy of banking|hoc vien ngan hang|banking acad"),
             "not": re.compile(r"ukrain|uzbek|sumy|arab academy|sudan|siberian|kazakh|russia|moscow")},
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def clean(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def hub_list() -> list[dict]:
    t = HUB_HTML.read_text(encoding="utf-8", errors="ignore")
    year, out = None, []
    for row in re.findall(r"<tr.*?</tr>", t, flags=re.S | re.I):
        cells = [clean(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)]
        if len(cells) == 1 and re.search(r"NĂM\s+(20\d\d)", cells[0], re.I):
            year = int(re.search(r"(20\d\d)", cells[0]).group(1))
            continue
        if len(cells) >= 2 and re.fullmatch(r"\d{1,4}\.?", cells[0]) and year:
            out.append({"list_year": year, "title": cells[1], "journal": cells[3] if len(cells) > 3 else ""})
    return out


def hvnh_list() -> list[dict]:
    with (R1 / "hvnh" / "hvnh_list.csv").open(encoding="utf-8-sig") as h:
        return [{"list_year": int(r["list_year"]), "title": r["title"], "journal": r.get("journal", "")}
                for r in csv.DictReader(h)]


def best_match(title: str, candidates: list[tuple[str, dict]]) -> tuple[float, dict | None]:
    nt = norm(title)
    best, score = None, 0.0
    for nc, w in candidates:
        sm = SequenceMatcher(None, nt, nc)
        if sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9:
            continue
        s = sm.ratio()
        if s > score:
            best, score = w, s
    return score, best


def classify(work: dict | None, school: dict) -> tuple[str, str]:
    if not work:
        return "A_not_in_openalex", ""
    raws = [s for a in work.get("authorships", []) for s in (a.get("raw_affiliation_strings") or [])]
    if not raws:
        return "B_no_affiliation_string", ""
    named = [s for s in raws if school["name"].search(norm(s)) and not school["not"].search(norm(s))]
    if named:
        return "D_name_string_not_linked", " | ".join(sorted(set(named)))[:250]
    return "C_strings_without_name", " | ".join(sorted(set(raws)))[:250]


def run(school_key: str, items: list[dict]) -> dict:
    school = SCHOOLS[school_key]
    linked = cursor_all(f"institutions.id:{school['id']},publication_year:2016-2025",
                        select="id,display_name,publication_year,type,doi")
    cands = [(norm(w.get("display_name") or ""), w) for w in linked if w.get("display_name")]
    rows = []
    for it in items:
        score, w = best_match(it["title"], cands)
        rec = {"school": school_key, "list_year": it["list_year"], "title": it["title"][:200],
               "journal": it.get("journal", "")[:100]}
        if w is not None:
            rec.update(status="covered", similarity=round(score, 3), openalex_id=w["id"],
                       openalex_type=w.get("type"), openalex_year=w.get("publication_year"), detail="")
        else:
            q = re.sub(r"[^\w\s]", " ", it["title"])[:200].strip()
            data = get("/works", {"filter": f"title.search:{q}", "per-page": 10,
                                  "select": "id,display_name,publication_year,type"}) if q else {}
            found, fs = None, 0.0
            for c in data.get("results", []):
                s = SequenceMatcher(None, norm(it["title"]), norm(c.get("display_name") or "")).ratio()
                if s >= 0.9 and abs((c.get("publication_year") or 0) - it["list_year"]) <= 1 and s > fs:
                    found, fs = c, s
            full = get("/works/" + found["id"].rsplit("/", 1)[-1]) if found else None
            cls, detail = classify(full, school)
            rec.update(status=cls, similarity=round(fs, 3) if found else "",
                       openalex_id=(full or {}).get("id", ""), openalex_type=(full or {}).get("type", ""),
                       openalex_year=(full or {}).get("publication_year", ""), detail=detail)
        rows.append(rec)
    write_csv(f"list_coverage_{school_key}.csv", rows)
    by_year = defaultdict(Counter)
    for r in rows:
        by_year[r["list_year"]][r["status"]] += 1
    total = Counter(r["status"] for r in rows)
    return {"n_linked_records_2016_2025": len(linked), "n_list_items": len(rows), "status": dict(total),
            "covered_share": round(total["covered"] / len(rows), 4),
            "by_year": {y: dict(c) for y, c in sorted(by_year.items())}}


PILOT = BASE.parent / "topic_7_research_plan" / "pilot"


def run_hub_local() -> dict:
    """HUB with the 24/10/2026 snapshot already on disk (no new API calls):
    step 1 against all HUB-linked records 2016-2025 in pilot/data/raw/works;
    step 2 from the title searches of the same date (pilot/runs/official_reconciliation.csv)."""
    linked = []
    for f in sorted((PILOT / "data" / "raw" / "works").glob("page_*.json")):
        linked += [w for w in json.loads(f.read_text(encoding="utf-8")).get("results", [])
                   if 2016 <= (w.get("publication_year") or 0) <= 2025]
    cands = [(norm(w.get("display_name") or ""), w) for w in linked if w.get("display_name")]
    with (PILOT / "runs" / "official_reconciliation.csv").open(encoding="utf-8-sig") as h:
        rec = {norm(r["listed_title"]): r for r in csv.DictReader(h)}
    rows = []
    for it in hub_list():
        score, w = best_match(it["title"], cands)
        out = {"school": "HUB", "list_year": it["list_year"], "title": it["title"][:200], "journal": it.get("journal", "")[:100]}
        if w is not None:
            out.update(status="covered", similarity=round(score, 3), openalex_id=w["id"], openalex_type=w.get("type"),
                       openalex_year=w.get("publication_year"), detail="")
        else:
            r = rec.get(norm(it["title"]))
            if r is None:
                status = "unclassified"
            elif r["match_status"] != "clear_title_match":
                status = "A_not_in_openalex"
            elif r["explicit_hub_raw"] == "True":
                status = "D_name_string_not_linked"
            elif not r["raw_affiliations"].strip(" |"):
                status = "B_no_affiliation_string"
            else:
                status = "C_strings_without_name"
            out.update(status=status, similarity=r["title_similarity"] if r else "", openalex_id=(r or {}).get("openalex_id", ""),
                       openalex_type=(r or {}).get("type", ""), openalex_year=(r or {}).get("openalex_year", ""),
                       detail=((r or {}).get("raw_affiliations", "") or "")[:250])
        rows.append(out)
    write_csv("list_coverage_HUB.csv", rows)
    by_year = defaultdict(Counter)
    for r in rows:
        by_year[r["list_year"]][r["status"]] += 1
    total = Counter(r["status"] for r in rows)
    return {"snapshot": "2026-10-24 (local)", "n_linked_records_2016_2025": len(linked), "n_list_items": len(rows),
            "status": dict(total), "covered_share": round(total["covered"] / len(rows), 4),
            "by_year": {y: dict(c) for y, c in sorted(by_year.items())}}


def main() -> None:
    only = sys.argv[1:] or ["HUB", "HVNH"]
    if only == ["HUB_LOCAL"]:
        res_path = OUT / "list_coverage.json"
        res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
        res["HUB"] = run_hub_local()
        write_json("list_coverage.json", res)
        print(json.dumps(res["HUB"], ensure_ascii=False, indent=1))
        return
    res_path = OUT / "list_coverage.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    res["query_date"] = date.today().isoformat()
    if "HUB" in only:
        res["HUB"] = run("HUB", hub_list())
    if "HVNH" in only:
        items = hvnh_list()
        res["HVNH"] = run("HVNH", items)
        sub = [r for r in csv.DictReader(open(OUT / "list_coverage_HVNH.csv", encoding="utf-8-sig"))
               if 2020 <= int(r["list_year"]) <= 2024]
        c = Counter(r["status"] for r in sub)
        res["HVNH_2020_2024"] = {"n": len(sub), "status": dict(c), "covered_share": round(c["covered"] / len(sub), 4)}
    write_json("list_coverage.json", res)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
