"""R1 step 1d: verified labelling procedure for topic names (no experts, no AI labelling).

Rule (fixed before computing):
  * The name of topic k was set from its 15 highest-weight terms and its 10 papers with the
    highest theta (ranks 1-10, the naming set).
  * Each name is written down with a list of key words that the name stands for (KEYS below).
  * Validation set: papers ranked 11-30 for that topic (not used for naming).
    hit rate = share of validation titles that contain at least one key word.
    A name passes if hit rate >= 70%.
  * Distinctiveness: hit rate of topic k's key words on its own validation set divided by the
    mean hit rate of the same key words on the other topics' validation sets (> 2 expected).
  * Agreement with LDA: share of the 10 highest LDA terms of the matched LDA topic that
    contain a key word.
Outputs: out/label_check_<school>.csv/json
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata

import numpy as np

from r1_common import BASE, CODE, OUT, PAPER_TO_DATA, SHORT_VI, hub_rows, write_csv, write_json

# HUB: data index -> (name in the paper, key words as word prefixes)
HUB_KEYS = {
    6: ("Tăng trưởng kinh tế và thương mại", ["growth", "trade", "export", "import", "fdi", "foreign direct",
                                               "gdp", "macroeconom", "economic development", "inflation"]),
    1: ("Hành vi người tiêu dùng và dịch vụ", ["consumer", "customer", "purchas", "buy", "brand", "satisf",
                                                "loyal", "intention", "service", "tourist", "shopping"]),
    0: ("Giáo dục và học ngôn ngữ", ["student", "learn", "teach", "educat", "english", "language", "school",
                                       "lectur", "classroom", "chinese", "vocabular"]),
    7: ("Thị trường chứng khoán và tài chính doanh nghiệp", ["stock", "share price", "investor", "portfolio",
                                                             "listed", "firm", "corporate", "dividend",
                                                             "capital structure", "return"]),
    4: ("Ngân hàng thương mại: tín dụng và rủi ro", ["bank", "credit", "loan", "lending", "capital adequacy",
                                                     "non performing", "risk"]),
    5: ("Phương trình vi phân phân thứ", ["fractional", "differential", "equation", "fuzzy", "integral",
                                          "derivative", "solution"]),
    2: ("Thanh khoản ngân hàng và chính sách tiền tệ", ["liquidity", "monetary", "uncertainty", "funding",
                                                        "deposit", "interest rate", "bank run"]),
    3: ("Ổn định và bao trùm tài chính (hỗn hợp)", ["stability", "inclusion", "financial development",
                                                    "fragility", "bayesian"]),
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " " + re.sub(r"[^a-z0-9]+", " ", s).strip() + " "


def hit(title: str, keys: list[str]) -> bool:
    t = norm(title)
    return any((" " + k) in t for k in keys)


def check(theta: np.ndarray, rows: list[dict], keys: dict, lda_terms: dict | None, codes: dict) -> list[dict]:
    elig = theta.sum(axis=1) > 0
    idx = np.where(elig)[0]
    val = {}
    for t in keys:
        order = idx[np.argsort(theta[idx, t])[::-1]]
        val[t] = [rows[i]["title"] for i in order[10:30]]
    out = []
    for t, (name, kw) in keys.items():
        own = np.mean([hit(x, kw) for x in val[t]])
        others = np.mean([np.mean([hit(x, kw) for x in val[u]]) for u in keys if u != t])
        rec = {"code": codes[t], "data_index": t, "name": name, "key_words": "; ".join(kw),
               "hit_rate_11_30": round(float(own), 2), "passes_70": bool(own >= 0.70),
               "hit_rate_other_topics": round(float(others), 2),
               "distinctiveness": round(float(own / others), 1) if others > 0 else None}
        if lda_terms is not None:
            terms = lda_terms[t]
            rec["lda_top10_with_key"] = round(float(np.mean([hit(x, kw) for x in terms])), 2)
        rec["validation_titles"] = " || ".join(val[t])
        out.append(rec)
    return out


def hub() -> None:
    rows = hub_rows()
    theta = np.load(BASE / "theta_k8.npy")
    lda_terms = None
    if (OUT / "lda_summary.json").exists():
        lda = json.loads((OUT / "lda_summary.json").read_text(encoding="utf-8"))
        lda_terms = {r["data_index"]: r["lda_top_terms"].split(", ") for r in lda["topics"]}
    res = check(theta, rows, HUB_KEYS, lda_terms, CODE)
    res.sort(key=lambda r: PAPER_TO_DATA.index(r["data_index"]))
    write_csv("label_check_HUB.csv", res)
    write_json("label_check_HUB.json", res)
    for r in res:
        print(r["code"], r["name"], r["hit_rate_11_30"], r["passes_70"], r["hit_rate_other_topics"],
              r["distinctiveness"], r.get("lda_top10_with_key"))


if __name__ == "__main__":
    hub()
