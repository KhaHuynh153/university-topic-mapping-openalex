"""Topic network, corroboration, bridge review, and temporal diagnostics.

Outputs are written only beside this file and use the fixed K=8 model.
Run each stage in order: build, validate, review, trends.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "topic_7_research_plan" / "pilot"
RAW = PILOT / "data" / "raw" / "works"
EXCLUSIONS = PILOT / "data" / "curated" / "prolific_author_exclusions.json"
LABELS = [
    "Education and language",
    "Consumer and services",
    "Bank liquidity and policy",
    "Financial stability / inclusion*",
    "Commercial banks and risk",
    "Fractional mathematics",
    "Growth and trade",
    "Stock markets",
]
SHORT = ["Education", "Consumers", "Liquidity", "Financial*", "Commercial banks", "Mathematics", "Growth/trade", "Stocks"]
THRESHOLDS = (0.15, 0.20, 0.25)
MAIN_THRESHOLD = 0.20
MIN_JOINT = 10
MIN_LIFT = 1.20


def corpus() -> list[dict]:
    with (HERE / "corpus_refreshed.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(name: str, data: object) -> None:
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name: str) -> list[dict]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def edges(theta: np.ndarray, threshold: float) -> list[dict]:
    eligible = theta.sum(axis=1) > 0
    x = theta[eligible]
    n = len(x)
    member = x >= threshold
    support = member.sum(axis=0)
    out = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            joint = int(np.count_nonzero(member[:, i] & member[:, j]))
            expected = float(support[i] * support[j] / n)
            union = int(support[i] + support[j] - joint)
            out.append({"topic_i": i, "topic_j": j, "label_i": LABELS[i], "label_j": LABELS[j],
                        "n": n, "threshold": threshold, "support_i": int(support[i]),
                        "support_j": int(support[j]), "joint_count": joint,
                        "expected_independence": round(expected, 5),
                        "lift": round(joint / expected, 5) if expected else 0.0,
                        "jaccard": round(joint / union, 5) if union else 0.0,
                        "passes_rule": joint >= MIN_JOINT and expected > 0 and joint / expected >= MIN_LIFT})
    return out


def selected_pairs(edge_rows: list[dict]) -> set[tuple[int, int]]:
    return {(int(e["topic_i"]), int(e["topic_j"])) for e in edge_rows
            if e["passes_rule"] is True or str(e["passes_rule"]).lower() == "true"}


def network_svg(selected: list[dict], node_rows: list[dict], name: str = "network_t20.svg",
                subtitle: str | None = None, title: str = "HUB topic co-membership network") -> None:
    positions = {0: (250, 190), 1: (600, 105), 2: (925, 205), 3: (1010, 405),
                 4: (855, 645), 5: (240, 620), 6: (520, 625), 7: (550, 345)}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760">',
             '<rect width="1200" height="760" fill="#ffffff"/>',
             f'<text x="40" y="42" font-family="Arial" font-size="25" font-weight="bold">{escape(title)}</text>',
             f'<text x="40" y="70" font-family="Arial" font-size="14" fill="#444">{escape(subtitle or "1,335 assignable works · NMF K=8 · membership ≥ 0.20 · joint ≥ 10 · lift ≥ 1.20")}</text>']
    for e in selected:
        i, j = int(e["topic_i"]), int(e["topic_j"])
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        width = min(11, 2 + float(e["joint_count"]) / 12)
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2477a3" stroke-opacity="0.55" stroke-width="{width:.1f}"/>')
        mx, my = (x1+x2)/2, (y1+y2)/2
        parts.append(f'<rect x="{mx-28:.1f}" y="{my-12:.1f}" width="56" height="24" rx="7" fill="white" stroke="#b4c5d2"/>')
        parts.append(f'<text x="{mx:.1f}" y="{my+5:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#173c54">{e["joint_count"]} · {float(e["lift"]):.2f}</text>')
    for node in node_rows:
        i = int(node["topic"])
        x, y = positions[i]
        radius = 27 + 55 * math.sqrt(float(node["mean_share"]))
        fill = "#9e765b" if i == 3 else "#285f85"
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius:.1f}" fill="{fill}" stroke="white" stroke-width="3"/>')
        parts.append(f'<text x="{x}" y="{y+6}" text-anchor="middle" font-family="Arial" font-size="19" font-weight="bold" fill="white">{i}</text>')
        parts.append(f'<text x="{x}" y="{y+radius+20:.1f}" text-anchor="middle" font-family="Arial" font-size="15" fill="#222">{escape(SHORT[i])}</text>')
    parts.append('<text x="40" y="728" font-family="Arial" font-size="13" fill="#555">Line labels: joint paper count · prevalence-adjusted lift. Node size: mean topic share. *Mixed, low-coherence topic.</text>')
    parts.append('</svg>')
    (HERE / name).write_text("\n".join(parts), encoding="utf-8")


def build() -> None:
    rows = corpus()
    theta = np.load(HERE / "theta_k8.npy")
    assert theta.shape == (len(rows), 8)
    eligible = theta.sum(axis=1) > 0
    x = theta[eligible]
    member = x >= MAIN_THRESHOLD
    catalog = json.loads((HERE / "topics_k8.json").read_text(encoding="utf-8"))
    nodes = [{"topic": i, "label": LABELS[i], "n_assignable": len(x),
              "mean_share": round(float(x[:, i].mean()), 5),
              "membership_support": int(member[:, i].sum()),
              "n_dominant": int(np.count_nonzero(x.argmax(axis=1) == i)),
              "npmi": round(float(catalog[i]["npmi"]), 5)} for i in range(8)]
    write_csv("network_nodes.csv", nodes,
              ["topic", "label", "n_assignable", "mean_share", "membership_support", "n_dominant", "npmi"])
    all_edges = edges(theta, MAIN_THRESHOLD)
    selected = sorted([e for e in all_edges if e["passes_rule"]],
                      key=lambda e: (-e["lift"], -e["joint_count"]))
    cols = list(all_edges[0])
    write_csv("network_edges_t20.csv", all_edges, cols)
    write_csv("network_selected_edges_t20.csv", selected, cols)
    bridges = []
    for edge in selected:
        i, j = edge["topic_i"], edge["topic_j"]
        ix = [d for d in range(len(rows)) if eligible[d] and theta[d, i] >= MAIN_THRESHOLD
              and theta[d, j] >= MAIN_THRESHOLD]
        ix.sort(key=lambda d: (-float(min(theta[d, i], theta[d, j])),
                               -float(theta[d, i] + theta[d, j]), rows[d]["id"]))
        for rank, d in enumerate(ix[:15], 1):
            r = rows[d]
            bridges.append({"topic_i": i, "topic_j": j, "rank": rank,
                            "id": r["id"], "year": r["year"], "title": r["title"],
                            "source": r["source"], "has_abstract": r["has_abstract"],
                            "affiliation_evidence": r["affiliation_evidence"],
                            "share_i": round(float(theta[d, i]), 5),
                            "share_j": round(float(theta[d, j]), 5),
                            "min_share": round(float(min(theta[d, i], theta[d, j])), 5)})
    write_csv("network_bridge_candidates.csv", bridges,
              ["topic_i", "topic_j", "rank", "id", "year", "title", "source",
               "has_abstract", "affiliation_evidence", "share_i", "share_j", "min_share"])
    network_svg(selected, nodes)
    write_json("network_build_summary.json", {"n_corpus": len(rows), "n_assignable": len(x),
              "n_pairs": len(all_edges), "n_selected": len(selected),
              "selection": {"membership_threshold": MAIN_THRESHOLD, "min_joint": MIN_JOINT,
                            "min_lift": MIN_LIFT}, "selected_edges": selected})
    print(json.dumps({"n_assignable": len(x), "n_selected": len(selected),
                      "selected": [(e["topic_i"], e["topic_j"], e["joint_count"], e["lift"])
                                   for e in selected]}, ensure_ascii=False), flush=True)


def raw_works(ids: set[str]) -> dict[str, dict]:
    found = {}
    for page in sorted(RAW.glob("page_*.json")):
        data = json.loads(page.read_text(encoding="utf-8"))
        for work in data.get("results", []):
            if work.get("id") in ids:
                found[work["id"]] = work
    return found


def reference_heatmap_svg(cosine: np.ndarray) -> None:
    x0, y0, cell = 300, 145, 60
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="850" height="730" viewBox="0 0 850 730">',
             '<rect width="850" height="730" fill="white"/>',
             '<text x="30" y="45" font-family="Arial" font-size="24" font-weight="bold">Shared-reference similarity across HUB topics</text>',
             '<text x="30" y="73" font-family="Arial" font-size="14" fill="#444">Dominant-topic papers only · 955 indexed works with cited references</text>']
    for i, label in enumerate(SHORT):
        parts.append(f'<text x="{x0-12}" y="{y0+i*cell+37}" text-anchor="end" font-family="Arial" font-size="13">{escape(label)}</text>')
        parts.append(f'<text x="{x0+i*cell+cell/2}" y="{y0-12}" text-anchor="middle" font-family="Arial" font-size="13">{i}</text>')
        for j in range(8):
            value = float(cosine[i, j])
            frac = max(0, min(value / .4, 1))
            red = int(247 - 160 * frac)
            green = int(249 - 105 * frac)
            blue = int(252 - 35 * frac)
            parts.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell}" height="{cell}" fill="rgb({red},{green},{blue})" stroke="white"/>')
            parts.append(f'<text x="{x0+j*cell+cell/2}" y="{y0+i*cell+37}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')
    parts.append('<text x="30" y="677" font-family="Arial" font-size="13" fill="#555">Cosine is descriptive; reference coverage is incomplete and varies by period. *Mixed topic.</text>')
    parts.append('</svg>')
    (HERE / "reference_similarity_heatmap.svg").write_text("\n".join(parts), encoding="utf-8")


def reference_layer(rows: list[dict], theta: np.ndarray) -> dict:
    eligible = theta.sum(axis=1) > 0
    works = raw_works({r["id"] for r in rows if r["source"] == "inherited_openalex"})
    refs = [set(works.get(r["id"], {}).get("referenced_works") or []) for r in rows]
    indexed = [i for i in range(len(rows)) if eligible[i] and refs[i]]
    df = Counter(ref for i in indexed for ref in refs[i])
    shared = {ref for ref, count in df.items() if count >= 2}
    columns = {ref: i for i, ref in enumerate(sorted(shared))}
    profiles = np.zeros((8, len(columns)), dtype=np.float64)
    dominant = theta.argmax(axis=1)
    for i in indexed:
        topic = int(dominant[i])
        # A work contributes references to exactly one topic profile. This
        # avoids inflating similarity through the bridge papers themselves.
        for ref in refs[i] & shared:
            profiles[topic, columns[ref]] += math.log((1 + len(indexed)) / (1 + df[ref])) + 1
    norms = np.linalg.norm(profiles, axis=1)
    cosine = profiles @ profiles.T / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    np.save(HERE / "reference_similarity_dominant.npy", cosine)
    reference_heatmap_svg(cosine)
    pairs = [{"topic_i": i, "topic_j": j, "label_i": LABELS[i], "label_j": LABELS[j],
              "reference_cosine": round(float(cosine[i, j]), 6),
              "n_indexed_i": int(sum(dominant[d] == i for d in indexed)),
              "n_indexed_j": int(sum(dominant[d] == j for d in indexed))}
             for i in range(8) for j in range(i+1, 8)]
    pairs.sort(key=lambda r: -r["reference_cosine"])
    for rank, row in enumerate(pairs, 1):
        row["reference_rank_of_28"] = rank
    write_csv("reference_similarity_edges.csv", pairs,
              ["topic_i", "topic_j", "label_i", "label_j", "reference_cosine",
               "n_indexed_i", "n_indexed_j", "reference_rank_of_28"])
    coverage = []
    for start in (2016, 2018, 2020, 2022, 2024):
        ix = [i for i, r in enumerate(rows) if eligible[i] and start <= r["year"] <= start + 1]
        coverage.append({"period": f"{start}-{start+1}", "n_assignable": len(ix),
                         "n_raw_found": sum(rows[i]["id"] in works for i in ix),
                         "n_with_references": sum(bool(refs[i]) for i in ix),
                         "fraction_with_references": round(sum(bool(refs[i]) for i in ix) / len(ix), 5)})
    write_csv("reference_coverage_by_period.csv", coverage,
              ["period", "n_assignable", "n_raw_found", "n_with_references", "fraction_with_references"])
    return {"n_raw_found": len(works), "n_with_references": len(indexed),
            "n_distinct_cited_works": len(df), "n_cited_by_two_or_more": len(shared),
            "coverage": coverage, "top_pairs": pairs[:10], "all_pairs": pairs}


def validate() -> None:
    rows = corpus()
    main = np.load(HERE / "theta_k8.npy")
    alt = json.loads((HERE / "sensitivity_k8.json").read_text(encoding="utf-8"))
    alt_by_name = {r["scenario"]: r for r in alt}
    drop = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    scenarios = {"main": (main, rows, [1.0] * 8),
                 "title_only_refit": (np.load(HERE / "theta_title_only_aligned_k8.npy"), rows,
                                      alt_by_name["title_only"]["aligned_cosine"]),
                 "abstract_subset_refit": (np.load(HERE / "theta_abstract_subset_aligned_k8.npy"),
                                            [r for r in rows if r["has_abstract"]],
                                            alt_by_name["abstract_subset"]["aligned_cosine"]),
                 "inherited_only_refit": (np.load(HERE / "theta_inherited_only_aligned_k8.npy"),
                                           [r for r in rows if r["source"] == "inherited_openalex"],
                                           alt_by_name["inherited_only"]["aligned_cosine"])}
    for n in (5, 10):
        removed = set(drop[str(n)]["paper_ids"])
        indices = [i for i, r in enumerate(rows) if r["id"] not in removed]
        scenarios[f"exclude_top_{n}_fixed_basis"] = (main[indices], [rows[i] for i in indices], [1.0] * 8)
    strong_indices = [i for i, r in enumerate(rows) if r["affiliation_evidence"] == "strong"]
    scenarios["strong_affiliation_fixed_basis"] = (main[strong_indices],
                                                    [rows[i] for i in strong_indices], [1.0] * 8)
    records = []
    for name, (theta, source_rows, match_cos) in scenarios.items():
        assert theta.shape == (len(source_rows), 8), name
        for threshold in THRESHOLDS:
            for e in edges(theta, threshold):
                i, j = e["topic_i"], e["topic_j"]
                records.append({"scenario": name, **e,
                                "cosine_i": round(float(match_cos[i]), 5),
                                "cosine_j": round(float(match_cos[j]), 5),
                                "semantic_match": min(match_cos[i], match_cos[j]) >= 0.8,
                                "passes_with_semantic_match": e["passes_rule"] and
                                min(match_cos[i], match_cos[j]) >= 0.8})
    write_csv("network_sensitivity_edges.csv", records,
              ["scenario", "topic_i", "topic_j", "label_i", "label_j", "n", "threshold",
               "support_i", "support_j", "joint_count", "expected_independence", "lift",
               "jaccard", "passes_rule", "cosine_i", "cosine_j", "semantic_match",
               "passes_with_semantic_match"])
    selected = selected_pairs(read_csv("network_selected_edges_t20.csv"))
    ref = reference_layer(rows, main)
    ref_pairs = {(r["topic_i"], r["topic_j"]): r for r in ref["all_pairs"]}
    summary = []
    for i, j in sorted(selected):
        relevant = [r for r in records if (r["topic_i"], r["topic_j"]) == (i, j)]
        selected_main = next(r for r in relevant if r["scenario"] == "main" and r["threshold"] == .20)
        checks = {}
        for name in scenarios:
            scenario_rows = [r for r in relevant if r["scenario"] == name]
            checks[name] = {"thresholds_passing": [r["threshold"] for r in scenario_rows if r["passes_rule"]],
                            "main_threshold_joint": next(r["joint_count"] for r in scenario_rows if r["threshold"] == .20),
                            "main_threshold_lift": next(r["lift"] for r in scenario_rows if r["threshold"] == .20),
                            "main_threshold_semantic_pass": next(r["passes_with_semantic_match"] for r in scenario_rows if r["threshold"] == .20)}
        rp = ref_pairs[(i, j)]
        summary.append({"topic_i": i, "topic_j": j, "main_joint": selected_main["joint_count"],
                        "main_lift": selected_main["lift"], "checks": checks,
                        "reference_cosine": rp["reference_cosine"],
                        "reference_rank_of_28": rp["reference_rank_of_28"]})
    write_json("network_validation_summary.json",
               {"selection_rule": {"thresholds": THRESHOLDS, "min_joint": MIN_JOINT,
                                   "min_lift": MIN_LIFT, "semantic_match_min_cosine": 0.8},
                "reference_method": "Reference profiles from distinct dominant-topic indexed papers; each work contributes to one profile; cited work DF>=2; TF-IDF-like weighting; cosine",
                "reference_coverage": {k: v for k, v in ref.items() if k not in ("all_pairs", "top_pairs")},
                "reference_top_pairs": ref["top_pairs"], "selected_edge_checks": summary})
    print(json.dumps({"reference_n": ref["n_with_references"],
                      "selected": [(r["topic_i"], r["topic_j"], r["checks"],
                                    r["reference_cosine"], r["reference_rank_of_28"])
                                   for r in summary]}, ensure_ascii=False), flush=True)


def normalized_title(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


MANUAL_TOP15 = {
    1: ("substantive", "Bank capital response to policy uncertainty links prudential capital and bank behavior."),
    2: ("substantive", "Liquidity after bank runs is explicitly related to capital, size and ownership."),
    3: ("substantive", "Banking uncertainty is analysed with credit risk, capitalization and liquidity."),
    4: ("substantive", "Sovereign-bond holdings are tied to liquidity reserves, lending and risk."),
    5: ("generic_bank_overlap", "Earnings management and dividend policy do not visibly establish the liquidity/policy side."),
    6: ("substantive", "Liquidity position is examined against bank credit risk and profitability."),
    7: ("duplicate_version", "Same title, year and abstract as rank 6 with a different DOI/version."),
    8: ("insufficient_text", "Title-only bank efficiency and non-performing loans; no direct liquidity/policy evidence."),
    9: ("substantive", "Bank earnings opacity is examined against funding sources."),
    10: ("substantive", "Bank liquidity is directly linked to lending growth."),
    11: ("substantive", "Loan growth is modelled against CAMELS factors, including liquidity in the abstract."),
    12: ("substantive", "Commercial-bank profitability includes liquidity and credit-loss provisions."),
    13: ("generic_bank_overlap", "Credit risk and stability are explicit; liquidity/policy is not clearly supported in the available abstract."),
    14: ("generic_bank_overlap", "Equity risk-return tradeoff lacks clear liquidity/policy content in the available abstract."),
    15: ("substantive", "Bank capital structure is related to liquidity creation in the abstract."),
}
MANUAL_TAIL_SAMPLE = {
    18: ("generic_bank_overlap", "Diversification and bank risk are studied without clear liquidity/policy content."),
    21: ("generic_bank_overlap", "Bank capital, profitability and risk are explicit; liquidity/policy is not."),
    24: ("insufficient_text", "Title-only credit-growth and net-interest-margin record."),
    27: ("substantive", "Natural disasters, deposits/liquidity and credit/default risk are jointly examined."),
    30: ("substantive", "Basel III stable funding ratio is analysed against bank risk and return."),
    33: ("generic_bank_overlap", "Dividend policy and leverage do not establish the liquidity/policy construct."),
    36: ("insufficient_text", "Title-only digital-banking/customer-experience record."),
    39: ("generic_bank_overlap", "CEO characteristics and stability lack direct liquidity/policy content."),
    42: ("substantive", "Monetary-policy transmission is analysed through commercial-bank risk."),
    45: ("substantive", "Liquidity creation is directly linked to bank risk taking."),
    48: ("substantive", "Bank balance-sheet/finance response to uncertainty links funding and risk behavior."),
    51: ("generic_bank_overlap", "Ownership and performance lack direct liquidity/policy content."),
}


def review() -> None:
    rows = corpus()
    theta = np.load(HERE / "theta_k8.npy")
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[(r["year"], normalized_title(r["title"]))].append(i)
    duplicate_groups = [v for v in grouped.values() if len(v) > 1]
    duplicate_rows = []
    keep = np.ones(len(rows), dtype=bool)
    for group_no, group in enumerate(duplicate_groups, 1):
        chosen = max(group, key=lambda i: (bool(rows[i]["has_abstract"]),
                                           len(rows[i]["text"]), rows[i]["id"]))
        for i in group:
            if i != chosen:
                keep[i] = False
            duplicate_rows.append({"group": group_no, "year": rows[i]["year"],
                                   "normalized_title": normalized_title(rows[i]["title"]),
                                   "id": rows[i]["id"], "doi": rows[i]["doi"],
                                   "title": rows[i]["title"], "has_abstract": rows[i]["has_abstract"],
                                   "retained_in_sensitivity": i == chosen,
                                   "classification": "same_title_year_possible_version; not automatically adjudicated as same intellectual work"})
    write_csv("possible_duplicate_versions.csv", duplicate_rows,
              ["group", "year", "normalized_title", "id", "doi", "title", "has_abstract",
               "retained_in_sensitivity", "classification"])
    dedup_edges = edges(theta[keep], MAIN_THRESHOLD)
    write_csv("network_dedup_sensitivity.csv", dedup_edges, list(dedup_edges[0]))

    left_terms = ["liquidity", "funding", "deposit", "uncertainty", "monetary", "bank run",
                  "bank-run", "sovereign bond", "financial integration"]
    right_terms = ["credit", "capital", "risk", "lending", "loan", "commercial bank",
                   "profitability", "bank performance"]
    index = {(r["id"]): i for i, r in enumerate(rows)}
    ix = [i for i, r in enumerate(rows) if theta[i, 2] >= MAIN_THRESHOLD and
          theta[i, 4] >= MAIN_THRESHOLD and theta[i].sum() > 0]
    ix.sort(key=lambda i: (-float(min(theta[i, 2], theta[i, 4])),
                            -float(theta[i, 2] + theta[i, 4]), rows[i]["id"]))
    bridge_rows = []
    for rank, i in enumerate(ix, 1):
        r = rows[i]
        content = f"{r['title']} {r['abstract']}".lower()
        left = [term for term in left_terms if term in content]
        right = [term for term in right_terms if term in content]
        manual_status, note = {**MANUAL_TOP15, **MANUAL_TAIL_SAMPLE}.get(rank, ("not_manually_reviewed", ""))
        bridge_rows.append({"rank": rank, "id": r["id"], "year": r["year"],
                            "title": r["title"], "doi": r["doi"], "has_abstract": r["has_abstract"],
                            "affiliation_evidence": r["affiliation_evidence"],
                            "share_liquidity": round(float(theta[i, 2]), 5),
                            "share_commercial_bank": round(float(theta[i, 4]), 5),
                            "left_terms_in_text": " | ".join(left),
                            "right_terms_in_text": " | ".join(right),
                            "both_vocabularies_present": bool(left and right),
                            "manual_status": manual_status, "review_note": note})
    write_csv("bridge_review_2_4.csv", bridge_rows,
              ["rank", "id", "year", "title", "doi", "has_abstract", "affiliation_evidence",
               "share_liquidity", "share_commercial_bank", "left_terms_in_text",
               "right_terms_in_text", "both_vocabularies_present", "manual_status", "review_note"])

    topics = json.loads((HERE / "topics_k8.json").read_text(encoding="utf-8"))
    label_notes = {
        0: "Top terms and representative papers concern teaching, students and language; broad education label supported.",
        1: "Brand, customer and service papers support the label; domain is broad.",
        2: "Liquidity, monetary policy and uncertainty papers support the label; absent in abstract-only refit.",
        3: "Financial inclusion, development and stability are mixed; NPMI 0.130; no independent field claim.",
        4: "Commercial-bank credit, capital and risk papers support the label.",
        5: "Fractional/fuzzy differential-equation papers consistently support the label.",
        6: "Growth, trade, ASEAN and FDI papers support a broad development label.",
        7: "Stock-market and corporate-finance papers support the label.",
    }
    labels = [{"topic": t["topic"], "model_label": LABELS[t["topic"]],
               "top_terms": " | ".join(t["top_terms"][:10]),
               "sampled_representative_titles": " | ".join(w["title"] for w in t["representative_works"][:5]),
               "review_note": label_notes[t["topic"]],
               "status": "mixed_do_not_interpret_as_independent_field" if t["topic"] == 3 else
                         "provisionally_supported_single_analyst_review",
               "external_expert_validation": "pending"} for t in topics]
    write_csv("topic_label_review_k8.csv", labels,
              ["topic", "model_label", "top_terms", "sampled_representative_titles",
               "review_note", "status", "external_expert_validation"])
    before = {(e["topic_i"], e["topic_j"]): e for e in edges(theta, MAIN_THRESHOLD)}
    after = {(e["topic_i"], e["topic_j"]): e for e in dedup_edges}
    write_json("bridge_review_summary.json", {"n_title_year_duplicate_groups": len(duplicate_groups),
               "n_records_removed_sensitivity": int((~keep).sum()),
               "n_bridge_2_4": len(bridge_rows),
               "n_manually_reviewed": len(MANUAL_TOP15) + len(MANUAL_TAIL_SAMPLE),
               "n_manually_substantive": sum(x[0] == "substantive" for x in
                                             list(MANUAL_TOP15.values()) + list(MANUAL_TAIL_SAMPLE.values())),
               "n_top15_substantive": sum(x[0] == "substantive" for x in MANUAL_TOP15.values()),
               "n_all_bridge_both_vocabularies_present": sum(r["both_vocabularies_present"] for r in bridge_rows),
               "edge_before_after": [{"pair": [i, j],
                                      "before_joint": before[(i, j)]["joint_count"],
                                      "after_joint": after[(i, j)]["joint_count"],
                                      "before_lift": before[(i, j)]["lift"],
                                      "after_lift": after[(i, j)]["lift"],
                                      "after_passes_rule": after[(i, j)]["passes_rule"]}
                                     for i, j in ((2, 4), (3, 4))]})
    main_edge = next(e for e in edges(theta, MAIN_THRESHOLD) if
                     (e["topic_i"], e["topic_j"]) == (2, 4))
    network_svg([main_edge], read_csv("network_nodes.csv"),
                name="network_primary_conditional.svg",
                subtitle="K=8 topic map · one content-supported edge · conditional on text completeness")
    print(json.dumps(json.loads((HERE / "bridge_review_summary.json").read_text(encoding="utf-8")),
                     ensure_ascii=False), flush=True)


HUB_INSTITUTION = "https://openalex.org/I3129629310"


def temporal_contrast(theta: np.ndarray, rows: list[dict]) -> dict:
    assert theta.shape == (len(rows), 8)
    eligible = theta.sum(axis=1) > 0
    years = np.array([r["year"] for r in rows])
    early = eligible & (years >= 2016) & (years <= 2019)
    late = eligible & (years >= 2022) & (years <= 2025)
    early_mean = theta[early].mean(axis=0)
    late_mean = theta[late].mean(axis=0)
    bin_means = []
    for start in (2016, 2018, 2020, 2022, 2024):
        ix = eligible & (years >= start) & (years <= start + 1)
        bin_means.append(theta[ix].mean(axis=0))
    return {"n_early": int(early.sum()), "n_late": int(late.sum()),
            "early_mean": early_mean, "late_mean": late_mean,
            "article_delta": late_mean - early_mean,
            "equal_bin_delta": np.mean(bin_means[3:5], axis=0) - np.mean(bin_means[0:2], axis=0),
            "bin_means": np.array(bin_means)}


def author_leader_clusters(rows: list[dict], bundle_additions: bool) -> tuple[list[str], dict]:
    works = raw_works({r["id"] for r in rows if r["source"] == "inherited_openalex"})
    by_work = []
    for r in rows:
        if r["source"] != "inherited_openalex":
            by_work.append([])
            continue
        authors = []
        for authorship in works.get(r["id"], {}).get("authorships", []):
            if any(inst.get("id") == HUB_INSTITUTION for inst in authorship.get("institutions", [])):
                aid = authorship.get("author", {}).get("id")
                if aid:
                    authors.append(aid)
        by_work.append(sorted(set(authors)))
    counts = Counter(a for authors in by_work for a in authors)
    labels = []
    no_tagged_author = 0
    for i, (r, authors) in enumerate(zip(rows, by_work)):
        if r["source"] != "inherited_openalex":
            labels.append("ADDED_ALL" if bundle_additions else "ADDED:" + r["id"])
        elif authors:
            labels.append("AUTHOR:" + max(authors, key=lambda a: (counts[a], a)))
        else:
            no_tagged_author += 1
            labels.append("UNTAGGED:" + r["id"])
    sizes = Counter(labels)
    return labels, {"n_clusters": len(sizes), "max_cluster_size": max(sizes.values()),
                    "n_inherited_without_tagged_hub_author": no_tagged_author,
                    "n_added_records": sum(r["source"] != "inherited_openalex" for r in rows)}


def cluster_bootstrap(theta: np.ndarray, rows: list[dict], bundle_additions: bool,
                      replicates: int = 2000, seed: int = 20261027) -> dict:
    eligible = theta.sum(axis=1) > 0
    active_rows = [r for i, r in enumerate(rows) if eligible[i]]
    x = theta[eligible]
    labels, metadata = author_leader_clusters(active_rows, bundle_additions)
    names = {name: i for i, name in enumerate(sorted(set(labels)))}
    cluster_idx = np.array([names[name] for name in labels], dtype=int)
    years = np.array([r["year"] for r in active_rows])
    early = years <= 2019
    late = years >= 2022
    rng = np.random.default_rng(seed + int(bundle_additions))
    sims = np.empty((replicates, 8), dtype=np.float64)
    for b in range(replicates):
        draws = rng.integers(0, len(names), size=len(names))
        weights = np.bincount(draws, minlength=len(names))[cluster_idx]
        ew, lw = weights[early], weights[late]
        if ew.sum() == 0 or lw.sum() == 0:
            sims[b] = np.nan
        else:
            sims[b] = np.average(x[late], axis=0, weights=lw) - np.average(x[early], axis=0, weights=ew)
    assert np.isfinite(sims).all()
    low, high = np.quantile(sims, [.025, .975], axis=0)
    return {"design": "all_added_one_cluster" if bundle_additions else "added_singletons",
            "replicates": replicates, **metadata,
            "ci_low": [float(v) for v in low], "ci_high": [float(v) for v in high]}


def trend_svg(bin_means: np.ndarray) -> None:
    x0, y0, cellw, cellh = 330, 130, 130, 55
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="680" viewBox="0 0 1040 680">',
             '<rect width="1040" height="680" fill="white"/>',
             '<text x="30" y="45" font-family="Arial" font-size="25" font-weight="bold">HUB topic shares by two-year period</text>',
             '<text x="30" y="72" font-family="Arial" font-size="14" fill="#555">Mean fractional membership per assignable publication · 2016–2025 · K=8</text>']
    for b, start in enumerate((2016, 2018, 2020, 2022, 2024)):
        parts.append(f'<text x="{x0+b*cellw+cellw/2}" y="{y0-14}" text-anchor="middle" font-family="Arial" font-size="14">{start}–{start+1}</text>')
    for k, label in enumerate(SHORT):
        parts.append(f'<text x="{x0-12}" y="{y0+k*cellh+34}" text-anchor="end" font-family="Arial" font-size="14">{escape(label)}</text>')
        for b in range(5):
            v = float(bin_means[b, k])
            frac = max(0, min(v / .28, 1))
            red = int(245 - 140 * frac)
            green = int(248 - 95 * frac)
            blue = int(252 - 35 * frac)
            parts.append(f'<rect x="{x0+b*cellw}" y="{y0+k*cellh}" width="{cellw}" height="{cellh}" fill="rgb({red},{green},{blue})" stroke="white"/>')
            parts.append(f'<text x="{x0+b*cellw+cellw/2}" y="{y0+k*cellh+34}" text-anchor="middle" font-family="Arial" font-size="15">{v:.3f}</text>')
    parts.append('<text x="30" y="625" font-family="Arial" font-size="13" fill="#555">* Financial stability/inclusion is a mixed low-coherence topic. Shading is descriptive, not inferential.</text>')
    parts.append('</svg>')
    (HERE / "temporal_topic_heatmap.svg").write_text("\n".join(parts), encoding="utf-8")


def trends() -> None:
    rows = corpus()
    main = np.load(HERE / "theta_k8.npy")
    alt = {r["scenario"]: r for r in json.loads((HERE / "sensitivity_k8.json").read_text(encoding="utf-8"))}
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    duplicate_rows = read_csv("possible_duplicate_versions.csv")
    remove_ids = {r["id"] for r in duplicate_rows if r["retained_in_sensitivity"].lower() == "false"}
    scenarios = {"main": (main, rows, [1.0]*8),
                 "dedup_title_year_fixed_basis": (main[[i for i, r in enumerate(rows) if r["id"] not in remove_ids]],
                                                   [r for r in rows if r["id"] not in remove_ids], [1.0]*8),
                 "strong_affiliation_fixed_basis": (main[[i for i, r in enumerate(rows) if r["affiliation_evidence"] == "strong"]],
                                                    [r for r in rows if r["affiliation_evidence"] == "strong"], [1.0]*8),
                 "title_only_refit": (np.load(HERE / "theta_title_only_aligned_k8.npy"), rows,
                                      alt["title_only"]["aligned_cosine"]),
                 "abstract_subset_refit": (np.load(HERE / "theta_abstract_subset_aligned_k8.npy"),
                                            [r for r in rows if r["has_abstract"]],
                                            alt["abstract_subset"]["aligned_cosine"]),
                 "inherited_only_refit": (np.load(HERE / "theta_inherited_only_aligned_k8.npy"),
                                           [r for r in rows if r["source"] == "inherited_openalex"],
                                           alt["inherited_only"]["aligned_cosine"])}
    for top in (5, 10):
        drop = set(exclusions[str(top)]["paper_ids"])
        ix = [i for i, r in enumerate(rows) if r["id"] not in drop]
        scenarios[f"exclude_top_{top}_fixed_basis"] = (main[ix], [rows[i] for i in ix], [1.0]*8)
    results = []
    main_contrast = None
    for name, (theta, subrows, cosine) in scenarios.items():
        contrast = temporal_contrast(theta, subrows)
        if name == "main":
            main_contrast = contrast
        for k in range(8):
            results.append({"scenario": name, "topic": k, "label": LABELS[k],
                            "n_early": contrast["n_early"], "n_late": contrast["n_late"],
                            "component_cosine": round(float(cosine[k]), 5),
                            "early_share": round(float(contrast["early_mean"][k]), 6),
                            "late_share": round(float(contrast["late_mean"][k]), 6),
                            "article_delta": round(float(contrast["article_delta"][k]), 6),
                            "equal_bin_delta": round(float(contrast["equal_bin_delta"][k]), 6)})
    write_csv("temporal_sensitivity_k8.csv", results,
              ["scenario", "topic", "label", "n_early", "n_late", "component_cosine",
               "early_share", "late_share", "article_delta", "equal_bin_delta"])
    bootstrap = [cluster_bootstrap(main, rows, False), cluster_bootstrap(main, rows, True)]
    write_json("temporal_cluster_bootstrap.json", {"estimand": "mean publication topic share 2022–2025 minus 2016–2019",
               "cluster_rule": "Each inherited indexed work assigned to its most frequent HUB-tagged author ID; added records either singletons or one conservative cluster; coauthor overlap not fully captured",
               "bootstrap": bootstrap,
               "main_article_delta": [float(v) for v in main_contrast["article_delta"]]})
    trend_svg(main_contrast["bin_means"])
    print(json.dumps({"main_n": [main_contrast["n_early"], main_contrast["n_late"]],
                      "main_delta": [round(float(v), 4) for v in main_contrast["article_delta"]],
                      "ci_education": [(round(b["ci_low"][0], 4), round(b["ci_high"][0], 4)) for b in bootstrap],
                      "ci_math": [(round(b["ci_low"][5], 4), round(b["ci_high"][5], 4)) for b in bootstrap],
                      "clusters": [(b["design"], b["n_clusters"], b["max_cluster_size"]) for b in bootstrap]},
                     ensure_ascii=False), flush=True)


def visualize() -> None:
    edge_rows = read_csv("network_edges_t20.csv")
    by_pair = {(int(r["topic_i"]), int(r["topic_j"])): r for r in edge_rows}
    assert len(by_pair) == 28
    nodes = read_csv("network_nodes.csv")
    screened = by_pair[(2, 4)]
    network_svg([screened], nodes, name="network_screened_links.svg",
                title="Topic links after conservative screening",
                subtitle="One conditional content-supported link; all 28 pairs appear in the companion matrix")

    x0, y0, cell = 380, 170, 67
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="800" viewBox="0 0 1060 800">',
             '<rect width="1060" height="800" fill="#ffffff"/>',
             '<text x="32" y="42" font-family="Arial" font-size="25" font-weight="bold">All 28 HUB topic pairs</text>',
             '<text x="32" y="70" font-family="Arial" font-size="14" fill="#444">Shared-paper count (top) and prevalence-adjusted lift (bottom), membership ≥ 0.20</text>']
    for k, label in enumerate(SHORT):
        parts.append(f'<text x="{x0-12}" y="{y0+k*cell+39}" text-anchor="end" font-family="Arial" font-size="14" fill="#222">{k} {escape(label)}</text>')
        parts.append(f'<text x="{x0+k*cell+cell/2}" y="{y0-12}" text-anchor="middle" font-family="Arial" font-size="14" fill="#222">{k}</text>')
    for i in range(8):
        for j in range(8):
            x, y = x0 + j*cell, y0 + i*cell
            if j < i:
                fill, border = "#f6f7f8", "#e9edf0"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{border}"/>')
            elif i == j:
                fill = "#e4ebef"
                support = next(int(r["membership_support"]) for r in nodes if int(r["topic"]) == i)
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"/>')
                parts.append(f'<text x="{x+cell/2}" y="{y+32}" text-anchor="middle" font-family="Arial" font-size="12" fill="#254254">topic {i}</text>')
                parts.append(f'<text x="{x+cell/2}" y="{y+50}" text-anchor="middle" font-family="Arial" font-size="13" fill="#254254">n={support}</text>')
            else:
                e = by_pair[(i, j)]
                joint, lift = int(e["joint_count"]), float(e["lift"])
                if (i, j) == (2, 4):
                    fill, fg, border, stroke_width = "#167b89", "white", "#075669", 3
                elif (i, j) == (3, 4):
                    fill, fg, border, stroke_width = "#e6c58c", "#513918", "#b17f36", 2
                elif lift >= 1:
                    fill, fg, border, stroke_width = "#d8ebf2", "#253c4b", "white", 1
                else:
                    fill, fg, border, stroke_width = "#edf0f2", "#42505a", "white", 1
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{border}" stroke-width="{stroke_width}"/>')
                parts.append(f'<text x="{x+cell/2}" y="{y+29}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="bold" fill="{fg}">{joint}</text>')
                parts.append(f'<text x="{x+cell/2}" y="{y+50}" text-anchor="middle" font-family="Arial" font-size="13" fill="{fg}">{lift:.2f}</text>')
    parts.append('<rect x="32" y="720" width="16" height="16" fill="#167b89"/><text x="56" y="733" font-family="Arial" font-size="12" fill="#333">conditional content-supported link</text>')
    parts.append('<rect x="330" y="720" width="16" height="16" fill="#e6c58c"/><text x="354" y="733" font-family="Arial" font-size="12" fill="#333">numeric pass, mixed topic excluded from main interpretation</text>')
    parts.append('<text x="32" y="763" font-family="Arial" font-size="13" fill="#555">An empty line in the screened graph does not mean zero co-membership; see each pair above.</text>')
    parts.append('</svg>')
    (HERE / "network_all_28_pairs_matrix.svg").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps({"matrix_pairs": len(by_pair), "screened_edges": 1,
                      "files": ["network_screened_links.svg", "network_all_28_pairs_matrix.svg"]}), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    if stage == "build":
        build()
    elif stage == "validate":
        validate()
    elif stage == "review":
        review()
    elif stage == "trends":
        trends()
    elif stage == "visualize":
        visualize()
    else:
        raise SystemExit("Use build, validate, review, trends, or visualize")
