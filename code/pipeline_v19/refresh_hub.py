"""Auditable HUB corpus refresh and NMF stability analysis.

Run from the project's bundled Python with OPENBLAS_NUM_THREADS=1:
    python refresh_hub.py audit
    python refresh_hub.py fit
    python refresh_hub.py sensitivity

Only files in this directory are written. The inherited model module is imported
for identical tokenization, TF-IDF, NMF, and coherence definitions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parent
OLD = MANUSCRIPT / "topic7_original_scope_2026-10-26"
PILOT = MANUSCRIPT / "topic_7_research_plan" / "pilot"
CURATED = PILOT / "data" / "curated"
SOURCE = CURATED / "english_texts_v2.jsonl"
MANIFEST = CURATED / "works_manifest.csv"
ADDITIONS = OLD / "18_usable_additions_after_vietnam_partner_review_2026-10-26.csv"
sys.path.insert(0, str(OLD))
import model_topics as original  # noqa: E402

# Geographic fragments in institution names formed a spurious K=10 topic in
# the first diagnostic fit. Remove those fragments before both unigram and
# bigram construction; preserve the diagnostic outputs in the sibling folder.
PLACE_STOP = {"chi", "minh", "city", "hcmc", "hochiminh"}
BASE_TOKENIZE = original.tokenize


def topic_tokenize(text: str) -> list[str]:
    return [term for term in BASE_TOKENIZE(text)
            if not any(part in PLACE_STOP for part in term.split())]


original.tokenize = topic_tokenize

EXCLUDED = "https://openalex.org/W3025768660"
K_GRID = (5, 8, 10)
SEEDS = tuple(range(10))
FEATURES = 3000
MAX_ITER = 160


def write_json(name: str, data: object) -> None:
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def doi_key(text: str) -> str:
    text = (text or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_rows() -> list[dict]:
    return [json.loads(line) for line in (HERE / "corpus_refreshed.jsonl").read_text(encoding="utf-8").splitlines()]


def audit() -> None:
    inherited = [r for r in original.load_rows()]
    manifest = {r["id"]: r for r in read_csv(MANIFEST)}
    originals_by_doi = defaultdict(list)
    originals_by_title = defaultdict(list)
    for r in inherited:
        originals_by_title[normalize(r["title"])].append(r)
        key = doi_key(manifest.get(r["id"], {}).get("doi", ""))
        if key:
            originals_by_doi[key].append(r)

    additions = read_csv(ADDITIONS)
    assert len(additions) == 125 and all(r["decision"] == "sử dụng" for r in additions)
    corpus = []
    for r in inherited:
        d = dict(r)
        d.update(source="inherited_openalex", doi=doi_key(manifest.get(r["id"], {}).get("doi", "")),
                 affiliation_evidence=manifest.get(r["id"], {}).get("affiliation_evidence", ""),
                 review_id="", primary_evidence_url="", listed_authors="")
        corpus.append(d)

    audit_rows = []
    accepted_title = defaultdict(list)
    accepted_doi = defaultdict(list)
    for a in additions:
        title = (a["matched_title"] or a["listed_title"]).strip()
        year = int(a["matched_year"] or a["listed_year"])
        doi = doi_key(a["doi"])
        key = normalize(title)
        match = None
        reason = ""
        if doi and originals_by_doi.get(doi):
            match = originals_by_doi[doi][0]
            reason = "same_doi_in_inherited"
        elif doi and accepted_doi.get(doi):
            match = accepted_doi[doi][0]
            reason = "same_doi_in_additions"
        elif originals_by_title.get(key):
            match = originals_by_title[key][0]
            reason = "same_title_in_inherited"
        elif accepted_title.get(key):
            match = accepted_title[key][0]
            reason = "same_title_in_additions"
        else:
            # Review close matches before any model fit. A close title alone is
            # not automatically a duplicate: it may be a distinct publication.
            close = [(SequenceMatcher(None, key, normalize(r["title"])).ratio(), r)
                     for r in inherited if abs(year - int(r["year"])) <= 1 and
                     abs(len(key) - len(normalize(r["title"]))) < 30]
            if close:
                sim, possible = max(close, key=lambda pair: pair[0])
                if sim >= 0.88:
                    old_journal = manifest.get(possible["id"], {}).get("primary_location_source", "")
                    new_journal = a["matched_journal"] or a["listed_journal"]
                    if normalize(old_journal) != normalize(new_journal) and year != int(possible["year"]):
                        reason = f"near_title_{sim:.3f}_distinct_journal_year"
                    else:
                        match = possible
                        reason = f"near_title_{sim:.3f}_requires_review"

        if match is not None:
            status = "duplicate" if not reason.endswith("requires_review") else "held_for_review"
            audit_rows.append({"review_id": a["review_id"], "status": status, "reason": reason,
                               "matched_id": match["id"], "listed_year": a["listed_year"],
                               "model_year": year, "doi": doi, "title": title,
                               "evidence_strength": a["evidence_strength"]})
            continue

        has_abstract = a["abstract_available"].lower() == "true" and bool(a["abstract_text"].strip())
        abstract = a["abstract_text"].strip() if has_abstract else ""
        work_id = f"https://doi.org/{doi}" if doi else f"HUB-review:{a['review_id']}"
        row = {"id": work_id, "year": year, "title": title, "abstract": abstract,
               "text": f"{title}. {abstract}".strip(), "has_abstract": has_abstract,
               "source": "verified_official_list_addition", "doi": doi,
               "affiliation_evidence": a["evidence_strength"], "review_id": a["review_id"],
               "primary_evidence_url": a["primary_evidence_url"],
               "listed_authors": a["listed_authors"]}
        corpus.append(row)
        accepted_title[key].append(row)
        if doi:
            accepted_doi[doi].append(row)
        audit_rows.append({"review_id": a["review_id"], "status": "added", "reason": reason or "eligible_verified_record",
                           "matched_id": work_id, "listed_year": a["listed_year"],
                           "model_year": year, "doi": doi, "title": title,
                           "evidence_strength": a["evidence_strength"]})

    assert len({r["id"] for r in corpus}) == len(corpus), "Duplicate corpus IDs"
    assert all(2016 <= int(r["year"]) <= 2025 for r in corpus)
    assert all(r["title"].strip() and r["text"].strip() for r in corpus)
    write_csv("addition_decisions.csv", audit_rows,
              ["review_id", "status", "reason", "matched_id", "listed_year", "model_year",
               "doi", "title", "evidence_strength"])
    with (HERE / "corpus_refreshed.jsonl").open("w", encoding="utf-8") as handle:
        for row in corpus:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = []
    for start in (2016, 2018, 2020, 2022, 2024):
        subset = [r for r in corpus if start <= r["year"] <= start + 1]
        old = [r for r in subset if r["source"] == "inherited_openalex"]
        counts.append({"period": f"{start}-{start+1}", "n_total": len(subset),
                       "n_inherited": len(old), "n_added": len(subset)-len(old),
                       "n_abstract": sum(r["has_abstract"] for r in subset),
                       "n_title_only": sum(not r["has_abstract"] for r in subset),
                       "abstract_rate": round(sum(r["has_abstract"] for r in subset) / len(subset), 4)})
    write_csv("corpus_by_period.csv", counts,
              ["period", "n_total", "n_inherited", "n_added", "n_abstract", "n_title_only", "abstract_rate"])
    summary = {"inherited_n": len(inherited), "reviewed_additions_n": len(additions),
               "decision_counts": dict(Counter(r["status"] for r in audit_rows)),
               "refreshed_n": len(corpus), "with_abstract_n": sum(r["has_abstract"] for r in corpus),
               "title_only_n": sum(not r["has_abstract"] for r in corpus),
               "addition_evidence": dict(Counter(r["affiliation_evidence"] for r in corpus
                                                 if r["source"] != "inherited_openalex")),
               "input_sha256": {str(p): sha256(p) for p in (SOURCE, MANIFEST, ADDITIONS, OLD / "model_topics.py")},
               "periods": counts}
    write_json("corpus_audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, ensure_ascii=False), flush=True)


def align_cosine(reference: np.ndarray, candidate: np.ndarray) -> tuple[list[int], list[float]]:
    a = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    b = candidate / np.maximum(np.linalg.norm(candidate, axis=1, keepdims=True), 1e-12)
    sim = a @ b.T
    n = sim.shape[0]
    # Exact maximum-weight assignment with bitmask dynamic programming (K <= 10).
    dp = {0: (0.0, [])}
    for i in range(n):
        ndp = {}
        for mask, (score, route) in dp.items():
            for j in range(n):
                if not mask & (1 << j):
                    new = mask | (1 << j)
                    val = score + float(sim[i, j])
                    if new not in ndp or val > ndp[new][0]:
                        ndp[new] = (val, route + [j])
        dp = ndp
    order = dp[(1 << n) - 1][1]
    return order, [float(sim[i, j]) for i, j in enumerate(order)]


def fit(ks: tuple[int, ...] = K_GRID) -> None:
    rows = corpus_rows()
    X, terms, B = original.vectorize(rows, "combined", FEATURES)
    eligible = np.linalg.norm(X, axis=1) > 0
    np.save(HERE / "tfidf_refreshed.npy", X)
    write_json("vocabulary.json", terms)
    all_results = []
    catalogs = {}
    for k in ks:
        fits = []
        for seed in SEEDS:
            W, H, error, iterations = original.fit_nmf(X, k, seed, MAX_ITER)
            fits.append((W, H, error, iterations))
            print(f"K={k} seed={seed} error={error:.5f} iterations={iterations}", flush=True)
        chosen = min(range(len(fits)), key=lambda i: fits[i][2])
        W, H, error, iterations = fits[chosen]
        theta = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
        np.save(HERE / f"theta_k{k}.npy", theta)
        np.save(HERE / f"components_k{k}.npy", H)
        coherence = original.npmi(H, B)
        stability = [align_cosine(H, item[1])[1] for item in fits]
        topic_stability = np.mean(np.array(stability), axis=0)
        topics = []
        for t in range(k):
            top = [terms[j] for j in np.argsort(H[t])[::-1][:15]]
            reps = [{"id": rows[i]["id"], "title": rows[i]["title"], "year": rows[i]["year"],
                     "share": round(float(theta[i, t]), 4)}
                    for i in np.argsort(theta[:, t])[::-1][:8]]
            topics.append({"topic": t, "top_terms": top, "npmi": float(coherence[t]),
                           "mean_share": float(theta[eligible, t].mean()),
                           "stability_cosine": float(topic_stability[t]),
                           "representative_works": reps})
        write_json(f"topics_k{k}.json", topics)
        all_results.append({"k": k, "n_documents": len(rows), "n_features": len(terms),
                            "best_seed": chosen, "best_error": float(error),
                            "mean_error": float(np.mean([f[2] for f in fits])),
                            "mean_npmi": float(np.mean(coherence)),
                            "min_npmi": float(np.min(coherence)),
                            "mean_stability_cosine": float(np.mean(stability)),
                            "min_topic_stability_cosine": float(np.min(topic_stability)),
                            "topic_stability": [float(x) for x in topic_stability]})
        catalogs[k] = topics
    existing_path = HERE / "grid_results.json"
    if existing_path.exists():
        prior = json.loads(existing_path.read_text(encoding="utf-8"))["results"]
        all_results = [item for item in prior if item["k"] not in ks] + all_results
        all_results.sort(key=lambda item: item["k"])
    write_json("grid_results.json", {"model": "TF-IDF + NumPy Euclidean NMF; inherited tokenizer plus place stop words",
                                      "features": FEATURES, "max_iter": MAX_ITER,
                                      "seeds": list(SEEDS), "place_stop": sorted(PLACE_STOP),
                                      "results": all_results})
    write_csv("grid_results.csv", all_results,
              ["k", "n_documents", "n_features", "best_seed", "best_error", "mean_error",
               "mean_npmi", "min_npmi", "mean_stability_cosine", "min_topic_stability_cosine",
               "topic_stability"])


def sensitivity(k: int = 8, only: str | None = None) -> None:
    rows = corpus_rows()
    vocabulary = json.loads((HERE / "vocabulary.json").read_text(encoding="utf-8"))
    main_h = np.load(HERE / f"components_k{k}.npy")
    scenarios = {"title_only": (rows, "title"),
                 "abstract_subset": ([r for r in rows if r["has_abstract"]], "combined"),
                 "inherited_only": ([r for r in rows if r["source"] == "inherited_openalex"], "combined")}
    results = []
    for name, (subset, mode) in scenarios.items():
        if only and name != only:
            continue
        # Hold the learned vocabulary fixed so component cosine is meaningful.
        tokenized = [original.tokenize(r["title"] if mode == "title" else r["text"]) for r in subset]
        index = {term: i for i, term in enumerate(vocabulary)}
        X = np.zeros((len(subset), len(vocabulary)), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            for term, count in Counter(tokens).items():
                if term in index:
                    X[i, index[term]] = count
        df = np.count_nonzero(X, axis=0)
        X *= (np.log((1 + len(subset)) / (1 + df)) + 1).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        fits = []
        for seed in range(5):
            W, H, error, iterations = original.fit_nmf(X, k, seed, MAX_ITER)
            fits.append((W, H, error))
            print(f"{name} seed={seed} error={error:.5f}", flush=True)
        W, H, error = min(fits, key=lambda fit: fit[2])
        np.save(HERE / f"components_{name}_k{k}.npy", H)
        order, cosine = align_cosine(main_h, H)
        theta = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
        aligned = theta[:, order]
        np.save(HERE / f"theta_{name}_aligned_k{k}.npy", aligned)
        topic_terms = [[vocabulary[j] for j in np.argsort(component)[::-1][:12]] for component in H]
        by_period = []
        for start in (2016, 2018, 2020, 2022, 2024):
            ix = [i for i, r in enumerate(subset) if start <= r["year"] <= start + 1]
            by_period.append({"period": f"{start}-{start+1}", "n": len(ix),
                              "mean_shares": [float(v) for v in aligned[ix].mean(axis=0)] if ix else []})
        results.append({"scenario": name, "n": len(subset), "error": float(error),
                        "aligned_cosine": cosine, "alignment_order": order,
                        "aligned_top_terms": [topic_terms[j] for j in order],
                        "mean_aligned_cosine": float(np.mean(cosine)),
                        "min_aligned_cosine": float(np.min(cosine)), "periods": by_period})
    if only:
        previous = json.loads((HERE / f"sensitivity_k{k}.json").read_text(encoding="utf-8"))
        results = [r for r in previous if r["scenario"] != only] + results
    write_json(f"sensitivity_k{k}.json", results)


LABELS_K8 = [
    "Giáo dục, ngôn ngữ và học tập",
    "Hành vi khách hàng, thương hiệu và dịch vụ",
    "Thanh khoản ngân hàng và chính sách tiền tệ",
    "Ổn định tài chính và tài chính toàn diện (nhóm pha trộn)",
    "Ngân hàng thương mại, tín dụng và rủi ro",
    "Phương trình vi phân phân số và logic mờ",
    "Tăng trưởng kinh tế, thương mại và phát triển",
    "Thị trường chứng khoán và tài chính doanh nghiệp",
]
LABELS_K7 = [
    "Tăng trưởng kinh tế, thương mại và phát triển",
    "Hành vi khách hàng, thương hiệu và dịch vụ",
    "Ngân hàng thương mại, tín dụng và rủi ro",
    "Giáo dục, ngôn ngữ và học tập",
    "Thị trường chứng khoán và tài chính doanh nghiệp",
    "Thanh khoản ngân hàng và chính sách tiền tệ",
    "Phương trình vi phân phân số và logic mờ",
]


def describe(k: int = 8) -> None:
    rows = corpus_rows()
    theta = np.load(HERE / f"theta_k{k}.npy")
    assert theta.shape == (len(rows), k)
    eligible = np.linalg.norm(np.load(HERE / "tfidf_refreshed.npy"), axis=1) > 0
    assert np.array_equal(eligible, theta.sum(axis=1) > 0)
    write_json("model_eligibility.json", {"corpus_n": len(rows), "model_eligible_n": int(eligible.sum()),
                                          "unassigned": [{"id": rows[i]["id"], "year": rows[i]["year"],
                                                          "title": rows[i]["title"]}
                                                         for i in np.where(~eligible)[0]]})
    labels = {7: LABELS_K7, 8: LABELS_K8}.get(k)
    if labels is None:
        raise ValueError("Manual labels presently reviewed only for K=7 and K=8")
    document_rows = []
    for i, row in enumerate(rows):
        doc = {field: row.get(field, "") for field in
               ("id", "year", "title", "source", "doi", "has_abstract", "affiliation_evidence", "review_id")}
        doc["model_eligible"] = bool(eligible[i])
        doc["dominant_topic"] = int(theta[i].argmax()) if eligible[i] else ""
        doc.update({f"topic_{j}_share": round(float(theta[i, j]), 6) for j in range(k)})
        document_rows.append(doc)
    write_csv(f"documents_k{k}.csv", document_rows, list(document_rows[0]))

    period_rows = []
    for start in (2016, 2018, 2020, 2022, 2024):
        ix = [i for i, row in enumerate(rows) if eligible[i] and start <= row["year"] <= start+1]
        old_ix = [i for i in ix if rows[i]["source"] == "inherited_openalex"]
        for group, selected in (("all", ix), ("inherited_only_fixed_basis", old_ix)):
            mean = theta[selected].mean(axis=0)
            dominant = np.bincount(theta[selected].argmax(axis=1), minlength=k)
            for topic in range(k):
                period_rows.append({"period": f"{start}-{start+1}", "group": group,
                                    "n": len(selected), "topic": topic, "label": labels[topic],
                                    "mean_share": round(float(mean[topic]), 6),
                                    "dominant_n": int(dominant[topic])})
    write_csv(f"topic_shares_by_period_k{k}.csv", period_rows,
              ["period", "group", "n", "topic", "label", "mean_share", "dominant_n"])

    review = []
    topics = json.loads((HERE / f"topics_k{k}.json").read_text(encoding="utf-8"))
    for item in topics:
        j = item["topic"]
        item["mean_share"] = float(theta[eligible, j].mean())
        review.append({"topic": j, "provisional_label": labels[j],
                       "top_terms": " | ".join(item["top_terms"][:12]),
                       "representative_titles": " | ".join(r["title"] for r in item["representative_works"][:5]),
                       "npmi": round(item["npmi"], 4),
                       "seed_stability_cosine": round(item["stability_cosine"], 4),
                       "expert_label": "", "expert_decision": "", "expert_notes": ""})
    write_json(f"topics_k{k}.json", topics)
    write_csv(f"topic_expert_review_k{k}.csv", review,
              ["topic", "provisional_label", "top_terms", "representative_titles", "npmi",
               "seed_stability_cosine", "expert_label", "expert_decision", "expert_notes"])

    excluded = json.loads((CURATED / "prolific_author_exclusions.json").read_text(encoding="utf-8"))
    author_results = []
    for count in (5, 10):
        excluded_ids = set(excluded[str(count)]["paper_ids"])
        selected = [i for i, r in enumerate(rows) if eligible[i] and r["id"] not in excluded_ids]
        author_results.append({"scenario": f"exclude_top_{count}_known_openalex_authors",
                               "removed_known_author": int(sum(eligible[i] and r["id"] in excluded_ids
                                                               for i, r in enumerate(rows))),
                               "remaining_model_eligible": len(selected),
                               "added_records_without_author_id": sum(rows[i]["source"] != "inherited_openalex"
                                                                      for i in selected),
                               "topic_shares": [float(v) for v in theta[selected].mean(axis=0)],
                               "early_share": [float(v) for v in theta[[i for i in selected if 2016 <= rows[i]["year"] <= 2019]].mean(axis=0)],
                               "late_share": [float(v) for v in theta[[i for i in selected if 2022 <= rows[i]["year"] <= 2025]].mean(axis=0)]})
    write_json(f"author_exclusion_fixed_basis_k{k}.json", author_results)

    main = [r for r in period_rows if r["group"] == "all"]
    old = [r for r in period_rows if r["group"] == "inherited_only_fixed_basis"]
    differences = []
    for j in range(k):
        early = next(r for r in main if r["topic"] == j and r["period"] == "2016-2017")
        late = next(r for r in main if r["topic"] == j and r["period"] == "2024-2025")
        new2020 = next(r for r in main if r["topic"] == j and r["period"] == "2020-2021")
        old2020 = next(r for r in old if r["topic"] == j and r["period"] == "2020-2021")
        differences.append({"topic": j, "label": labels[j],
                            "early_share": early["mean_share"], "late_share": late["mean_share"],
                            "late_minus_early": round(late["mean_share"]-early["mean_share"], 6),
                            "all_2020_2021": new2020["mean_share"],
                            "old_only_2020_2021": old2020["mean_share"]})
    write_csv(f"descriptive_changes_k{k}.csv", differences,
              ["topic", "label", "early_share", "late_share", "late_minus_early",
               "all_2020_2021", "old_only_2020_2021"])

    contrasts = []
    for group in ("all", "inherited_only_fixed_basis"):
        selected = [i for i, r in enumerate(rows) if eligible[i] and
                    (group == "all" or r["source"] == "inherited_openalex")]
        early_ix = [i for i in selected if 2016 <= rows[i]["year"] <= 2019]
        late_ix = [i for i in selected if 2022 <= rows[i]["year"] <= 2025]
        early_article = theta[early_ix].mean(axis=0)
        late_article = theta[late_ix].mean(axis=0)
        early_bins = np.mean([theta[[i for i in selected if start <= rows[i]["year"] <= start+1]].mean(axis=0)
                              for start in (2016, 2018)], axis=0)
        late_bins = np.mean([theta[[i for i in selected if start <= rows[i]["year"] <= start+1]].mean(axis=0)
                             for start in (2022, 2024)], axis=0)
        for j in range(k):
            contrasts.append({"group": group, "topic": j, "label": labels[j],
                              "early_article_n": len(early_ix), "late_article_n": len(late_ix),
                              "early_article_share": float(early_article[j]),
                              "late_article_share": float(late_article[j]),
                              "article_contrast": float(late_article[j]-early_article[j]),
                              "early_equal_bin_share": float(early_bins[j]),
                              "late_equal_bin_share": float(late_bins[j]),
                              "equal_bin_contrast": float(late_bins[j]-early_bins[j])})
    write_csv(f"trend_contrasts_k{k}.csv", contrasts,
              ["group", "topic", "label", "early_article_n", "late_article_n",
               "early_article_share", "late_article_share", "article_contrast",
               "early_equal_bin_share", "late_equal_bin_share", "equal_bin_contrast"])
    print(json.dumps(differences, ensure_ascii=False), flush=True)


def freeze(k: int = 8) -> None:
    grid = json.loads((HERE / "grid_results.json").read_text(encoding="utf-8"))
    fit = next(item for item in grid["results"] if item["k"] == k)
    files = ["refresh_hub.py", "corpus_refreshed.jsonl", "vocabulary.json",
             f"components_k{k}.npy", f"theta_k{k}.npy", f"topics_k{k}.json",
             f"documents_k{k}.csv", f"sensitivity_k{k}.json",
             f"trend_contrasts_k{k}.csv", "model_eligibility.json"]
    write_json("model_lock.json", {"k": k, "chosen_seed": fit["best_seed"],
                                   "n_corpus": len(corpus_rows()),
                                   "n_model_eligible": json.loads((HERE / "model_eligibility.json").read_text(encoding="utf-8"))["model_eligible_n"],
                                   "settings": {key: grid[key] for key in ("model", "features", "max_iter", "seeds", "place_stop")},
                                   "file_sha256": {name: sha256(HERE / name) for name in files},
                                   "interpretation_status": "computational coordinates fixed; topic labels provisional pending expert review"})


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if stage == "audit":
        audit()
    elif stage == "fit":
        fit(tuple(int(arg) for arg in sys.argv[2:]) if len(sys.argv) > 2 else K_GRID)
    elif stage == "sensitivity":
        sensitivity(int(sys.argv[2]) if len(sys.argv) > 2 else 8,
                    sys.argv[3] if len(sys.argv) > 3 else None)
    elif stage == "describe":
        describe(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif stage == "freeze":
        freeze(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        raise SystemExit("Use audit, fit, sensitivity [K] [scenario], describe [K], or freeze [K]")
