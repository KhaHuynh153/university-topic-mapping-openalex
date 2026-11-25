"""Phân tích bổ sung cho bản thảo sửa ngày 17/11/2026.

Script chỉ đọc các kết quả đã khóa (theta_k8.npy, components_k*.npy, corpus_refreshed.jsonl ...)
và ghi kết quả mới vào các file extra_*.json / extra_*.csv. Không ghi đè file nào của mô hình chính.

Các phần:
  kgrid        Lưới K = 5..10: NPMI, độ ổn định giữa 10 lần khởi tạo, khớp lại chỉ bằng tiêu đề.
  lift         Kiểm định siêu bội cho 28 cặp chủ đề (ngưỡng theta >= 0,20).
  counts       Số bài quy đổi (tổng theta) và số bài có theta >= 0,20 theo nhóm hai năm.
  bridge       Kiểm tra từ khóa trong 51 bài thuộc cả chủ đề 2 và chủ đề 4.
  authors      Mức tập trung theo tác giả trong tập OpenAlex ban đầu.
  convergence  Chạy NMF K = 8 (seed đã chọn) đến 3.000 vòng, không dùng tiêu chí dừng.

Chạy: python extra_analyses.py [tên phần ...]   (không có tham số: chạy tất cả)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

import refresh_hub as rh

HERE = Path(__file__).resolve().parent
original = rh.original
BINS = [(2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025)]
THRESHOLD = 0.20


def write_json(name: str, data: object) -> None:
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def title_matrix(rows: list[dict], vocabulary: list[str]) -> np.ndarray:
    """Ma trận TF-IDF chỉ từ tiêu đề, giữ nguyên bộ thuật ngữ của mô hình chính."""
    index = {term: i for i, term in enumerate(vocabulary)}
    X = np.zeros((len(rows), len(vocabulary)), dtype=np.float32)
    for i, r in enumerate(rows):
        for term, count in Counter(original.tokenize(r["title"])).items():
            if term in index:
                X[i, index[term]] = count
    df = np.count_nonzero(X, axis=0)
    X *= (np.log((1 + len(rows)) / (1 + df)) + 1).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def kgrid() -> None:
    rows = rh.corpus_rows()
    X, terms, B = original.vectorize(rows, "combined", rh.FEATURES)
    saved_vocab = json.loads((HERE / "vocabulary.json").read_text(encoding="utf-8"))
    assert terms == saved_vocab, "Bộ thuật ngữ khác mô hình đã khóa"
    Xt = title_matrix(rows, terms)
    out = []
    for k in (5, 6, 7, 8, 9, 10):
        fits = [original.fit_nmf(X, k, seed, rh.MAX_ITER) for seed in rh.SEEDS]
        chosen = min(range(len(fits)), key=lambda i: fits[i][2])
        H = fits[chosen][1]
        saved = HERE / f"components_k{k}.npy"
        if saved.exists():  # K đã có trong mô hình chính: phải tái lập đúng kết quả đã lưu
            _, cos = rh.align_cosine(np.load(saved), H)
            assert min(cos) > 0.9999, f"K={k} không tái lập được kết quả đã lưu"
        coherence = original.npmi(H, B)
        stability = np.mean(np.array([rh.align_cosine(H, f[1])[1] for f in fits]), axis=0)
        tfits = [original.fit_nmf(Xt, k, seed, rh.MAX_ITER) for seed in range(5)]
        Ht = min(tfits, key=lambda f: f[2])[1]
        _, tcos = rh.align_cosine(H, Ht)
        top = [[terms[j] for j in np.argsort(H[t])[::-1][:10]] for t in range(k)]
        out.append({"k": k, "best_seed": chosen, "iterations_best": int(fits[chosen][3]),
                    "max_iterations_any_seed": int(max(f[3] for f in fits)),
                    "relative_error": round(float(fits[chosen][2]), 5),
                    "mean_npmi": round(float(np.mean(coherence)), 4),
                    "min_npmi": round(float(np.min(coherence)), 4),
                    "mean_stability_cosine": round(float(np.mean(stability)), 4),
                    "min_stability_cosine": round(float(np.min(stability)), 4),
                    "title_only_mean_cosine": round(float(np.mean(tcos)), 4),
                    "title_only_min_cosine": round(float(np.min(tcos)), 4),
                    "top_terms": top})
        print({key: val for key, val in out[-1].items() if key != "top_terms"}, flush=True)
    write_json("extra_kgrid.json", out)
    write_csv("extra_kgrid.csv", [{key: val for key, val in r.items() if key != "top_terms"} for r in out])


def load_main() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = rh.corpus_rows()
    theta = np.load(HERE / "theta_k8.npy")
    eligible = theta.sum(axis=1) > 0
    return rows, theta, eligible


def lift_test() -> None:
    rows, theta, eligible = load_main()
    member = theta[eligible] >= THRESHOLD
    N = int(eligible.sum())
    support = member.sum(axis=0)
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            nij = int(np.count_nonzero(member[:, i] & member[:, j]))
            ni, nj = int(support[i]), int(support[j])
            expected = ni * nj / N
            # P(X >= nij) khi X theo phân phối siêu bội (N bài, ni bài thuộc chủ đề i, rút nj bài)
            p = sum(comb(ni, x) * comb(N - ni, nj - x) for x in range(nij, min(ni, nj) + 1)) / comb(N, nj)
            out.append({"topic_i": i, "topic_j": j, "n_i": ni, "n_j": nj, "n_ij": nij,
                        "expected": round(expected, 2), "lift": round(nij / expected, 4),
                        "p_upper": float(f"{p:.3e}"), "p_bonferroni_28": float(f"{min(1.0, p * 28):.3e}"),
                        "passes_rule": nij >= 10 and nij / expected >= 1.20})
    write_csv("extra_lift_tests.csv", out)
    for r in sorted(out, key=lambda r: r["p_upper"])[:6]:
        print(r, flush=True)


def counts() -> None:
    rows, theta, eligible = load_main()
    years = np.array([int(r["year"]) for r in rows])
    out = []
    for k in range(8):
        rec = {"topic": k}
        for a, b in BINS:
            ix = eligible & (years >= a) & (years <= b)
            rec[f"n_{a}_{b}"] = int(ix.sum())
            rec[f"share_{a}_{b}"] = round(float(theta[ix, k].mean()), 4)
            rec[f"equivalent_{a}_{b}"] = round(float(theta[ix, k].sum()), 1)
            rec[f"members_{a}_{b}"] = int((theta[ix, k] >= THRESHOLD).sum())
        early = eligible & (years <= 2019)
        late = eligible & (years >= 2022)
        rec.update(share_early=round(float(theta[early, k].mean()), 4),
                   share_late=round(float(theta[late, k].mean()), 4),
                   equivalent_early=round(float(theta[early, k].sum()), 1),
                   equivalent_late=round(float(theta[late, k].sum()), 1),
                   members_all=int((theta[eligible, k] >= THRESHOLD).sum()),
                   dominant_all=int((theta[eligible].argmax(axis=1) == k).sum()))
        out.append(rec)
        print(rec, flush=True)
    write_csv("extra_topic_counts_by_period.csv", out)


def bridge() -> None:
    with (HERE / "bridge_review_2_4.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    both = [r for r in rows if r["both_vocabularies_present"] == "True"]
    summary = {"n_bridge": len(rows),
               "with_abstract": sum(r["has_abstract"] == "True" for r in rows),
               "liquidity_terms_present": sum(bool(r["left_terms_in_text"]) for r in rows),
               "credit_risk_terms_present": sum(bool(r["right_terms_in_text"]) for r in rows),
               "both_present": len(both),
               "both_present_among_with_abstract": sum(r["has_abstract"] == "True" for r in both),
               "examples_both_present": [{"rank": r["rank"], "year": r["year"], "title": r["title"],
                                          "doi": r["doi"]} for r in both[:8]]}
    write_json("extra_bridge_keyword_check.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


def authors() -> None:
    path = HERE.parent / "topic_7_research_plan" / "pilot" / "data" / "curated" / "prolific_author_exclusions.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = rh.corpus_rows()
    inherited = {r["id"] for r in rows if r["source"] == "inherited_openalex"}
    out = []
    for top, item in data.items():
        removed = set(item["paper_ids"]) & inherited
        out.append({"top_authors": int(top), "records": len(removed),
                    "share_of_openalex_records": round(len(removed) / len(inherited), 4)})
    write_csv("extra_author_concentration.csv", out)
    print(out, flush=True)


def convergence() -> None:
    X = np.load(HERE / "tfidf_refreshed.npy").astype(np.float64)
    ref = np.load(HERE / "components_k8.npy")
    lock = json.loads((HERE / "model_lock.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(lock["chosen_seed"])
    k = 8
    W = (rng.random((X.shape[0], k), dtype=np.float32) + 0.1).astype(np.float64)
    H = (rng.random((k, X.shape[1]), dtype=np.float32) + 0.1).astype(np.float64)
    H *= np.sqrt(float(X.mean()) / max(float((W @ H).mean()), 1e-8))
    W *= np.sqrt(float(X.mean()) / max(float((W @ H).mean()), 1e-8))
    norm = np.linalg.norm(X)
    out = []
    for it in range(1, 3001):
        H *= (W.T @ X) / (W.T @ W @ H + 1e-8)
        W *= (X @ H.T) / (W @ H @ H.T + 1e-8)
        if it in (140, 500, 1000, 3000):
            order, cos = rh.align_cosine(ref, H)
            theta = (W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12))[:, order]
            ok = theta.sum(axis=1) > 0
            member = theta[ok] >= THRESHOLD
            n2, n4 = int(member[:, 2].sum()), int(member[:, 4].sum())
            n24 = int((member[:, 2] & member[:, 4]).sum())
            out.append({"iterations": it, "relative_error": round(float(np.linalg.norm(X - W @ H) / norm), 6),
                        "min_cosine_to_locked_topics": round(float(min(cos)), 4),
                        "mean_shares": [round(float(v), 4) for v in theta[ok].mean(axis=0)],
                        "n_2_4": n24, "lift_2_4": round(n24 * int(ok.sum()) / (n2 * n4), 3)})
            print(out[-1], flush=True)
    write_json("extra_convergence_check.json", out)


if __name__ == "__main__":
    parts = {"kgrid": kgrid, "lift": lift_test, "counts": counts, "bridge": bridge,
             "authors": authors, "convergence": convergence}
    for name in (sys.argv[1:] or list(parts)):
        print(f"== {name}", flush=True)
        parts[name]()
