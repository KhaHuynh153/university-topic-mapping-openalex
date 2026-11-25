"""Reproducible NumPy-only TF-IDF/NMF topic fit for the new Topic 7 study.

The previous project had scikit-learn fits. This independent implementation is
used because that runtime is not installed in the current workspace. All input
paths point at inherited files; all outputs remain beside this script.
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "topic_7_research_plan" / "pilot" / "data" / "curated" / "english_texts_v2.jsonl"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
EXCLUDED = "https://openalex.org/W3025768660"
TOKEN = re.compile(r"[a-zA-Z][a-zA-Z-]{2,}")
STOP = set("""a about above across after again against all almost also although always am among an and any are around as at be because been before being below between both but by can cannot could did do does doing done down during each either else enough especially even ever every few first for from further get gets got had has have having he her here hers herself him himself his how however i if in into is it its itself just last less let like likely may me might more most much must my myself neither no nor not of off often on once one only or other our ours ourselves out over own per perhaps rather same she should since so some such than that the their theirs them themselves then there therefore these they this those through throughout to too toward towards under until up upon us use used using very via was we were what when where which while who whom why will with within without would you your yours yourself yourselves study studies research results analysis data based method methods new effect effects impact impacts role case cases paper papers article articles show shows shown among authors author university universities vietnam vietnamese findings found significant relationship relationships factors factor model models approach approaches evidence""".split())


def load_rows() -> list[dict]:
    rows = [json.loads(s) for s in SOURCE.read_text(encoding="utf-8").splitlines() if s]
    return [r for r in rows if 2016 <= r["year"] <= 2025 and r["id"] != EXCLUDED]


def tokenize(text: str) -> list[str]:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    words = [w for w in TOKEN.findall(norm) if w not in STOP and len(w) >= 3]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def vectorize(rows: list[dict], mode: str, max_features: int = 5000) -> tuple[np.ndarray, list[str], np.ndarray]:
    texts = [r["title"] if mode == "title" else r["text"] for r in rows]
    tokenized = [tokenize(t) for t in texts]
    df = Counter()
    tf_all = Counter()
    for ts in tokenized:
        df.update(set(ts))
        tf_all.update(ts)
    terms = [t for t, n in df.items() if 3 <= n <= int(0.85 * len(rows))]
    terms.sort(key=lambda t: (-tf_all[t], t))
    terms = terms[:max_features]
    index = {t: i for i, t in enumerate(terms)}
    X = np.zeros((len(rows), len(terms)), dtype=np.float32)
    B = np.zeros_like(X, dtype=bool)
    for d, ts in enumerate(tokenized):
        for t, n in Counter(ts).items():
            if t in index:
                X[d, index[t]] = n
                B[d, index[t]] = True
    document_frequency = B.sum(axis=0)
    idf = (np.log((1 + len(rows)) / (1 + document_frequency)) + 1).astype(np.float32)
    X *= idf
    norm = np.linalg.norm(X, axis=1)
    X /= np.where(norm == 0, 1, norm)[:, None]
    return X, terms, B


def fit_nmf(X: np.ndarray, k: int, seed: int, max_iter: int = 240) -> tuple[np.ndarray, np.ndarray, float, int]:
    rng = np.random.default_rng(seed)
    W = (rng.random((X.shape[0], k), dtype=np.float32) + 0.1).astype(np.float32)
    H = (rng.random((k, X.shape[1]), dtype=np.float32) + 0.1).astype(np.float32)
    H *= np.sqrt(float(X.mean()) / max(float((W @ H).mean()), 1e-8))
    W *= np.sqrt(float(X.mean()) / max(float((W @ H).mean()), 1e-8))
    prev_loss = float("inf")
    for iteration in range(1, max_iter + 1):
        H *= (W.T @ X) / (W.T @ W @ H + 1e-8)
        W *= (X @ H.T) / (W @ H @ H.T + 1e-8)
        if iteration % 20 == 0 or iteration == max_iter:
            loss = float(np.linalg.norm(X - W @ H) / max(np.linalg.norm(X), 1e-8))
            if prev_loss - loss < 1e-6 and iteration >= 100:
                break
            prev_loss = loss
    return W, H, loss, iteration


def npmi(H: np.ndarray, B: np.ndarray) -> list[float]:
    n = B.shape[0]
    df = B.sum(axis=0)
    result = []
    for h in H:
        ix = np.argsort(h)[-10:]
        scores = []
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                nij = int(np.count_nonzero(B[:, i] & B[:, j]))
                if nij == 0:
                    scores.append(-1.0)
                else:
                    pi, pj, pij = df[i] / n, df[j] / n, nij / n
                    scores.append(float(np.log(pij / (pi * pj)) / -np.log(pij)))
        result.append(float(np.mean(scores)))
    return result


def run() -> None:
    rows = load_rows()
    X, terms, B = vectorize(rows, "combined")
    best = None
    seed_results = []
    for seed in range(5):
        W, H, loss, it = fit_nmf(X, 8, seed)
        seed_results.append({"seed": seed, "relative_error": loss, "iterations": it})
        print(f"K8 seed={seed} error={loss:.5f} iterations={it}", flush=True)
        if best is None or loss < best[2]:
            best = (W, H, loss, seed)
    W, H, loss, seed = best
    theta = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-8)
    np.save(OUT / "theta_k8.npy", theta)
    np.save(OUT / "components_k8.npy", H)
    (OUT / "vocabulary_k8.json").write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")
    scores = npmi(H, B)
    topics = []
    for k in range(8):
        top = [terms[j] for j in np.argsort(H[k])[::-1][:15]]
        exemplars = [{"id": rows[i]["id"], "year": rows[i]["year"], "title": rows[i]["title"], "share": float(theta[i, k])} for i in np.argsort(theta[:, k])[::-1][:5]]
        topics.append({"topic": k, "top_terms": top, "npmi": scores[k], "mean_share": float(theta[:, k].mean()), "effective_documents": float(theta[:, k].sum()), "representative_works": exemplars})
    (OUT / "topics_k8.json").write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "documents_k8.csv").open("w", encoding="utf-8", newline="") as handle:
        cols = ["id", "year", "title", "has_abstract"] + [f"topic_{k}" for k in range(8)]
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({"id": r["id"], "year": r["year"], "title": r["title"], "has_abstract": r["has_abstract"], **{f"topic_{k}": float(theta[i, k]) for k in range(8)}})
    summary = {"method": "NumPy multiplicative-update NMF, Euclidean loss", "n_documents": len(rows), "n_terms": len(terms), "k": 8, "chosen_seed": seed, "relative_error": loss, "seed_results": seed_results, "numpy": np.__version__, "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS")}
    (OUT / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
