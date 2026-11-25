"""Reproduce the main HUB results from the released term-count matrix.

Usage:  python code/reproduce_main_results.py        (from the repository root)
Needs:  numpy, scipy (see requirements.txt). Runs in a few minutes on a laptop.

Steps (same settings as the paper):
  1. TF-IDF: w = tf * (ln((1+N)/(1+df)) + 1), rows scaled to unit length.
  2. NMF (multiplicative updates, Lee & Seung 2001), K = 8, seeds 0-9, at most 160
     iterations, convergence checked every 20 iterations from iteration 100 (tol 1e-6);
     the seed with the smallest relative error is kept.
  3. theta = rows of W divided by their sums.
  4. Links: theta >= 0.20; joint >= 10 and lift >= 1.20; hypergeometric p (Bonferroni, 28 pairs).
  5. Share change 2016-2019 -> 2022-2025 with a 2,000-draw author-cluster bootstrap.
The printed topic order is the order of the data (0..7); the paper's codes CĐ1..CĐ8 are
data topics [6, 1, 0, 7, 4, 5, 2, 3].
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAPER_ORDER = [6, 1, 0, 7, 4, 5, 2, 3]


def tfidf(C: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    df = (C > 0).sum(axis=0)
    X = C * (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    norm = np.linalg.norm(X, axis=1)
    return (X / np.where(norm == 0, 1, norm)[:, None]).astype(np.float32)


def fit_nmf(X, k, seed, max_iter=160):
    rng = np.random.default_rng(seed)
    W = (rng.random((X.shape[0], k), dtype=np.float32) + 0.1).astype(np.float32)
    H = (rng.random((k, X.shape[1]), dtype=np.float32) + 0.1).astype(np.float32)
    H *= np.sqrt(float(X.mean()) / max(float((W @ H).mean()), 1e-8))
    W *= np.sqrt(float(X.mean()) / max(float((W @ H).mean()), 1e-8))
    prev = float("inf")
    for it in range(1, max_iter + 1):
        H *= (W.T @ X) / (W.T @ W @ H + 1e-8)
        W *= (X @ H.T) / (W @ H @ H.T + 1e-8)
        if it % 20 == 0 or it == max_iter:
            loss = float(np.linalg.norm(X - W @ H) / max(np.linalg.norm(X), 1e-8))
            if prev - loss < 1e-6 and it >= 100:
                break
            prev = loss
    return W, H, loss


def main() -> None:
    C = sparse.load_npz(DATA / "hub_counts_combined.npz").toarray().astype(np.float32)
    vocab = json.loads((DATA / "hub_vocabulary.json").read_text(encoding="utf-8"))
    with (DATA / "hub_papers.csv").open(encoding="utf-8-sig") as h:
        papers = list(csv.DictReader(h))
    X = tfidf(C)
    fits = [fit_nmf(X, 8, s) for s in range(10)]
    best = min(range(10), key=lambda i: fits[i][2])
    W, H, err = fits[best]
    print(f"chosen seed {best}, relative error {err:.5f}")
    eligible = np.linalg.norm(X, axis=1) > 0
    theta = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    theta[~eligible] = 0
    shares = theta[eligible].mean(axis=0)
    for c, k in enumerate(PAPER_ORDER):
        top = [vocab[j] for j in np.argsort(H[k])[::-1][:8]]
        print(f"CĐ{c+1} (data {k}): share {100*shares[k]:.1f}%  {', '.join(top)}")
    # links
    x = theta[eligible]
    n = len(x)
    mem = x >= 0.20
    sup = mem.sum(axis=0)
    print("\nLinks passing the rule (joint >= 10, lift >= 1.20):")
    for i in range(8):
        for j in range(i + 1, 8):
            joint = int((mem[:, i] & mem[:, j]).sum())
            exp = sup[i] * sup[j] / n
            if joint >= 10 and joint / exp >= 1.20:
                p = hypergeom.sf(joint - 1, n, sup[i], sup[j])
                print(f"  data {i}-{j}: joint {joint}, expected {exp:.1f}, lift {joint/exp:.2f}, p {p:.2e}")
    # temporal change with author-cluster bootstrap
    years = np.array([int(p["year"]) for p in papers])
    early, late = eligible & (years <= 2019), eligible & (years >= 2022)
    delta = theta[late].mean(axis=0) - theta[early].mean(axis=0)
    labels = np.array([p["author_cluster"] for p in papers])[eligible]
    names = {s: i for i, s in enumerate(sorted(set(labels)))}
    idx = np.array([names[s] for s in labels])
    e2, l2 = early[eligible], late[eligible]
    rng = np.random.default_rng(20261027)   # same seed as the paper (design: list additions as single-paper clusters)
    sims = np.empty((2000, 8))
    for b in range(2000):
        w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[idx]
        sims[b] = np.average(x[l2], axis=0, weights=w[l2]) - np.average(x[e2], axis=0, weights=w[e2])
    lo, hi = np.quantile(sims, [.025, .975], axis=0)
    print(f"\nShare change 2016-2019 (n={early.sum()}) -> 2022-2025 (n={late.sum()}), {len(names)} clusters:")
    for c, k in enumerate(PAPER_ORDER):
        print(f"  CĐ{c+1}: {100*delta[k]:+.1f} pp  [{100*lo[k]:+.1f}; {100*hi[k]:+.1f}]")


if __name__ == "__main__":
    main()
