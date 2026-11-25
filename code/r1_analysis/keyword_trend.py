"""R1 step 1a (model-free check): share of papers whose title+abstract contains the key
words of each topic name (label_check.HUB_KEYS), 2016-2019 vs 2022-2025, with the same
author-cluster bootstrap as the main analysis. Output: out/keyword_trend.json
"""
from __future__ import annotations

import numpy as np

from label_check import HUB_KEYS, hit
from r1_common import CODE, PAPER_TO_DATA, hub_rows, network_hub, write_json


def main():
    rows = hub_rows()
    years = np.array([r["year"] for r in rows])
    early, late = years <= 2019, years >= 2022
    M = np.array([[hit(r["text"], HUB_KEYS[d][1]) for d in range(8)] for r in rows], dtype=float)
    labels, meta = network_hub.author_leader_clusters(rows, bundle_additions=False)
    names = {n: i for i, n in enumerate(sorted(set(labels)))}
    idx = np.array([names[n] for n in labels])
    rng = np.random.default_rng(20261125)
    sims = np.empty((2000, 8))
    for b in range(2000):
        w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[idx]
        sims[b] = np.average(M[late], axis=0, weights=w[late]) - np.average(M[early], axis=0, weights=w[early])
    lo, hi = np.quantile(sims, [.025, .975], axis=0)
    out = {}
    for d in PAPER_TO_DATA:
        out[CODE[d]] = {"early_pct": round(100 * M[early, d].mean(), 1), "late_pct": round(100 * M[late, d].mean(), 1),
                        "delta_pp": round(100 * (M[late, d].mean() - M[early, d].mean()), 1),
                        "ci": [round(100 * lo[d], 1), round(100 * hi[d], 1)],
                        "n_early": int(M[early, d].sum()), "n_late": int(M[late, d].sum())}
        print(CODE[d], out[CODE[d]])
    out["_meta"] = {"n_early": int(early.sum()), "n_late": int(late.sum()), **meta}
    write_json("keyword_trend.json", out)


if __name__ == "__main__":
    main()
