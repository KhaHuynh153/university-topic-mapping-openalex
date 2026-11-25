"""R1 step 1a (follow-up): is the LDA result stable across its 10 seeds?
For every seed: align to NMF, report term cosine, share change (early vs late) and the
CĐ7-CĐ5 link. Output: out/lda_seeds.csv, out/lda_seeds.json
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import LatentDirichletAllocation

from r1_common import BASE, CODE, PAPER_TO_DATA, count_matrix, hub_rows, hub_vocabulary, hungarian, network_hub, row_cosine, write_csv, write_json


def one(seed, Xe):
    lda = LatentDirichletAllocation(n_components=8, learning_method="batch", max_iter=200, random_state=seed, n_jobs=1)
    doc = lda.fit_transform(Xe)
    return seed, lda.components_, doc, float(lda.score(Xe))


def main():
    rows = hub_rows()
    X = count_matrix(rows, hub_vocabulary())
    elig = X.sum(axis=1) > 0
    Xe = X[elig]
    H = np.load(BASE / "components_k8.npy")
    years = np.array([r["year"] for r in rows])
    early, late = elig & (years <= 2019), elig & (years >= 2022)
    res = Parallel(n_jobs=5)(delayed(one)(s, Xe) for s in range(10))
    out = []
    for seed, comp, doc, ll in res:
        comp = comp / comp.sum(axis=1, keepdims=True)
        sim = row_cosine(H, comp)
        o = hungarian(sim)
        th = np.zeros((len(rows), 8))
        th[elig] = doc[:, o]
        e = network_hub.edges(th, 0.20)
        pair = next(x for x in e if x["topic_i"] == 2 and x["topic_j"] == 4)
        rec = {"seed": seed, "loglik": round(ll, 1), "liq_cb_joint": pair["joint_count"], "liq_cb_lift": round(pair["lift"], 2),
               "n_links_passing": sum(bool(x["passes_rule"]) for x in e)}
        for d in PAPER_TO_DATA:
            rec[f"cos_{CODE[d]}"] = round(float(sim[d, o[d]]), 2)
            rec[f"delta_{CODE[d]}"] = round(100 * float(th[late, d].mean() - th[early, d].mean()), 1)
        out.append(rec)
        print(rec, flush=True)
    out.sort(key=lambda r: r["seed"])
    write_csv("lda_seeds.csv", out)
    summ = {}
    for d in PAPER_TO_DATA:
        c = CODE[d]
        cos = [r[f"cos_{c}"] for r in out]
        dl = [r[f"delta_{c}"] for r in out]
        summ[c] = {"cos_min": min(cos), "cos_median": float(np.median(cos)), "cos_max": max(cos),
                   "n_seeds_cos_ge_0_7": sum(x >= 0.7 for x in cos),
                   "delta_min": min(dl), "delta_median": float(np.median(dl)), "delta_max": max(dl)}
    summ["liq_cb_lift"] = [r["liq_cb_lift"] for r in out]
    summ["liq_cb_passes"] = sum(r["liq_cb_joint"] >= 10 and r["liq_cb_lift"] >= 1.2 for r in out)
    write_json("lda_seeds.json", summ)
    print(summ)


if __name__ == "__main__":
    main()
