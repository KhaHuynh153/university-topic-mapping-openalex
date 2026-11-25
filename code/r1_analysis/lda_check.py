"""R1 step 1a: LDA robustness check against the locked NMF K=8 model (HUB).

Same 1,335 eligible papers, same 3,000-term vocabulary; LDA needs raw counts.
Outputs (analysis/out/): lda_summary.json, lda_topics.csv, lda_theta_aligned.npy
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from r1_common import (BASE, CODE, PAPER_TO_DATA, SHORT_VI, OUT, count_matrix, hub_rows,
                       hub_vocabulary, hungarian, network_hub, row_cosine, write_csv, write_json)

K = 8
SEEDS = range(10)
MAX_ITER = 200


def main() -> None:
    rows = hub_rows()
    vocab = hub_vocabulary()
    X = count_matrix(rows, vocab)
    eligible = X.sum(axis=1) > 0
    assert int(eligible.sum()) == 1335, eligible.sum()
    Xe = X[eligible]
    H_nmf = np.load(BASE / "components_k8.npy")
    theta_nmf = np.load(BASE / "theta_k8.npy")

    fits = []
    for seed in SEEDS:
        t0 = time.time()
        lda = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=MAX_ITER,
                                        evaluate_every=0, random_state=seed, n_jobs=1)
        doc = lda.fit_transform(Xe)
        ll = float(lda.score(Xe))
        fits.append({"seed": seed, "components": lda.components_.copy(), "doc": doc, "loglik": ll,
                     "n_iter": int(lda.n_iter_)})
        print(f"seed={seed} loglik={ll:.1f} iter={lda.n_iter_} {time.time()-t0:.0f}s", flush=True)
    best = max(fits, key=lambda f: f["loglik"])
    comp = best["components"] / best["components"].sum(axis=1, keepdims=True)

    # Align LDA topics to NMF topics by term-vector cosine (Hungarian).
    sim = row_cosine(H_nmf, comp)
    order = hungarian(sim)
    term_cos = [float(sim[i, order[i]]) for i in range(K)]
    theta_lda = np.zeros((len(rows), K))
    theta_lda[eligible] = best["doc"][:, order]
    np.save(OUT / "lda_theta_aligned.npy", theta_lda)
    # Document-level agreement: correlation of theta columns.
    doc_corr = [float(np.corrcoef(theta_nmf[eligible, i], theta_lda[eligible, i])[0, 1]) for i in range(K)]
    # Also the best doc-level alignment, to see whether term alignment is consistent.
    corr_full = np.corrcoef(theta_nmf[eligible].T, best["doc"].T)[:K, K:]
    order_doc = hungarian(corr_full)

    # Seed stability of LDA (mean matched cosine of each seed against the best fit).
    stab = []
    for f in fits:
        c = f["components"] / f["components"].sum(axis=1, keepdims=True)
        s = row_cosine(comp, c)
        o = hungarian(s)
        stab.append([float(s[i, o[i]]) for i in range(K)])
    stab = np.array(stab)
    stab_aligned = stab.mean(axis=0)[order]  # stability per NMF-ordered topic

    # Shares, temporal contrast, cluster bootstrap, links.
    shares = theta_lda[eligible].mean(axis=0)
    tc = network_hub.temporal_contrast(theta_lda, rows)
    years = np.array([r["year"] for r in rows])
    early = eligible & (years <= 2019)
    late = eligible & (years >= 2022)
    boot = network_hub.cluster_bootstrap(theta_lda, rows, bundle_additions=False)
    links = {}
    for thr in (0.15, 0.20, 0.25):
        e = network_hub.edges(theta_lda, thr)
        links[str(thr)] = [x for x in e]
    nmf_links = network_hub.edges(theta_nmf, 0.20)

    vocab_arr = np.array(vocab)
    topic_rows = []
    for d in PAPER_TO_DATA:
        j = order[d]
        top = vocab_arr[np.argsort(comp[j])[::-1][:10]].tolist()
        topic_rows.append({
            "code": CODE[d], "short": SHORT_VI[d], "data_index": d,
            "term_cosine_nmf_lda": round(term_cos[d], 3),
            "doc_theta_corr": round(doc_corr[d], 3),
            "doc_alignment_agrees": order_doc[d] == order[d],
            "lda_seed_stability": round(float(stab_aligned[d]), 3),
            "lda_share_pct": round(100 * float(shares[d]), 1),
            "nmf_share_pct": round(100 * float(theta_nmf[eligible, d].mean()), 1),
            "lda_early_pct": round(100 * float(theta_lda[early, d].mean()), 1),
            "lda_late_pct": round(100 * float(theta_lda[late, d].mean()), 1),
            "lda_delta_pp": round(100 * float(tc["article_delta"][d]), 1),
            "lda_ci_low": round(100 * boot["ci_low"][d], 1),
            "lda_ci_high": round(100 * boot["ci_high"][d], 1),
            "lda_count_early": round(float(theta_lda[early, d].sum()), 1),
            "lda_count_late": round(float(theta_lda[late, d].sum()), 1),
            "lda_top_terms": ", ".join(top),
        })
    write_csv("lda_topics.csv", topic_rows)

    def pair(edge_list, a, b):
        a, b = sorted((a, b))
        return next(x for x in edge_list if x["topic_i"] == a and x["topic_j"] == b)

    liq_cb = {thr: pair(v, 2, 4) for thr, v in links.items()}
    passing = {thr: [(CODE[x["topic_i"]], CODE[x["topic_j"]], x["joint_count"], round(x["lift"], 2))
                     for x in v if x["passes_rule"]] for thr, v in links.items()}
    summary = {
        "settings": {"k": K, "seeds": list(SEEDS), "max_iter": MAX_ITER, "learning_method": "batch",
                     "doc_topic_prior": "default 1/K", "topic_word_prior": "default 1/K",
                     "input": "raw counts of the 3,000 locked NMF terms", "n_documents": int(eligible.sum()),
                     "sklearn": __import__("sklearn").__version__},
        "best_seed": best["seed"], "best_loglik": best["loglik"], "best_n_iter": best["n_iter"],
        "logliks": [f["loglik"] for f in fits],
        "mean_term_cosine": float(np.mean(term_cos)), "min_term_cosine": float(np.min(term_cos)),
        "mean_lda_seed_stability": float(stab.mean()), "min_lda_topic_stability": float(stab.mean(axis=0).min()),
        "doc_alignment_agreement": int(sum(order_doc[i] == order[i] for i in range(K))),
        "cluster_bootstrap": {k: v for k, v in boot.items() if k not in ("ci_low", "ci_high")},
        "liquidity_commercial_bank_pair": {thr: {"joint": x["joint_count"], "lift": round(x["lift"], 3),
                                                 "expected": x["expected_independence"],
                                                 "passes": bool(x["passes_rule"])}
                                           for thr, x in liq_cb.items()},
        "passing_pairs": passing,
        "nmf_liquidity_pair": {"joint": pair(nmf_links, 2, 4)["joint_count"], "lift": pair(nmf_links, 2, 4)["lift"]},
        "topics": topic_rows,
    }
    write_json("lda_summary.json", summary)
    for r in topic_rows:
        print(r["code"], r["short"], r["term_cosine_nmf_lda"], r["doc_theta_corr"], r["lda_seed_stability"],
              r["nmf_share_pct"], r["lda_share_pct"], r["lda_delta_pp"], (r["lda_ci_low"], r["lda_ci_high"]),
              r["lda_top_terms"][:80])
    print("liq-cb", summary["liquidity_commercial_bank_pair"])
    print("passing", passing)


if __name__ == "__main__":
    main()
