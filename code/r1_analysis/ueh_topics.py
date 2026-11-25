"""R1 step 2b-U4: apply the HUB topic-mapping procedure to UEH (no manual additions).

Pre-specified K rule (written before fitting): among K = 5..15, keep the models whose
every topic reappears when only titles are used (minimum title-only cosine >= 0.50);
choose the one with the highest mean NPMI. If none qualifies, choose the K with the
highest minimum title-only cosine.
Everything else mirrors HUB: 3,000 TF-IDF terms (df >= 3, <= 85%), NMF with 10 seeds and
160 iterations, theta >= 0.20 membership, link rule joint >= 10 and lift >= 1.20 with a
Bonferroni hypergeometric test, early 2016-2019 vs late 2022-2025, author-cluster bootstrap.
Outputs: ueh/out/*.json|csv|npy
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from r1_common import hungarian, original, row_cosine

R1 = Path(__file__).resolve().parent.parent
OUTU = R1 / "ueh" / "out"
OUTU.mkdir(parents=True, exist_ok=True)
K_GRID = range(5, 16)
SEEDS = range(10)
MAX_ITER = 160
FEATURES = 3000
EXTRA_STOP = {"ueh"}

_base_tok = original.tokenize


def tok(text: str) -> list[str]:
    return [t for t in _base_tok(text) if not any(p in EXTRA_STOP for p in t.split())]


original.tokenize = tok


def wjson(name, data):
    (OUTU / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def load() -> list[dict]:
    return [json.loads(l) for l in (R1 / "ueh" / "ueh_corpus.jsonl").read_text(encoding="utf-8").splitlines() if l]


def tfidf_fixed(texts: list[str], vocab: list[str]) -> np.ndarray:
    index = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for d, t in enumerate(texts):
        for term, n in Counter(tok(t)).items():
            j = index.get(term)
            if j is not None:
                X[d, j] = n
    df = np.count_nonzero(X, axis=0)
    X *= (np.log((1 + len(texts)) / (1 + df)) + 1).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


def best_fit(X, k, seeds):
    fits = [original.fit_nmf(X, k, s, MAX_ITER) for s in seeds]
    i = min(range(len(fits)), key=lambda j: fits[j][2])
    return fits, i


def align(Href, H):
    sim = row_cosine(Href, H)
    o = hungarian(sim)
    return o, [float(sim[i, o[i]]) for i in range(len(o))]


def _fit_job(X, k, seed):
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    return original.fit_nmf(X, k, seed, MAX_ITER)


def grid(rows) -> dict:
    X, vocab, B = original.vectorize(rows, "combined", FEATURES)
    np.save(OUTU / "tfidf.npy", X)
    wjson("vocabulary.json", vocab)
    Xt = tfidf_fixed([r["title"] for r in rows], vocab)
    from joblib import Parallel, delayed
    jobs = [("main", k, s) for k in K_GRID for s in SEEDS] + [("title", k, s) for k in K_GRID for s in range(5)]
    out = Parallel(n_jobs=8, verbose=5)(delayed(_fit_job)(X if kind == "main" else Xt, k, s) for kind, k, s in jobs)
    fitted = {(kind, k, s): f for (kind, k, s), f in zip(jobs, out)}
    results = []
    for k in K_GRID:
        fits = [fitted[("main", k, s)] for s in SEEDS]
        ib = min(range(len(fits)), key=lambda j: fits[j][2])
        W, H, err, it = fits[ib]
        coh = original.npmi(H, B)
        stab = np.mean([align(H, f[1])[1] for f in fits], axis=0)
        tf = [fitted[("title", k, s)] for s in range(5)]
        it_ = min(range(5), key=lambda j: tf[j][2])
        _, tcos = align(H, tf[it_][1])
        np.save(OUTU / f"components_k{k}.npy", H)
        np.save(OUTU / f"W_k{k}.npy", W)
        res = {"k": k, "best_seed": ib, "iterations": it, "error": float(err),
               "mean_npmi": float(np.mean(coh)), "min_npmi": float(np.min(coh)),
               "mean_seed_cos": float(np.mean(stab)), "min_seed_cos": float(np.min(stab)),
               "mean_title_cos": float(np.mean(tcos)), "min_title_cos": float(np.min(tcos)),
               "npmi": [float(x) for x in coh], "seed_cos": [float(x) for x in stab],
               "title_cos": tcos}
        results.append(res)
        print(f"K={k} npmi={res['mean_npmi']:.3f} seed={res['mean_seed_cos']:.3f}/{res['min_seed_cos']:.3f} "
              f"title={res['mean_title_cos']:.3f}/{res['min_title_cos']:.3f}", flush=True)
    ok = [r for r in results if r["min_title_cos"] >= 0.50]
    chosen = max(ok, key=lambda r: r["mean_npmi"]) if ok else max(results, key=lambda r: r["min_title_cos"])
    out = {"n_documents": len(rows), "n_eligible": int((np.linalg.norm(X, axis=1) > 0).sum()),
           "n_features": len(vocab), "rule": "min title cosine >= 0.50, then max mean NPMI",
           "qualifying_k": [r["k"] for r in ok], "chosen_k": chosen["k"], "results": results}
    wjson("grid.json", out)
    return out


def leader_clusters(rows):
    counts = Counter(a for r in rows for a in r["ueh_author_ids"])
    labels = []
    for r in rows:
        labels.append("A:" + max(r["ueh_author_ids"], key=lambda a: (counts[a], a)) if r["ueh_author_ids"]
                      else "U:" + r["id"])
    return labels, counts


def boot(theta, rows, early, late, reps=2000, seed=20261125):
    labels, _ = leader_clusters(rows)
    names = {n: i for i, n in enumerate(sorted(set(labels)))}
    idx = np.array([names[n] for n in labels])
    rng = np.random.default_rng(seed)
    sims = np.empty((reps, theta.shape[1]))
    for b in range(reps):
        w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[idx]
        sims[b] = np.average(theta[late], axis=0, weights=w[late]) - np.average(theta[early], axis=0, weights=w[early])
    lo, hi = np.quantile(sims, [.025, .975], axis=0)
    sizes = Counter(labels)
    return lo, hi, {"n_clusters": len(names), "max_cluster": max(sizes.values())}


def analyse(rows, k) -> dict:
    X = np.load(OUTU / "tfidf.npy")
    vocab = json.loads((OUTU / "vocabulary.json").read_text(encoding="utf-8"))
    H = np.load(OUTU / f"components_k{k}.npy")
    W = np.load(OUTU / f"W_k{k}.npy")
    elig = np.linalg.norm(X, axis=1) > 0
    theta = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    theta[~elig] = 0
    np.save(OUTU / f"theta_k{k}.npy", theta)
    years = np.array([r["year"] for r in rows])
    early = elig & (years <= 2019)
    late = elig & (years >= 2022)
    shares = theta[elig].mean(axis=0)
    # abstract-only re-fit
    ab = [i for i, r in enumerate(rows) if r["has_abstract"]]
    Xa = tfidf_fixed([rows[i]["text"] for i in ab], vocab)
    af, ia = best_fit(Xa, k, range(5))
    _, acos = align(H, af[ia][1])
    # links
    n = int(elig.sum())
    mem = theta[elig] >= 0.20
    sup = mem.sum(axis=0)
    npairs = k * (k - 1) // 2
    links = []
    for i in range(k):
        for j in range(i + 1, k):
            joint = int((mem[:, i] & mem[:, j]).sum())
            exp = sup[i] * sup[j] / n
            p = float(hypergeom.sf(joint - 1, n, sup[i], sup[j]))
            links.append({"i": i, "j": j, "joint": joint, "expected": round(float(exp), 2),
                          "lift": round(joint / exp, 3) if exp else 0, "p": p,
                          "passes_rule": joint >= 10 and exp > 0 and joint / exp >= 1.20,
                          "passes_bonferroni": p < 0.05 / npairs})
    # temporal
    delta = theta[late].mean(axis=0) - theta[early].mean(axis=0)
    lo, hi, cmeta = boot(theta, rows, early, late)
    # author concentration and exclusion (fixed basis)
    labels, counts = leader_clusters(rows)
    top = [a for a, _ in counts.most_common(10)]
    n_all = len(rows)
    conc = {f"top{m}": sum(any(a in top[:m] for a in r["ueh_author_ids"]) for r in rows) for m in (1, 5, 10)}
    excl = {}
    for m in (5, 10):
        keep = np.array([not any(a in top[:m] for a in r["ueh_author_ids"]) for r in rows])
        e2, l2 = early & keep, late & keep
        excl[f"drop_top{m}"] = [float(x) for x in theta[l2].mean(axis=0) - theta[e2].mean(axis=0)]
    topics = []
    vocab_a = np.array(vocab)
    for t in range(k):
        order = np.argsort(theta[:, t])[::-1]
        topics.append({"topic": t, "share": float(shares[t]),
                       "top_terms": vocab_a[np.argsort(H[t])[::-1][:15]].tolist(),
                       "n_member": int(sup[t]),
                       "titles_1_10": [rows[i]["title"] for i in order[:10]],
                       "titles_11_30": [rows[i]["title"] for i in order[10:30]],
                       "abstract_only_cos": acos[t], "delta_pp": 100 * float(delta[t]),
                       "ci_low_pp": 100 * float(lo[t]), "ci_high_pp": 100 * float(hi[t]),
                       "drop_top5_pp": 100 * excl["drop_top5"][t], "drop_top10_pp": 100 * excl["drop_top10"][t],
                       "early_pct": 100 * float(theta[early, t].mean()), "late_pct": 100 * float(theta[late, t].mean()),
                       "count_early": float(theta[early, t].sum()), "count_late": float(theta[late, t].sum())})
    res = {"k": k, "n_eligible": n, "n_early": int(early.sum()), "n_late": int(late.sum()),
           "n_with_abstract": len(ab), "cluster_bootstrap": cmeta,
           "author_concentration": {kk: {"n": v, "share": v / n_all} for kk, v in conc.items()},
           "links_passing_rule": [l for l in links if l["passes_rule"]],
           "n_links_passing_rule_and_bonferroni": sum(l["passes_rule"] and l["passes_bonferroni"] for l in links),
           "links": links, "topics": topics,
           "robust_trends": [t["topic"] for t in topics if t["ci_low_pp"] > 0 or t["ci_high_pp"] < 0]}
    wjson(f"analysis_k{k}.json", res)
    return res


if __name__ == "__main__":
    rows = load()
    if len(sys.argv) > 1 and sys.argv[1] == "analyse":
        k = int(sys.argv[2])
    else:
        k = grid(rows)["chosen_k"]
    res = analyse(rows, k)
    for t in res["topics"]:
        print(t["topic"], f"{100*t['share']:.1f}%", t["top_terms"][:8], f"abs={t['abstract_only_cos']:.2f}",
              f"d={t['delta_pp']:+.1f} [{t['ci_low_pp']:+.1f};{t['ci_high_pp']:+.1f}]")
    print("links", [(l["i"], l["j"], l["joint"], l["lift"], l["passes_bonferroni"]) for l in res["links_passing_rule"]])
    print("conc", res["author_concentration"], res["cluster_bootstrap"])
