"""R1 step 1b: where do the liquidity (CĐ7, data index 2) papers go in the
abstract-only re-fit, where the liquidity topic does not reappear (cosine 0.063)?
Output: out/liquidity_check.json
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from r1_common import BASE, CODE, hub_rows, hub_vocabulary, row_cosine, write_json

LIQ, CB = 2, 4


def main() -> None:
    rows = hub_rows()
    vocab = np.array(hub_vocabulary())
    theta = np.load(BASE / "theta_k8.npy")
    H = np.load(BASE / "components_k8.npy")
    Ha = np.load(BASE / "components_abstract_subset_k8.npy")
    ta = np.load(BASE / "theta_abstract_subset_aligned_k8.npy")   # columns aligned to main topics
    ab = [i for i, r in enumerate(rows) if r["has_abstract"]]
    assert ta.shape[0] == len(ab)
    sim = row_cosine(H, Ha)
    # which abstract-model topic sits in the aligned "liquidity" slot, and its best match
    order = []
    for col in range(8):
        # recover alignment: find the Ha row whose theta column equals ta[:, col]
        order.append(int(np.argmax(sim[col])) if False else None)
    best_for_liq = int(np.argmax(sim[LIQ]))
    th = theta[ab]
    liq_members = th[:, LIQ] >= 0.20
    shared = liq_members & (th[:, CB] >= 0.20)
    dom_liq = Counter(CODE[int(np.argmax(ta[i]))] for i in np.where(liq_members)[0])
    dom_shared = Counter(CODE[int(np.argmax(ta[i]))] for i in np.where(shared)[0])
    mean_liq = {CODE[c]: round(float(ta[liq_members, c].mean()), 3) for c in range(8)}
    res = {
        "n_abstract_docs": len(ab),
        "liquidity_members_in_abstract_subset": int(liq_members.sum()),
        "shared_liq_cb_in_abstract_subset": int(shared.sum()),
        "cosine_main_liq_vs_each_abstract_topic": [round(float(x), 3) for x in sim[LIQ]],
        "cosine_main_cb_vs_each_abstract_topic": [round(float(x), 3) for x in sim[CB]],
        "best_abstract_topic_for_liq": best_for_liq,
        "best_abstract_topic_for_cb": int(np.argmax(sim[CB])),
        "abstract_topic_terms": {i: vocab[np.argsort(Ha[i])[::-1][:10]].tolist() for i in range(8)},
        "dominant_topic_of_liq_members_in_abstract_model": dict(dom_liq),
        "dominant_topic_of_shared_in_abstract_model": dict(dom_shared),
        "mean_aligned_theta_of_liq_members": mean_liq,
        "main_liq_terms": vocab[np.argsort(H[LIQ])[::-1][:10]].tolist(),
    }
    write_json("liquidity_check.json", res)
    import json
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
