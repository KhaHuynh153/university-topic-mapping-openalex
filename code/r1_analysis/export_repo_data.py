"""R1 step 3: export the data files of the public replication repository.

No abstracts or titles of the 123 list-based additions are redistributed as text; the
replication starts from the term-count matrix (terms x papers), which is a derived,
non-expressive representation. OpenAlex metadata is CC0.
Outputs go to repo_anonymous/data/.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse

from r1_common import BASE, R1, count_matrix, hub_rows, hub_vocabulary, network_hub

REPO = R1 / "repo_anonymous"
DATA = REPO / "data"
DATA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = hub_rows()
    vocab = hub_vocabulary()
    # counts for the combined text and for titles only (sensitivity), same vocabulary
    for mode in ("combined", "title"):
        X = count_matrix(rows, vocab, mode)
        sparse.save_npz(DATA / f"hub_counts_{mode}.npz", sparse.csr_matrix(X.astype(np.int32)))
    (DATA / "hub_vocabulary.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    dups = set()
    with (BASE / "possible_duplicate_versions.csv").open(encoding="utf-8-sig") as h:
        for r in csv.DictReader(h):
            for k, v in r.items():
                if "id" in k.lower() and v:
                    dups.add(v)
    eligible = [r for r in rows]
    clusters, _ = network_hub.author_leader_clusters(eligible, bundle_additions=False)
    with (DATA / "hub_papers.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        w.writerow(["row", "openalex_id", "doi", "year", "source", "affiliation_evidence", "has_abstract",
                    "possible_duplicate_version", "author_cluster", "title_if_no_identifier", "evidence_url"])
        for i, (r, c) in enumerate(zip(rows, clusters)):
            oa = r["id"] if "openalex.org/W" in r["id"] else ""
            doi = r.get("doi") or (r["id"].replace("https://doi.org/", "") if "doi.org" in r["id"] else "")
            no_id = not oa and not doi
            w.writerow([i, oa, doi, r["year"], "openalex" if r["source"] == "inherited_openalex" else "hub_list",
                        r.get("affiliation_evidence", ""), int(bool(r["has_abstract"])),
                        int(r["id"] in dups), c, r["title"] if no_id else "",
                        r.get("primary_evidence_url", "") if r["source"] != "inherited_openalex" else ""])
    for name in ("model_lock.json", "grid_results.json", "extra_kgrid.json", "sensitivity_k8.json",
                 "topics_k8.json", "possible_duplicate_versions.csv"):
        shutil.copy(BASE / name, DATA / name)
    print("rows", len(rows), "files", sorted(p.name for p in DATA.iterdir()))


if __name__ == "__main__":
    main()
