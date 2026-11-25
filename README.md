# Replication package: university-level topic mapping from OpenAlex (coverage and reliability)

This repository accompanies the manuscript *"Bản đồ chủ đề nghiên cứu cấp trường từ OpenAlex: độ bao phủ và độ tin cậy tại Việt Nam"* (Mapping University Research Topics from OpenAlex: Coverage and Reliability in Vietnam). It is anonymised for peer review.

## Quick start

```text
python -m pip install -r requirements.txt
python code/reproduce_main_results.py
```

`reproduce_main_results.py` starts from the released term-count matrix and reproduces the main results for Ho Chi Minh University of Banking (HUB). It refits NMF with K = 8 and seeds 0–9, then prints the 8 topics with their shares and top terms, the topic pairs that pass the link rule (51 shared papers, lift 1.88), and the share changes from 2016–2019 to 2022–2025 with author-cluster bootstrap intervals. It runs in a few minutes on a laptop.

## Data sources and query dates

| Source | What | Date |
|---|---|---|
| OpenAlex API | works with an author linked to HUB, `institutions.id:I3129629310` (ROR 03fsckq75), publication years 2010–2025 | 24 Oct 2026 |
| HUB website | list of 315 international journal articles 2020–2024 | 24 Oct 2026 |
| HVNH (Banking Academy of Vietnam) website | yearly lists of international publications 2020–2025, `hvnh.edu.vn/hvnh/vi/cong-bo/cong-bo-quoc-te-nam-YYYY-*.html` | 25 Nov 2026 |
| OpenAlex API | works linked to HVNH, `institutions.id:I4403386521`, 2016–2025 | see `data/r1_outputs/list_coverage.json` |
| OpenAlex API | works linked to UEH (University of Economics Ho Chi Minh City), `institutions.id:I181727666` (ROR 010yce376), 2016–2025 | see `data/ueh/ueh_screening.json` |

OpenAlex metadata is released under CC0. The university lists are public web pages; `data/hvnh_list.csv` and `data/r1_outputs/list_coverage_HUB.csv` reproduce only their bibliographic fields (title, journal, year).

Abstracts are **not** redistributed. Abstracts of the 123 papers added from the HUB list came from publisher pages and may be copyrighted. The analysis therefore starts from the term-count matrix, which reproduces the locked TF–IDF matrix exactly (maximum absolute difference 0.0).

## Files

### `data/`
- `hub_papers.csv`: the 1,337 HUB papers, in the row order of the matrices. Columns: OpenAlex ID, DOI, year, source (`openalex` or `hub_list`), affiliation evidence for list additions (`strong` = publisher page or PDF shows the HUB affiliation; `moderate` = bibliographic match with the official list), abstract flag, possible-duplicate flag, author cluster used by the bootstrap, and, for papers without any identifier, the title and an evidence URL.
- `hub_counts_combined.npz`, `hub_counts_title.npz`: sparse matrices of term counts (1,337 papers × 3,000 terms) for title + abstract and for titles only.
- `hub_vocabulary.json`: the 3,000 terms (unigrams and bigrams).
- `model_lock.json`, `grid_results.json`, `extra_kgrid.json`, `sensitivity_k8.json`, `topics_k8.json`: settings, seeds (0–9; chosen seed 3), K grid, sensitivity re-fits, and the locked topic catalogue with SHA-256 hashes of the defining files.
- `possible_duplicate_versions.csv`: 13 same-title, same-year pairs (sensitivity check).
- `extra_name_variants.csv`: English name variants of HUB in OpenAlex raw affiliation strings, by period.
- `hvnh_list.csv`: the parsed HVNH lists.
- `r1_outputs/`: outputs of the revision analyses: LDA comparison (`lda_*`), label validation (`label_check_HUB.*`), model-free keyword trend (`keyword_trend.json`), liquidity-topic check (`liquidity_check.json`), and list-based coverage (`list_coverage*`).
- `ueh/`: UEH identifiers, screening summary and topic-model outputs.

### `code/`
- `reproduce_main_results.py`: self-contained reproduction of the main HUB results from `data/`.
- `pipeline_v19/`: the scripts that produced the locked HUB results (`refresh_hub.py`, `network_hub.py`, `extra_analyses.py`, `model_topics.py`). They read the raw OpenAlex download from the original project folder tree and are included for transparency.
- `r1_analysis/`: the revision analyses:
  - `lda_check.py`, `lda_seeds.py`: LDA with K = 8, batch variational Bayes (scikit-learn), 200 iterations, seeds 0–9, matched to NMF topics by term-vector cosine.
  - `label_check.py`: validation of the topic names on the papers ranked 11–30.
  - `keyword_trend.py`: model-free trend check based on keywords in titles.
  - `liquidity_check.py`: where the liquidity-topic papers go in the abstract-only re-fit.
  - `hvnh_parse.py`, `list_coverage.py`: automatic list-based coverage for HUB and HVNH.
  - `ueh_download.py`, `ueh_topics.py`: download, screening and topic mapping for UEH (K = 5..15; pre-specified rule: among models whose minimum title-only cosine is at least 0.50, choose the highest mean NPMI).
  - `openalex_client.py`: a small cached client. OpenAlex now meters anonymous use, so an API key may be needed.

## Main settings (HUB)

- Text: title + abstract. Lower-cased, stop words removed (common words, generic academic words, place names and university names), unigrams + bigrams. Terms kept: document frequency ≥ 3 and ≤ 85%, top 3,000 by total frequency.
- TF–IDF: `tf × (ln((1+N)/(1+df)) + 1)`, rows scaled to unit length.
- NMF: multiplicative updates, K = 8, seeds 0–9, at most 160 iterations, convergence checked every 20 iterations from iteration 100 (tolerance 1e-6); the seed with the smallest relative error is kept (seed 3).
- Topic membership: θ ≥ 0.20. Link rule: at least 10 shared papers and lift ≥ 1.20, with a hypergeometric test (Bonferroni over 28 pairs).
- Temporal contrast: 2016–2019 vs 2022–2025. Author-cluster bootstrap with 2,000 draws (seed 20261027); 520 clusters.

## Software

Python 3.12 with numpy 2.2.6, scipy 1.14.1, scikit-learn 1.6.1 and joblib 1.4.2 (see `requirements.txt`).

## Licence

Code: MIT. Data derived from OpenAlex: CC0. University list fields are reproduced from the public web pages named above.
