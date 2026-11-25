"""Shared helpers for the R1 revision analyses.

Reads the locked HUB results from the parent folder (never writes there) and
writes every new output into R1_revise_2026-11-25/analysis/out/.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
R1 = HERE.parent
BASE = R1.parent                       # topic7_hub_refresh_2026-10-26
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE))
import refresh_hub  # noqa: E402  (patches the tokenizer with place stop words)
import network_hub  # noqa: E402

original = refresh_hub.original
tokenize = refresh_hub.original.tokenize

# Manuscript order CĐ1..CĐ8 -> data index
PAPER_TO_DATA = [6, 1, 0, 7, 4, 5, 2, 3]
SHORT_VI = {6: "Tăng trưởng", 1: "Người tiêu dùng", 0: "Giáo dục", 7: "Chứng khoán",
            4: "Ngân hàng thương mại", 5: "Toán học", 2: "Thanh khoản", 3: "Tài chính*"}
CODE = {d: f"CĐ{i+1}" for i, d in enumerate(PAPER_TO_DATA)}


def write_json(name: str, data: object) -> None:
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(name: str, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or list(rows[0].keys())
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def hub_rows() -> list[dict]:
    return refresh_hub.corpus_rows()


def hub_vocabulary() -> list[str]:
    return json.loads((BASE / "vocabulary.json").read_text(encoding="utf-8"))


def count_matrix(rows: list[dict], vocabulary: list[str], mode: str = "combined") -> np.ndarray:
    index = {t: i for i, t in enumerate(vocabulary)}
    X = np.zeros((len(rows), len(vocabulary)), dtype=np.float64)
    for d, r in enumerate(rows):
        text = r["title"] if mode == "title" else r["text"]
        for term, n in Counter(tokenize(text)).items():
            j = index.get(term)
            if j is not None:
                X[d, j] = n
    return X


def hungarian(sim: np.ndarray) -> list[int]:
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-sim)
    order = [0] * sim.shape[0]
    for i, j in zip(r, c):
        order[i] = int(j)
    return order


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T
