"""Build the HUB vs UEH comparison table (Bảng 7) -> ueh/out/summary_R1.json"""
from __future__ import annotations

import json
from pathlib import Path

R1 = Path(__file__).resolve().parent.parent
U = R1 / "ueh" / "out"


def vn(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def th(n):
    return f"{int(n):,}".replace(",", ".")


def main():
    scr = json.loads((R1 / "ueh" / "ueh_screening.json").read_text(encoding="utf-8"))
    grid = json.loads((U / "grid.json").read_text(encoding="utf-8"))
    k = grid["chosen_k"]
    a = json.loads((U / f"analysis_k{k}.json").read_text(encoding="utf-8"))
    g = next(r for r in grid["results"] if r["k"] == k)
    n_title = sum(c >= 0.5 for c in g["title_cos"])
    n_abs = sum(t["abstract_only_cos"] >= 0.5 for t in a["topics"])
    n_all = sum(g["title_cos"][t["topic"]] >= 0.5 and t["abstract_only_cos"] >= 0.5 and g["seed_cos"][t["topic"]] >= 0.5
                for t in a["topics"])
    npairs = k * (k - 1) // 2
    top10 = a["author_concentration"]["top10"]["share"]
    rows = [
        ["Số bài sau sàng lọc", "1.337", th(scr["n_corpus"])],
        ["Tỷ lệ bài có tóm tắt", "73,7%", vn(100 * scr["n_with_abstract"] / scr["n_corpus"]) + "%"],
        ["Tỷ lệ số bài có tác giả thuộc nhóm 10 tác giả nhiều bài nhất", "35,2%", vn(100 * top10) + "%"],
        ["Số chủ đề *K* được chọn", "8", str(k)],
        ["Số chủ đề xuất hiện lại khi chỉ dùng tiêu đề (cosin ≥ 0,50)", "8/8", f"{n_title}/{k}"],
        ["Số chủ đề xuất hiện lại khi chỉ dùng bài có tóm tắt (cosin ≥ 0,50)", "7/8", f"{n_abs}/{k}"],
        ["Số chủ đề vượt qua cả ba phép kiểm tra (cosin ≥ 0,50)", "7/8", f"{n_all}/{k}"],
        ["Mô hình thỏa luật chọn *K* định trước", "Có", "Không (dùng luật dự phòng)" if not grid["qualifying_k"] else "Có"],
        ["Số cặp chủ đề đạt quy tắc liên kết và kiểm định Bonferroni", "1/28", f"{int(a['n_links_passing_rule_and_bonferroni'])}/{npairs}"],
        ["Số chủ đề có khoảng bootstrap 95% của mức thay đổi không chứa 0", "2/8", f"{len(a['robust_trends'])}/{k}"],
    ]
    out = {"rows": rows, "k": k, "qualifying_k": grid["qualifying_k"], "n_title": n_title, "n_abs": n_abs}
    (U / "summary_R1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
