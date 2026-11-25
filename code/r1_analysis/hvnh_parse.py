"""Parse the Banking Academy (HVNH) yearly international-publication lists.

Input: hvnh/raw/hvnh_YYYY-NNNN.html (downloaded from hvnh.edu.vn, see download_date.txt)
Output: hvnh/hvnh_list.csv (one row per listed item)
"""
from __future__ import annotations

import csv
import html
import re
from pathlib import Path

R1 = Path(__file__).resolve().parent.parent
RAW = R1 / "hvnh" / "raw"


def clean(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def main() -> None:
    out = []
    for f in sorted(RAW.glob("hvnh_*.html")):
        year = int(f.stem.split("_")[1][:4])
        t = f.read_text(encoding="utf-8", errors="ignore")
        header = None
        for row in re.findall(r"<tr.*?</tr>", t, flags=re.S | re.I):
            cells = [clean(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)]
            if not cells or not any(cells):
                continue
            low = [c.lower() for c in cells]
            if any(("tên bài" in c or "tên công trình" in c or "tên bài báo" in c) for c in low):
                header = low
                continue
            if header is None or len(cells) < 3:
                continue
            first = cells[0]
            if not re.fullmatch(r"\d{1,4}\.?", first):
                continue
            rec = {"list_year": year, "stt": first.rstrip(".")}
            for name, c in zip(header, cells):
                if "tên bài" in name or "tên công trình" in name:
                    rec["title"] = c
                elif "tác giả" in name:
                    rec["authors"] = c
                elif "tạp chí" in name or "nơi công bố" in name:
                    rec["journal"] = c
                elif "thời gian" in name or "năm" in name:
                    rec["date"] = c
                elif "xếp hạng" in name or "danh mục" in name or "chỉ số" in name:
                    rec["rank"] = c
            if rec.get("title"):
                out.append(rec)
    cols = ["list_year", "stt", "title", "authors", "journal", "date", "rank"]
    with (R1 / "hvnh" / "hvnh_list.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(Counter(r["list_year"] for r in out))
    for y in sorted({r["list_year"] for r in out}):
        ex = next(r for r in out if r["list_year"] == y)
        print(y, ex)


if __name__ == "__main__":
    main()
