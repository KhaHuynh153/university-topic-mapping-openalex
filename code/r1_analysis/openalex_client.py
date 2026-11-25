"""Tiny cached OpenAlex client (no API key, no personal contact data)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
BASE = "https://api.openalex.org"
UA = "topic-map-replication/1.0"
# Optional key: environment variable OPENALEX_API_KEY or a one-line file .openalex_key next to
# this script (never copied to the repository). The key is added only to the request sent,
# never to the cache file name or the stored URL.
_KEY_FILE = Path(__file__).resolve().parent / ".openalex_key"
API_KEY = os.environ.get("OPENALEX_API_KEY") or (_KEY_FILE.read_text().strip() if _KEY_FILE.exists() else "")


def get(path: str, params: dict | None = None, use_cache: bool = True, pause: float = 0.15) -> dict:
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params, safe=":,|!*\"")
    key = hashlib.sha1(url.encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if use_cache and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    for attempt in range(8):
        try:
            send = url + (("&" if "?" in url else "?") + "api_key=" + API_KEY if API_KEY else "")
            req = urllib.request.Request(send, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            f.write_text(json.dumps({"_url": url, **data} if isinstance(data, dict) else data,
                                    ensure_ascii=False), encoding="utf-8")
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                body = e.read().decode("utf-8", "ignore")
                if "budget" in body.lower():
                    raise RuntimeError("OpenAlex daily budget exhausted; add an API key or retry after 00:00 UTC") from e
            if e.code == 404:
                data = {"_url": url, "_status": 404}
                f.write_text(json.dumps(data), encoding="utf-8")
                return data
            wait = 5 * (attempt + 1) if e.code in (429, 500, 502, 503) else 2
            time.sleep(wait)
        except Exception:  # network hiccup
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def cursor_all(filter_str: str, select: str | None = None, per_page: int = 200) -> list[dict]:
    out, cursor = [], "*"
    while cursor:
        params = {"filter": filter_str, "per-page": per_page, "cursor": cursor}
        if select:
            params["select"] = select
        data = get("/works", params)
        out.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not data.get("results"):
            break
    return out


def abstract_text(work: dict) -> str:
    inv = work.get("abstract_inverted_index") or {}
    if not inv:
        return ""
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))
