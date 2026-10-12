"""Polite scraper for PUBLIC Telegram channels via the t.me/s web preview.

Stores (id, published_at UTC, text) per post. The publication timestamp is the only time we
use, so a feature built at origin t may only include posts with published_at < t.
Caveat: the preview shows the *current* text of a post, which may have been edited later.
"""
import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (academic research, low-rate)"}
POST = re.compile(r'<div class="tgme_widget_message_wrap.*?(?=<div class="tgme_widget_message_wrap|\Z)', re.S)


def _page(channel, before=None):
    url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode()
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def _parse(page, channel):
    out = []
    for blk in POST.findall(page):
        m = re.search(r'data-post="[^/]+/(\d+)"', blk)
        t = re.search(r'<time datetime="([^"]+)"', blk)
        if not (m and t):
            continue
        txt = re.search(r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', blk, re.S)
        body = re.sub(r"<br\s*/?>", " ", txt.group(1)) if txt else ""
        body = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        out.append({"id": int(m.group(1)), "published_at": t.group(1), "text": body})
    return out


def scrape(channel, since, delay=0.6):
    path = RAW / f"telegram_{channel}.jsonl"
    RAW.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line); done[r["id"]] = r
    since = pd.Timestamp(since, tz="UTC")
    before = min(done) if done and pd.Timestamp(min(done.values(), key=lambda r: r["id"])["published_at"]) > since else None
    if not done:
        before = None
    pages = 0
    with path.open("a", encoding="utf-8") as f:
        while True:
            posts = _parse(_page(channel, before), channel)
            if not posts:
                break
            new = [p for p in posts if p["id"] not in done]
            for p in new:
                done[p["id"]] = p
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            f.flush()
            oldest = min(posts, key=lambda p: p["id"])
            pages += 1
            if pages % 20 == 0:
                print(channel, pages, oldest["published_at"], len(done), flush=True)
            if pd.Timestamp(oldest["published_at"]) <= since or (before is not None and oldest["id"] >= before):
                break
            before = oldest["id"]
            time.sleep(delay)
    print("done", channel, len(done), flush=True)


def load(channel) -> pd.DataFrame:
    df = pd.read_json(RAW / f"telegram_{channel}.jsonl", lines=True)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df.drop_duplicates("id").sort_values("published_at").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("channel")
    ap.add_argument("--since", default="2025-12-01")
    a = ap.parse_args()
    scrape(a.channel, a.since)
