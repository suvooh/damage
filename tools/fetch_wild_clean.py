"""Fetch open-licensed car photos from Wikimedia Commons as 'wild' clean images.

These are curated (featured/quality/valued) automobile photos: classic cars,
chrome, car shows, glare, clutter -- the conditions the COCO clean set lacks.
Damage is not labelled; images are assumed clean and any model alert is
reviewed by eye before being counted as a false positive. Licence and
attribution are recorded per image. Splits are by photo series (file-name stem)
so near-duplicate shots of one car never straddle train/validation/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "VehicleDamageDev/1.0 (local research; chaman.kumar01@infosys.com)"}
CATEGORIES = [
    "Featured pictures of automobiles",
    "Quality images of automobiles",
    "Valued images of automobiles",
]
OK_LICENSES = ("cc by", "cc0", "public domain", "pd")
SESSION = requests.Session()
SESSION.headers.update(UA)


def get(params: dict) -> dict:
    for attempt in range(5):
        r = SESSION.get(API, params={**params, "format": "json"}, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return {}


def category_files(category: str) -> list[str]:
    titles, token = [], None
    while True:
        params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                  "cmtype": "file", "cmlimit": "500"}
        if token:
            params["cmcontinue"] = token
        data = get(params)
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        token = data.get("continue", {}).get("cmcontinue")
        if not token:
            return titles


def image_info(titles: list[str]) -> list[dict]:
    rows = []
    for i in range(0, len(titles), 40):
        data = get({"action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime", "iiurlwidth": "1280"})
        for page in data["query"]["pages"].values():
            info = (page.get("imageinfo") or [None])[0]
            if not info or info.get("mime") not in ("image/jpeg",):
                continue
            meta = info.get("extmetadata", {})
            license_name = meta.get("LicenseShortName", {}).get("value", "")
            if not license_name.lower().startswith(OK_LICENSES):
                continue
            rows.append({
                "title": page["title"], "url": info.get("thumburl") or info["url"],
                "page": info["descriptionurl"], "license": license_name,
                "artist": re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:120],
            })
    return rows


def series_key(title: str) -> str:
    stem = re.sub(r"^File:", "", title).rsplit(".", 1)[0]
    return re.sub(r"[\d_\-\s()]+", " ", stem).strip().lower()[:40]


def split_for(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 60 else "validation" if bucket < 75 else "calibration" if bucket < 85 else "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=450)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    titles = sorted({t for c in CATEGORIES for t in category_files(c)})
    random.Random(args.seed).shuffle(titles)
    print("candidate files", len(titles), flush=True)
    rows = image_info(titles[: args.limit * 2])
    rows = rows[: args.limit]

    def download(item):
        i, row = item
        path = out / "images" / f"wild_{i:04d}.jpg"
        if not path.is_file():
            for attempt in range(4):
                r = SESSION.get(row["url"], timeout=90)
                if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                    path.write_bytes(r.content)
                    break
                time.sleep(3 * (attempt + 1))
            else:
                return None
        return {**row, "image": path.as_posix(), "group_id": f"wild_{i:04d}",
                "split": split_for(series_key(row["title"])), "series": series_key(row["title"])}

    with ThreadPoolExecutor(3) as pool:
        done = [d for d in pool.map(download, list(enumerate(rows))) if d]
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(d) for d in done), encoding="utf-8")
    counts = {s: sum(d["split"] == s for d in done) for s in ("train", "validation", "calibration", "test")}
    print(json.dumps({"downloaded": len(done), "splits": counts}))


if __name__ == "__main__":
    main()
