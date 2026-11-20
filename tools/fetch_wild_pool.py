"""Fetch a larger pool of open-licensed clean car photos for hard-negative mining.

Pulls files from the given Wikimedia Commons categories (and one level of
subcategories), drops titles that suggest damage, removes anything already in
an existing manifest, and splits by photo series (train / test). The test split
is kept untouched for the final false-positive measurement.

Usage: python tools/fetch_wild_pool.py OUT_DIR --exclude manifest.jsonl [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_wild_clean import SESSION, get, image_info, series_key  # noqa: E402

CATEGORIES = [
    # Indian-market models (closest to the target fleet)
    "Maruti Suzuki Swift", "Maruti Suzuki Dzire", "Maruti Suzuki Baleno", "Maruti Suzuki Ertiga",
    "Maruti Suzuki Wagon R", "Maruti Suzuki Alto", "Maruti Suzuki Vitara Brezza", "Maruti Suzuki Ciaz",
    "Maruti Suzuki Celerio", "Suzuki Swift", "Hyundai Creta", "Hyundai i20", "Hyundai Venue", "Tata Nexon",
    "Tata Punch", "Tata Tiago", "Mahindra Scorpio", "Mahindra XUV700", "Mahindra Thar", "Honda City",
    "Toyota Innova", "Kia Seltos", "Kia Sonet", "Renault Kwid",
    # reflective paint and common colours
    "Silver cars", "White cars", "Black cars", "Grey cars", "Red cars", "Blue cars",
    # everyday scenes
    "Parked cars", "Cars in India", "Taxis in India", "Car dealerships",
]
DAMAGE_WORDS = re.compile(
    r"crash|accident|damag|wreck|rust|burn|abandon|derelict|destroy|scrap|junk|broken|dent|collision|"
    r"flood|fire|hulk|vandal|smash|totaled|totalled|ruin|decay|graveyard", re.I)


def files_in(category: str, depth: int = 1) -> list[str]:
    titles, token = [], None
    while True:
        params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                  "cmtype": "file|subcat", "cmlimit": "500"}
        if token:
            params["cmcontinue"] = token
        data = get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            if m["ns"] == 6:
                titles.append(m["title"])
            elif m["ns"] == 14 and depth > 0 and not DAMAGE_WORDS.search(m["title"]):
                titles += files_in(m["title"].removeprefix("Category:"), depth - 1)
        token = data.get("continue", {}).get("cmcontinue")
        if not token or len(titles) > 4000:
            return titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--limit", type=int, default=2200)
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--categories", help="comma-separated override of the built-in list")
    ap.add_argument("--prefix", default="pool")
    args = ap.parse_args()
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    seen = set()
    for path in args.exclude:
        seen |= {json.loads(line)["title"] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    rng = random.Random(args.seed)
    picked: list[str] = []
    for category in (args.categories.split(",") if args.categories else CATEGORIES):
        try:
            titles = [t for t in set(files_in(category)) if t not in seen and not DAMAGE_WORDS.search(t)]
        except Exception as exc:  # a missing category must not stop the run
            print("skip", category, exc, flush=True)
            continue
        rng.shuffle(titles)
        take = titles[: args.per_category]
        seen |= set(take)
        picked += take
        print(f"{category}: {len(titles)} candidates, took {len(take)}", flush=True)
    rng.shuffle(picked)
    rows = image_info(picked[: int(args.limit * 1.3)])[: args.limit]

    def download(item):
        i, row = item
        path = out / "images" / f"{args.prefix}_{i:05d}.jpg"
        if not path.is_file():
            for attempt in range(4):
                try:
                    r = SESSION.get(row["url"], timeout=90)
                    if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                        path.write_bytes(r.content)
                        break
                except Exception:
                    pass
                time.sleep(3 * (attempt + 1))
            else:
                return None
        bucket = int(hashlib.sha256(series_key(row["title"]).encode()).hexdigest(), 16) % 100
        return {**row, "image": path.as_posix(), "group_id": f"{args.prefix}_{i:05d}",
                "split": "test" if bucket < 20 else "train", "series": series_key(row["title"])}

    with ThreadPoolExecutor(3) as pool:
        done = [d for d in pool.map(download, list(enumerate(rows))) if d]
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(d) for d in done), encoding="utf-8")
    print(json.dumps({"downloaded": len(done), "train": sum(d["split"] == "train" for d in done),
                      "test": sum(d["split"] == "test" for d in done)}))


if __name__ == "__main__":
    main()
