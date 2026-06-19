"""Export a gallery subset from data.parquet into a compact JSON the web demo consumes.

Each schematic's 32x32x32 voxel grid is preprocessed exactly like the training
pipeline (top-254 block mapping, bounding-box crop, nearest-neighbour resize to 32^3),
then stored as base64( gzip( uint8 grid ) ) -- ~400 bytes per build thanks to the large
contiguous runs the NN-resize produces. The browser/Node decompresses it natively with
DecompressionStream('gzip'), so the demo ships with zero extra runtime dependency for voxels.

Usage:
    python scripts/export_gallery.py --n 240 --out public/data/gallery.json
"""

import argparse
import base64
import gzip
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

GRID = 32
MAX_BLOCK_TYPES = 256


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_tags(raw) -> list[str]:
    if not isinstance(raw, str):
        return []
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(t).strip() for t in val if str(t).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return [t.strip() for t in clean_text(raw).split(",") if t.strip()]


def build_text(row) -> str:
    """Mirror src/dataset.py:build_text — title + subtitle + description + tags."""
    parts = []
    for field in ("title", "subtitle", "description"):
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(clean_text(val))
    tags = parse_tags(row.get("tags"))
    if tags:
        parts.append(", ".join(tags))
    return " ".join(parts)


def build_block_mapping(voxel_series, max_types=MAX_BLOCK_TYPES):
    """Top-(max_types-2) frequent non-air blocks. 0=air, 1=rare, 2..=frequent."""
    counter: Counter = Counter()
    for vd in voxel_series:
        counter.update(np.asarray(vd).tolist())
    counter.pop(0, None)
    top = [b for b, _ in counter.most_common(max_types - 2)]
    mapping = {0: 0}
    for i, bid in enumerate(top, start=2):
        mapping[bid] = i
    return mapping


def preprocess_voxel(flat, mapping):
    """Remap -> bbox crop -> NN-resize to 32^3. Returns (uint8 grid 32^3, raw_dims[w,d,h])."""
    arr = np.asarray(flat, dtype=np.int64)
    remapped = np.array([mapping.get(v, 1) for v in arr], dtype=np.int64).reshape(GRID, GRID, GRID)
    non_air = remapped != 0
    dims = [GRID, GRID, GRID]
    if non_air.any():
        coords = np.argwhere(non_air)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        cropped = remapped[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        dims = [int(d) for d in cropped.shape]
        factors = (GRID / cropped.shape[0], GRID / cropped.shape[1], GRID / cropped.shape[2])
        remapped = zoom(cropped, factors, order=0)
    return remapped.astype(np.uint8), dims


def encode_grid(grid: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(grid.tobytes(), 9)).decode("ascii")


def select_subset(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Pick a diverse, reasonably high-quality subset balanced across subtitle categories."""
    df = df.copy()
    df["subtitle"] = df["subtitle"].fillna("Other Map")
    # quality score: favour builds people engaged with, but keep variety
    df["_score"] = (
        np.log1p(df["diamondCount"].clip(lower=0))
        + 0.5 * np.log1p(df["downloads"].clip(lower=0))
        + 0.3 * np.log1p(df["views"].clip(lower=0))
    )
    cats = df["subtitle"].value_counts()
    per_cat = max(4, n // max(1, len(cats)))
    picked = []
    rng = np.random.default_rng(seed)
    for cat in cats.index:
        sub = df[df["subtitle"] == cat].sort_values("_score", ascending=False)
        # take the strongest, plus a couple random ones for diversity
        head = sub.head(per_cat)
        picked.append(head)
    out = pd.concat(picked).drop_duplicates(subset=["url"]).sort_values("_score", ascending=False)
    if len(out) > n:
        out = out.head(n)
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="../data/data.parquet")
    ap.add_argument("--out", default="public/data/gallery.json")
    ap.add_argument("--n", type=int, default=240)
    args = ap.parse_args()

    print(f"Loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df)} records")

    print("Building block mapping (top-254) ...")
    mapping = build_block_mapping(df["voxel_data"])

    print(f"Selecting ~{args.n} diverse builds ...")
    subset = select_subset(df, args.n)
    print(f"  selected {len(subset)} builds across {subset['subtitle'].nunique()} categories")

    items = []
    for i, row in subset.iterrows():
        grid, dims = preprocess_voxel(row["voxel_data"], mapping)
        nonair = int((grid != 0).sum())
        if nonair < 20:
            continue  # skip near-empty
        date = row.get("date")
        year = None
        if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
            year = int(date[:4])
        items.append({
            "id": f"b{i:04d}",
            "title": clean_text(row.get("title")) or "Untitled Build",
            "category": clean_text(row.get("subtitle")) or "Other Map",
            "description": clean_text(row.get("description"))[:320],
            "tags": parse_tags(row.get("tags"))[:8],
            "text": build_text(row),
            "user": clean_text(row.get("user")),
            "url": row.get("url") if isinstance(row.get("url"), str) else None,
            "img": row.get("img") if isinstance(row.get("img"), str) else None,
            "diamonds": int(row.get("diamondCount") or 0),
            "views": int(row.get("views") or 0),
            "downloads": int(row.get("downloads") or 0),
            "year": year,
            "dims": dims,             # original cropped extents [w, d, h] (pre-resize)
            "fill": round(nonair / (GRID ** 3), 4),
            "voxels": encode_grid(grid),  # base64(gzip(uint8 32^3))
        })

    payload = {
        "meta": {
            "grid": GRID,
            "count": len(items),
            "maxBlockTypes": MAX_BLOCK_TYPES,
            "source": "Planet Minecraft (rom1504/minecraft-schematics-dataset)",
            "note": "Voxels preprocessed identically to the training pipeline (top-254 mapping, bbox crop, NN-resize to 32^3).",
        },
        "items": items,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    import os
    size = os.path.getsize(args.out)
    print(f"Wrote {len(items)} builds -> {args.out} ({size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
