"""Precompute gallery artifacts for the retrained tri-modal model (ir_best_model.pt).

Streams public/data/data.parquet (numeric voxel_data + metadata), uses the
checkpoint's block_mapping + 3D-CNN voxel encoder to embed every build, and writes:
  public/data/gallery.json            metadata index (client + server)
  public/data/raw/<id>.bin            gzip(uint16 LE 32^3) real state-ids (3D preview)
  public/data/block_mapping.json      numeric block-id -> compact index (reference)
  model_server/gallery_voxel_emb.npy  float32 [N,256] L2-normalised voxel embeddings

Run inside the mcmodel conda env, from the web project root:
  python model_server/precompute.py
"""

import os
import sys
import json
import gzip

import numpy as np
import pyarrow.parquet as pq
import torch

import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trimodal2 import GRID, load_model, remap_voxel  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET = os.path.join(ROOT, "public/data/data.parquet")
CKPT = os.path.join(ROOT, "src/app/model/ir_best_model.pt")
OUT_DATA = os.path.join(ROOT, "public/data")
OUT_RAW = os.path.join(OUT_DATA, "raw")
OUT_EMB = os.path.join(os.path.dirname(__file__), "gallery_voxel_emb.npy")

VOXELS = GRID * GRID * GRID
BATCH = 48


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_tags(raw):
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()][:8]
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return [str(t).strip() for t in v if str(t).strip()][:8]
        except Exception:
            pass
        return [t.strip() for t in clean_text(raw).split(",") if t.strip()][:8]
    return []


def year_of(d):
    return int(d[:4]) if isinstance(d, str) and d[:4].isdigit() else None


def dims_fill(vd):
    vol = (np.asarray(vd, dtype=np.int64) != 0).reshape(GRID, GRID, GRID)
    coords = np.argwhere(vol)
    if coords.size == 0:
        return [0, 0, 0], 0.0
    mins, maxs = coords.min(0), coords.max(0) + 1
    return [int(maxs[0] - mins[0]), int(maxs[1] - mins[1]), int(maxs[2] - mins[2])], round(float(vol.sum()) / VOXELS, 4)


META = ["title", "subtitle", "description", "tags", "user", "date", "diamondCount", "views", "downloads", "url"]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading model + checkpoint ...")
    model, block_mapping, cfg = load_model(CKPT, device)
    crop_bbox = cfg["data"].get("crop_bbox", True)
    print(f"  block_mapping size: {len(block_mapping)}  crop_bbox={crop_bbox}")

    os.makedirs(OUT_RAW, exist_ok=True)
    for f in os.listdir(OUT_RAW):
        if f.endswith(".bin"):
            os.remove(os.path.join(OUT_RAW, f))

    pf = pq.ParquetFile(PARQUET)
    total = pf.metadata.num_rows
    print(f"Parquet rows: {total}. Encoding voxels ...")

    items, embs = [], []
    n = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=["voxel_data"] + META):
        rows = batch.to_pylist()
        keep, vox = [], []
        for row in rows:
            vd = row.get("voxel_data")
            if vd is None or len(vd) != VOXELS:
                continue
            keep.append(row)
            vox.append(remap_voxel(vd, block_mapping, crop_bbox=crop_bbox))
        if not vox:
            continue
        v = model.encode_voxel(torch.stack(vox).to(device)).cpu().numpy().astype("<f4")

        for row, emb in zip(keep, v):
            vd = np.asarray(row["voxel_data"], dtype=np.int64)
            sid = np.clip(vd, 0, 65535).astype("<u2")
            dims, fill = dims_fill(vd)
            bid = f"b{n:05d}"
            with open(os.path.join(OUT_RAW, f"{bid}.bin"), "wb") as fh:
                fh.write(gzip.compress(sid.tobytes(), 9))
            items.append({
                "id": bid,
                "title": clean_text(row.get("title")) or "Untitled Build",
                "category": clean_text(row.get("subtitle")) or "Other Map",
                "description": clean_text(row.get("description"))[:320],
                "tags": parse_tags(row.get("tags")),
                "user": clean_text(row.get("user")),
                "url": row.get("url") if isinstance(row.get("url"), str) else None,
                "img": None,
                "diamonds": int(row.get("diamondCount") or 0),
                "views": int(row.get("views") or 0),
                "downloads": int(row.get("downloads") or 0),
                "year": year_of(row.get("date")),
                "dims": dims,
                "fill": fill,
            })
            embs.append(emb)
            n += 1
        if n % 480 == 0:
            print(f"  {n}/{total}")

    emb_mat = np.stack(embs).astype("<f4")
    np.save(OUT_EMB, emb_mat)
    json.dump(block_mapping, open(os.path.join(OUT_DATA, "block_mapping.json"), "w"))
    json.dump({
        "meta": {"grid": GRID, "count": n, "embedDim": int(emb_mat.shape[1]),
                 "source": "Planet Minecraft (minecraft-schematics-mvm)",
                 "note": "Ranking by the retrained tri-modal model (TinyCLIP text/image + 3D-CNN voxel)."},
        "items": items,
    }, open(os.path.join(OUT_DATA, "gallery.json"), "w"), separators=(",", ":"))
    print(f"Done. builds={n}  emb={emb_mat.shape}  -> gallery.json, block_mapping.json, gallery_voxel_emb.npy, raw/*.bin")


if __name__ == "__main__":
    main()
