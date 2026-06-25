"""Precompute gallery artifacts for the model-backed search.

Streams the training parquet (data_with_voxel_names_multiview_image.parquet),
builds the exact name2id vocab, loads the trained TriModalEncoder, and writes:
  public/data/gallery.json            metadata index (client + server)
  public/data/raw/<id>.bin            gzip(uint16 LE 32^3) real state-ids (3D preview)
  public/data/name2id.json            block-name -> vocab index (Next voxel-query mapping)
  model_server/gallery_voxel_emb.npy  float32 [N,256] L2-normalised voxel embeddings

Run inside the mcmodel conda env, from the web project root:
  python model_server/precompute.py
"""

import os
import sys
import json
import gzip
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trimodal import (  # noqa: E402
    CFG, GRID, TriModalEncoder, build_name_vocab, remap_voxel_names, clean_text,
)
from transformers import CLIPProcessor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET = "/Users/rasyadlubis/ProjectRasyad/college/semester 6/retrieval-information/minecraft-schematics-dataset/data_with_voxel_names_multiview_image.parquet"
CKPT = os.path.join(ROOT, "src/app/model/schematics_best_model.pth")
OUT_DATA = os.path.join(ROOT, "public/data")
OUT_RAW = os.path.join(OUT_DATA, "raw")
OUT_EMB = os.path.join(os.path.dirname(__file__), "gallery_voxel_emb.npy")

VOXELS = GRID * GRID * GRID
BATCH = 48
META_COLS = ["title", "subtitle", "description", "tags", "user", "date",
             "diamondCount", "views", "downloads", "url"]

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

name2stateid = json.load(open(os.path.join(OUT_DATA, "name2stateid.json")))


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


def year_of(date):
    return int(date[:4]) if isinstance(date, str) and date[:4].isdigit() else None


def grid_state_ids(names):
    out = np.zeros(VOXELS, dtype=np.uint16)
    for i, n in enumerate(names):
        if isinstance(n, str) and "air" not in n:
            out[i] = name2stateid.get(n, 1)
    return out


def dims_fill(state_ids):
    vol = (state_ids != 0).reshape(GRID, GRID, GRID)
    coords = np.argwhere(vol)
    if coords.size == 0:
        return [0, 0, 0], 0.0
    mins = coords.min(0); maxs = coords.max(0) + 1
    dims = [int(maxs[0] - mins[0]), int(maxs[1] - mins[1]), int(maxs[2] - mins[2])]
    return dims, round(float(vol.sum()) / VOXELS, 4)


def main():
    pf = pq.ParquetFile(PARQUET)
    total = pf.metadata.num_rows
    print(f"Parquet rows: {total}")

    # ── Pass 1: build name2id vocab (stream voxel_name_data) ──
    print("Pass 1/2: building name vocab ...")
    counter = Counter()
    for batch in pf.iter_batches(batch_size=256, columns=["voxel_name_data"]):
        for vnd in batch.column("voxel_name_data").to_pylist():
            if vnd:
                counter.update(n for n in vnd if isinstance(n, str) and n != "minecraft:air")
    top = [b for b, _ in counter.most_common(CFG["max_block_types"] - 2)]
    name2id = {"minecraft:air": 0, "<rare>": 1}
    for i, n in enumerate(top, start=2):
        name2id[n] = i
    print(f"  vocab size: {len(name2id)}")

    # ── Build model + load checkpoint ──
    print("Loading checkpoint + model ...")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    ckpt_vocab = state["voxel_encoder.block_embedding.weight"].shape[0]
    print(f"  checkpoint vocab: {ckpt_vocab}  (built: {len(name2id)})")
    assert ckpt_vocab == len(name2id), "vocab mismatch — name2id must match training"

    processor = CLIPProcessor.from_pretrained(CFG["tinyclip_hf_model"])
    model = TriModalEncoder(CFG, len(name2id), processor)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if "proj" in k or "voxel_encoder" in k]
    print(f"  loaded. missing={len(missing)} unexpected={len(unexpected)} | trained-keys missing: {len(bad)}")
    assert not bad, f"trained weights failed to load: {bad[:5]}"
    model.to(device).eval()

    os.makedirs(OUT_RAW, exist_ok=True)
    # clear stale raw bins from the previous gallery
    for f in os.listdir(OUT_RAW):
        if f.endswith(".bin"):
            os.remove(os.path.join(OUT_RAW, f))

    # ── Pass 2: per build → voxel embedding + raw bin + metadata ──
    print("Pass 2/2: encoding voxels + writing assets ...")
    items, embs = [], []
    n = 0
    cols = ["voxel_name_data"] + META_COLS
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        rows = batch.to_pylist()
        vox_tensors, idxs = [], []
        for row in rows:
            vnd = row.get("voxel_name_data")
            if not vnd or len(vnd) != VOXELS:
                continue
            idxs.append(row)
            vox_tensors.append(remap_voxel_names(vnd, name2id, crop_bbox=CFG["crop_bbox"]))
        if not vox_tensors:
            continue
        with torch.no_grad():
            v = model.encode_voxel(torch.stack(vox_tensors).to(device)).cpu().numpy().astype("<f4")

        for row, emb in zip(idxs, v):
            sid = grid_state_ids(row["voxel_name_data"])
            dims, fill = dims_fill(sid)
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
    json.dump(name2id, open(os.path.join(OUT_DATA, "name2id.json"), "w"))
    json.dump({
        "meta": {"grid": GRID, "count": n, "embedDim": CFG["embed_dim"],
                 "source": "Planet Minecraft (minecraft-schematics-mvm)",
                 "note": "Searchable gallery; ranking by the trained tri-modal model (voxel embeddings)."},
        "items": items,
    }, open(os.path.join(OUT_DATA, "gallery.json"), "w"), separators=(",", ":"))

    print(f"Done. builds={n}  emb={emb_mat.shape}  -> gallery.json, name2id.json, {os.path.basename(OUT_EMB)}, raw/*.bin")


if __name__ == "__main__":
    main()
