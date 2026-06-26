"""Inference sidecar for the retrained tri-modal model (ir_best_model.pt).

Loads the model once + the precomputed gallery voxel embeddings, and ranks the
gallery for text / image / voxel queries in the shared 256-d space. The Next.js
API routes proxy here; the browser never talks to this service directly.

Run inside the mcmodel conda env, from the web project root:
  python model_server/serve.py            # http://127.0.0.1:8008
"""

import os
import sys
import io
import json
import gzip
import base64
import time

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trimodal2 import GRID, load_model, remap_voxel  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ROOT, "src/app/model/ir_best_model.pt")
OUT_DATA = os.path.join(ROOT, "public/data")
EMB_PATH = os.path.join(os.path.dirname(__file__), "gallery_voxel_emb.npy")
VOXELS = GRID * GRID * GRID

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

print("Loading gallery + model ...")
gallery = json.load(open(os.path.join(OUT_DATA, "gallery.json")))
IDS = [it["id"] for it in gallery["items"]]
EMB = np.load(EMB_PATH).astype(np.float32)
assert EMB.shape[0] == len(IDS), "embedding/gallery count mismatch"

model, block_mapping, cfg = load_model(CKPT, device)
CROP = cfg["data"].get("crop_bbox", True)
print(f"Ready. device={device}  gallery={len(IDS)}  embedDim={EMB.shape[1]}  blocks={len(block_mapping)}")


def rank(qvec: np.ndarray, k: int):
    scores = EMB @ qvec.astype(np.float32)
    k = min(k, len(IDS))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [{"id": IDS[int(i)], "score": float(scores[int(i)])} for i in top]


app = FastAPI()


class TextReq(BaseModel):
    query: str
    k: int = 48


class VoxelReq(BaseModel):
    grid: str   # base64( gzip( uint16 LE [32^3] ) ) of REAL block-state ids
    k: int = 48


@app.get("/health")
def health():
    return {"ok": True, "gallery": len(IDS), "device": str(device), "model": "trimodal_tinyclip"}


@app.post("/search/text")
def search_text(req: TextReq):
    t0 = time.time()
    q = model.encode_text([req.query]).cpu().numpy()[0]
    return {"mode": "text", "method": "model:text->voxel",
            "results": rank(q, req.k), "tookMs": round((time.time() - t0) * 1000)}


@app.post("/search/voxel")
def search_voxel(req: VoxelReq):
    t0 = time.time()
    raw = gzip.decompress(base64.b64decode(req.grid))
    ids = np.frombuffer(raw, dtype="<u2")
    if ids.size != VOXELS:
        raise HTTPException(400, f"grid must be {VOXELS} uint16, got {ids.size}")
    vox = remap_voxel(ids, block_mapping, crop_bbox=CROP).unsqueeze(0).to(device)
    q = model.encode_voxel(vox).cpu().numpy()[0]
    return {"mode": "voxel", "method": "model:voxel->voxel",
            "results": rank(q, req.k), "tookMs": round((time.time() - t0) * 1000)}


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), k: int = Form(48)):
    t0 = time.time()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")
    px = model.preprocess(img).unsqueeze(0).to(device)
    q = model.encode_image(px).cpu().numpy()[0]
    return {"mode": "image", "method": "model:image->voxel",
            "results": rank(q, k), "tookMs": round((time.time() - t0) * 1000)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")
