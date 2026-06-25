"""Inference sidecar for the trained tri-modal model.

Loads schematics_best_model.pth once, plus the precomputed gallery voxel
embeddings, and ranks the gallery for text / image / voxel queries — all in the
SAME 256-d space the model was trained in. The Next.js API routes proxy here;
the browser never talks to this service directly.

Run inside the mcmodel conda env, from the web project root:
  python model_server/serve.py            # serves on http://127.0.0.1:8008
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
from trimodal import CFG, GRID, TriModalEncoder, remap_ids  # noqa: E402
from transformers import CLIPProcessor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT = os.path.join(ROOT, "src/app/model/schematics_best_model.pth")
OUT_DATA = os.path.join(ROOT, "public/data")
EMB_PATH = os.path.join(os.path.dirname(__file__), "gallery_voxel_emb.npy")
VOXELS = GRID * GRID * GRID

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

print("Loading gallery + model ...")
gallery = json.load(open(os.path.join(OUT_DATA, "gallery.json")))
IDS = [it["id"] for it in gallery["items"]]
EMB = np.load(EMB_PATH).astype(np.float32)          # (N, 256), L2-normalised
assert EMB.shape[0] == len(IDS), "embedding/gallery count mismatch"
name2id = json.load(open(os.path.join(OUT_DATA, "name2id.json")))
VOCAB = len(name2id)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
processor = CLIPProcessor.from_pretrained(CFG["tinyclip_hf_model"])
model = TriModalEncoder(CFG, VOCAB, processor)
model.load_state_dict(state, strict=False)
model.to(device).eval()
print(f"Ready. device={device}  gallery={len(IDS)}  vocab={VOCAB}")


def rank(qvec: np.ndarray, k: int):
    scores = EMB @ qvec.astype(np.float32)           # cosine (both L2-normed)
    k = min(k, len(IDS))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [{"id": IDS[int(i)], "score": float(scores[int(i)])} for i in top]


app = FastAPI()


class TextReq(BaseModel):
    query: str
    k: int = 48


class VoxelReq(BaseModel):
    grid: str   # base64( gzip( uint16 LE [32^3] ) ) of name2id indices
    k: int = 48


@app.get("/health")
def health():
    return {"ok": True, "gallery": len(IDS), "device": str(device)}


@app.post("/search/text")
def search_text(req: TextReq):
    t0 = time.time()
    with torch.no_grad():
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
    vox = remap_ids(ids, crop_bbox=CFG["crop_bbox"]).unsqueeze(0).to(device)
    with torch.no_grad():
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
    px = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        q = model.encode_image(px).cpu().numpy()[0]
    return {"mode": "image", "method": "model:image->voxel",
            "results": rank(q, k), "tookMs": round((time.time() - t0) * 1000)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")
