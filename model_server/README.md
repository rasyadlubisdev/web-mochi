# Model inference sidecar

All three searches (text / image / schematic) are powered by the retrained
tri-modal model in `src/app/model/ir_best_model.pt` (from
`information-retrieval-4/mc-retrieval` @ `trimodal`, `configs/trimodal_tinyclip.yaml`):

- **text + image** — a frozen **TinyCLIP** (`TinyCLIP-auto-ViT-45M-32-Text-18M`,
  `LAIONYFCC400M`) encoder via open_clip, with a shared learned `clip_proj` → 256-d
- **voxel** — a **3D-CNN** `VoxelEncoder` over a 32³ compact block-ID grid → 256-d

The Next.js API routes proxy here; the browser never calls it directly.

```
vendor/open_clip      exact open_clip copy from the repo (guarantees checkpoint key-match)
trimodal2.py          VoxelEncoder + TrimodalEncoder rebuild + load_model()
precompute.py         gallery voxel embeddings + gallery.json + raw preview bins
serve.py              FastAPI: /search/text, /search/image, /search/voxel, /health
```

The checkpoint already contains the `cfg` and `block_mapping` (numeric block-id →
compact index), so nothing about the vocab needs to be reconstructed. The TinyCLIP
backbone is built via `pretrained='LAIONYFCC400M'` (downloaded once, ~177 MB, cached)
to reproduce the pruned structure, then overwritten by the checkpoint weights.

## One-time setup

```bash
conda create -y -n mcmodel python=3.12
conda activate mcmodel
pip install torch transformers timm pandas pyarrow scipy pillow fastapi "uvicorn[standard]" python-multipart numpy ftfy regex
```

## Precompute gallery embeddings (run once, or after changing the gallery)

Reads `public/data/data.parquet` (numeric `voxel_data` + metadata) and writes
`public/data/gallery.json`, `public/data/raw/<id>.bin` (3D preview), and
`model_server/gallery_voxel_emb.npy`.

```bash
conda activate mcmodel
python model_server/precompute.py      # or: npm run model-precompute
```

## Run (two processes)

```bash
# terminal 1 — model sidecar (http://127.0.0.1:8008)
conda activate mcmodel && python model_server/serve.py     # or: npm run model-server

# terminal 2 — web app
npm run dev
```

Override the sidecar URL with `MODEL_SERVER_URL` (default `http://127.0.0.1:8008`).
