# Model inference sidecar

All three searches (text / image / schematic) are powered by the trained
tri-modal model in `src/app/model/schematics_best_model.pth` — a CLIP ViT-B/16
text+image encoder + a PointBERT voxel encoder projected into one shared 256-d
space. The Next.js API routes proxy to this Python service; the browser never
calls it directly.

```
trimodal.py    rebuild of TriModalEncoder (matches the checkpoint's model_state)
precompute.py  builds name2id + gallery voxel embeddings + gallery.json + raw preview bins
serve.py       FastAPI: /search/text, /search/image, /search/voxel, /health
```

## One-time setup

```bash
conda create -y -n mcmodel python=3.12
conda activate mcmodel
pip install torch transformers timm pandas pyarrow scipy pillow fastapi "uvicorn[standard]" python-multipart numpy
```

## Precompute gallery embeddings (run once, or after changing the dataset)

Reads the training parquet (`…/minecraft-schematics-dataset/data_with_voxel_names_multiview_image.parquet`
— edit `PARQUET` in `precompute.py` if it moves) and writes:

- `public/data/gallery.json`            metadata index (client + server)
- `public/data/raw/<id>.bin`            real block-state grids for the 3D preview
- `public/data/name2id.json`            block-name → vocab index (used by the schematic route)
- `model_server/gallery_voxel_emb.npy`  float32 [N,256] gallery voxel embeddings

```bash
conda activate mcmodel
python model_server/precompute.py     # or: npm run model-precompute
```

> Voxel point-sampling uses **random** sampling (as in training). FPS was only the
> eval-time config and is ~10× slower; flip `pb_use_fps_eval=True` in `trimodal.py`
> for exact eval-parity.

## Run (two processes)

```bash
# terminal 1 — model sidecar (http://127.0.0.1:8008)
conda activate mcmodel && python model_server/serve.py     # or: npm run model-server

# terminal 2 — web app
npm run dev
```

Override the sidecar URL with `MODEL_SERVER_URL` if needed (default `http://127.0.0.1:8008`).
