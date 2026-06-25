"""TriModal encoder (text + image + voxel) — inference rebuild of the user's
trained model. Classes/submodule names are kept identical to the training code
(src/app/model/code_model.py) so the checkpoint's `model_state` loads cleanly.
"""

import os
import re
import json
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from transformers import CLIPModel, CLIPProcessor

try:
    from timm.layers import DropPath, trunc_normal_
except Exception:  # older timm
    from timm.models.layers import DropPath, trunc_normal_

# ── Inference config (subset of training CFG that affects the architecture) ──
CFG = {
    "tinyclip_hf_model": "openai/clip-vit-base-patch16",
    "freeze_clip": True,
    "text_max_length": 77,
    "image_size": 224,
    "embed_dim": 256,
    "max_block_types": 672,
    "crop_bbox": True,
    "text_mode": "cleaned_aug",
    "pb_pretrained_path": "",      # full trained weights are loaded afterwards
    "pb_freeze_backbone": True,
    "pb_trans_dim": 384,
    "pb_depth": 12,
    "pb_num_heads": 6,
    "pb_mlp_ratio": 4.0,
    "pb_drop_path": 0.1,
    "pb_num_points": 1024,
    "pb_block_embed_dim": 128,
    "pb_dropout": 0.1,
    # Random point sampling (as used during training). FPS was only the eval-time
    # config; the encoder mean-pools so it's order-invariant. Random is ~10x faster
    # for precomputing 8k embeddings. Set True for exact eval-parity (much slower).
    "pb_use_fps_eval": False,
}

GRID = 32

# ── Text + voxel preprocessing (verbatim from training) ─────────────────────

def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text(row, text_mode: str = "cleaned_aug", top_k_materials: int = 5) -> str:
    parts = []
    for field in ("title", "subtitle", "description"):
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(clean_text(val))
    tags = row.get("tags")
    if isinstance(tags, str):
        try:
            tag_list = json.loads(tags)
            if isinstance(tag_list, list):
                parts.append(", ".join(str(t) for t in tag_list))
        except Exception:
            parts.append(clean_text(tags))
    text = " ".join(parts)
    if text_mode == "cleaned_aug":
        vnd = row.get("voxel_name_data")
        if vnd is not None:
            try:
                names = list(vnd) if not isinstance(vnd, list) else vnd
                names = [n for n in names if isinstance(n, str) and "air" not in n]
                counter = Counter(names)
                top = [n.replace("minecraft:", "").replace("_", " ")
                       for n, _ in counter.most_common(top_k_materials)]
                if top:
                    text += " | materials: " + ", ".join(top)
            except Exception:
                pass
    return text.strip()


def build_name_vocab(df: pd.DataFrame, max_types: int = 672):
    counter: Counter = Counter()
    for vnd in df["voxel_name_data"]:
        try:
            names = list(vnd) if not isinstance(vnd, list) else vnd
            counter.update(n for n in names if isinstance(n, str) and n != "minecraft:air")
        except Exception:
            pass
    top_blocks = [b for b, _ in counter.most_common(max_types - 2)]
    name2id = {"minecraft:air": 0, "<rare>": 1}
    for i, name in enumerate(top_blocks, start=2):
        name2id[name] = i
    id2name = {v: k for k, v in name2id.items()}
    return name2id, id2name


def remap_voxel_names(voxel_name_flat, name2id: dict, crop_bbox: bool = True, target_size: int = 32) -> torch.LongTensor:
    names = list(voxel_name_flat)
    arr = np.array([name2id.get(n, 1) for n in names], dtype=np.int64)
    vol = arr.reshape(32, 32, 32)
    if crop_bbox:
        non_air = vol != 0
        if non_air.any():
            coords = np.argwhere(non_air)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            cropped = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            shape = cropped.shape
            if all(s > 0 for s in shape):
                factors = (target_size / shape[0], target_size / shape[1], target_size / shape[2])
                vol = zoom(cropped.astype(np.float32), factors, order=0).astype(np.int64)
                vol = np.clip(vol, 0, None)
    return torch.from_numpy(vol.reshape(32, 32, 32)).long()


def remap_ids(id_flat, crop_bbox: bool = True, target_size: int = 32) -> torch.LongTensor:
    """Same as remap_voxel_names but the input is already name2id indices (int)."""
    vol = np.asarray(id_flat, dtype=np.int64).reshape(32, 32, 32)
    if crop_bbox:
        non_air = vol != 0
        if non_air.any():
            coords = np.argwhere(non_air)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            cropped = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            shape = cropped.shape
            if all(s > 0 for s in shape):
                factors = (target_size / shape[0], target_size / shape[1], target_size / shape[2])
                vol = zoom(cropped.astype(np.float32), factors, order=0).astype(np.int64)
                vol = np.clip(vol, 0, None)
    return torch.from_numpy(vol.reshape(32, 32, 32)).long()


# ── PointBERT voxel encoder ─────────────────────────────────────────────────

def fps_sample(xyz: np.ndarray, n_samples: int) -> np.ndarray:
    N = len(xyz)
    if N == 0:
        return np.zeros(n_samples, dtype=np.int64)
    if N <= n_samples:
        idx = np.arange(N)
        pad = np.random.choice(N, n_samples - N, replace=True)
        return np.concatenate([idx, pad])
    selected = np.zeros(n_samples, dtype=np.int64)
    dist = np.full(N, np.inf)
    farthest = np.random.randint(N)
    for i in range(n_samples):
        selected[i] = farthest
        d = np.sum((xyz - xyz[farthest]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    return selected


class VoxelToPoints(nn.Module):
    def __init__(self, num_points: int = 512, use_fps_eval: bool = True):
        super().__init__()
        self.M = num_points
        self.use_fps_eval = use_fps_eval

    def _process_one(self, grid: torch.LongTensor, training: bool):
        device = grid.device
        M = self.M
        non_air = (grid != 0)
        coords = non_air.nonzero(as_tuple=False).float()
        bids = grid[non_air]
        N = coords.shape[0]
        if N == 0:
            return (torch.zeros(M, 3, device=device), torch.zeros(M, dtype=torch.long, device=device))
        if N <= M:
            pad_idx = torch.randint(N, (M - N,), device=device)
            coords = torch.cat([coords, coords[pad_idx]], dim=0)
            bids = torch.cat([bids, bids[pad_idx]], dim=0)
        elif training or not self.use_fps_eval:
            idx = torch.randperm(N, device=device)[:M]
            coords, bids = coords[idx], bids[idx]
        else:
            idx = fps_sample(coords.cpu().numpy(), M)
            idx = torch.from_numpy(idx).to(device)
            coords, bids = coords[idx], bids[idx]
        coords = coords / 31.0
        return coords, bids

    def forward(self, grids: torch.LongTensor):
        training = self.training
        B = grids.shape[0]
        all_xyz, all_bids = [], []
        for b in range(B):
            xyz, bids = self._process_one(grids[b], training)
            all_xyz.append(xyz)
            all_bids.append(bids)
        return torch.stack(all_xyz), torch.stack(all_bids)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PointBERTVoxelEncoder(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int):
        super().__init__()
        trans_dim = cfg["pb_trans_dim"]
        block_embed_dim = cfg["pb_block_embed_dim"]
        self.voxel_to_pts = VoxelToPoints(num_points=cfg["pb_num_points"], use_fps_eval=cfg["pb_use_fps_eval"])
        self.block_embedding = nn.Embedding(vocab_size, block_embed_dim)
        self.input_proj = nn.Linear(3 + block_embed_dim, trans_dim)
        dpr = [x.item() for x in torch.linspace(0, cfg["pb_drop_path"], cfg["pb_depth"])]
        self.blocks = nn.ModuleList([
            Block(dim=trans_dim, num_heads=cfg["pb_num_heads"], mlp_ratio=cfg["pb_mlp_ratio"],
                  qkv_bias=True, drop=cfg["pb_dropout"], attn_drop=0., drop_path=dpr[i])
            for i in range(cfg["pb_depth"])
        ])
        self.norm = nn.LayerNorm(trans_dim)
        self.output_head = nn.Sequential(nn.Linear(trans_dim, cfg["embed_dim"]), nn.Dropout(cfg["pb_dropout"]))

    def forward(self, voxels: torch.LongTensor) -> torch.Tensor:
        xyz, bids = self.voxel_to_pts(voxels)
        b_feat = self.block_embedding(bids)
        x = torch.cat([xyz, b_feat], dim=-1)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.output_head(x)


# ── CLIP text + image encoders ──────────────────────────────────────────────

class TinyClipTextEncoder(nn.Module):
    def __init__(self, cfg: dict, processor: CLIPProcessor):
        super().__init__()
        clip = CLIPModel.from_pretrained(cfg["tinyclip_hf_model"])
        self.text_model = clip.text_model
        self.text_proj = clip.text_projection
        self.processor = processor
        self.max_length = min(cfg["text_max_length"], 77)
        clip_out_dim = clip.config.projection_dim
        self.proj = nn.Linear(clip_out_dim, cfg["embed_dim"])

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        inputs = self.processor(text=texts, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt").to(device)
        out = self.text_model(**inputs)
        feats = out.pooler_output
        feats = self.text_proj(feats)
        feats = self.proj(feats)
        return F.normalize(feats, dim=-1)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode_text(texts)


class TinyClipImageEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        clip = CLIPModel.from_pretrained(cfg["tinyclip_hf_model"])
        self.vision_model = clip.vision_model
        self.visual_proj = clip.visual_projection
        clip_out_dim = clip.config.projection_dim
        self.proj = nn.Linear(clip_out_dim, cfg["embed_dim"])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values)
        feats = out.pooler_output
        feats = self.visual_proj(feats)
        feats = self.proj(feats)
        return F.normalize(feats, dim=-1)


class TriModalEncoder(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int, processor: CLIPProcessor):
        super().__init__()
        self.text_encoder = TinyClipTextEncoder(cfg, processor)
        self.image_encoder = TinyClipImageEncoder(cfg)
        self.voxel_encoder = PointBERTVoxelEncoder(cfg, vocab_size)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        return self.text_encoder(texts)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values)

    def encode_voxel(self, voxels: torch.LongTensor) -> torch.Tensor:
        return F.normalize(self.voxel_encoder(voxels), dim=-1)
