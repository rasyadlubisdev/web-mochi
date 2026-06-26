"""Inference rebuild of the retrained tri-modal model (ir_best_model.pt).

Architecture (from information-retrieval-4/mc-retrieval @ trimodal,
configs/trimodal_tinyclip.yaml):
  • voxel  : 3D-CNN VoxelEncoder over a 32^3 compact block-ID grid
  • text   : TinyCLIP (open_clip) text encoder  + shared clip_proj → 256-d
  • image  : TinyCLIP (open_clip) image encoder + shared clip_proj → 256-d

Classes/submodule names match the training code so model_state loads cleanly.
The TinyCLIP arch is built via the vendored open_clip with
pretrained='LAIONYFCC400M' (auto weight-inheritance + prune → 603-key structure
identical to the checkpoint; weights are then overwritten by the checkpoint).
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom

# vendored open_clip (exact copy used in training → guaranteed key match)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import open_clip  # noqa: E402

GRID = 32


# ── Voxel preprocessing (verbatim from src/dataset.py: remap_voxel) ──────────
def remap_voxel(voxel_flat, mapping: dict, crop_bbox: bool = True, target_size: int = 32) -> torch.LongTensor:
    arr = np.asarray(voxel_flat, dtype=np.int64)
    remapped = np.array([mapping.get(int(v), 1) for v in arr], dtype=np.int64)
    vol = remapped.reshape(32, 32, 32)
    if crop_bbox:
        non_air = vol != 0
        if non_air.any():
            coords = np.argwhere(non_air)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            cropped = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            shape = cropped.shape
            factors = (target_size / shape[0], target_size / shape[1], target_size / shape[2])
            vol = zoom(cropped, factors, order=0)
    return torch.from_numpy(vol.copy()).long()


# ── VoxelEncoder (verbatim from src/model.py) ────────────────────────────────
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class VoxelEncoder(nn.Module):
    def __init__(self, num_block_types=256, block_embed_dim=64, channels=[128, 256, 512],
                 embed_dim=256, dropout=0.3, use_learned_stem=False, use_depthwise_separable=False):
        super().__init__()
        self.block_embedding = nn.Embedding(num_block_types, block_embed_dim)
        layers = []
        in_ch = block_embed_dim
        if use_learned_stem:
            layers.extend([nn.Conv3d(in_ch, in_ch, kernel_size=4, stride=2, padding=1),
                           nn.BatchNorm3d(in_ch), nn.GELU()])
        for out_ch in channels[:-1]:
            conv = (DepthwiseSeparableConv3d(in_ch, out_ch, kernel_size=3, padding=1)
                    if use_depthwise_separable else nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.extend([conv, nn.BatchNorm3d(out_ch), nn.GELU(), nn.Dropout3d(dropout), nn.MaxPool3d(2)])
            in_ch = out_ch
        last = (DepthwiseSeparableConv3d(in_ch, channels[-1], kernel_size=3, padding=1)
                if use_depthwise_separable else nn.Conv3d(in_ch, channels[-1], kernel_size=3, padding=1))
        layers.extend([last, nn.BatchNorm3d(channels[-1]), nn.GELU(), nn.Dropout3d(dropout), nn.AdaptiveAvgPool3d(1)])
        self.conv_stack = nn.Sequential(*layers)
        self.project = nn.Sequential(nn.Linear(channels[-1], embed_dim), nn.Dropout(dropout))

    def forward(self, voxels: torch.LongTensor) -> torch.Tensor:
        x = self.block_embedding(voxels)               # (B,32,32,32,D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()       # (B,D,32,32,32)
        x = self.conv_stack(x)
        x = x.flatten(1)
        return self.project(x)


# ── Trimodal encoder (voxel + TinyCLIP) ──────────────────────────────────────
class TrimodalEncoder(nn.Module):
    def __init__(self, cfg: dict, num_block_types: int):
        super().__init__()
        mc = cfg["model"]
        self.embed_dim = mc["embed_dim"]
        self.voxel_encoder = VoxelEncoder(
            num_block_types=num_block_types,
            block_embed_dim=mc["block_embed_dim"],
            channels=mc["voxel_channels"],
            embed_dim=self.embed_dim,
            dropout=mc.get("dropout", 0.3),
            use_learned_stem=mc.get("use_learned_stem", False),
            use_depthwise_separable=mc.get("use_depthwise_separable", False),
        )
        arch = mc.get("tinyclip_arch", "TinyCLIP-auto-ViT-45M-32-Text-18M")
        pretrained = mc.get("tinyclip_pretrained", "LAIONYFCC400M")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(arch)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        if getattr(self.clip_model, "text_projection", None) is not None:
            clip_embed_dim = self.clip_model.text_projection.shape[1]
        elif hasattr(self.clip_model.visual, "output_dim"):
            clip_embed_dim = self.clip_model.visual.output_dim
        else:
            clip_embed_dim = 512
        self.clip_proj = nn.Linear(clip_embed_dim, self.embed_dim)

    @torch.no_grad()
    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(next(self.clip_model.parameters()).device)
        emb = self.clip_model.encode_text(tokens)
        return F.normalize(self.clip_proj(emb), dim=-1)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor):
        emb = self.clip_model.encode_image(images)
        return F.normalize(self.clip_proj(emb), dim=-1)

    @torch.no_grad()
    def encode_voxel(self, voxels: torch.LongTensor):
        return F.normalize(self.voxel_encoder(voxels), dim=-1)


def load_model(ckpt_path: str, device):
    """Build TrimodalEncoder from the checkpoint's cfg + block_mapping and load weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    block_mapping = {int(k): int(v) for k, v in ckpt["block_mapping"].items()}
    num_blocks = cfg["data"]["max_block_types"]
    model = TrimodalEncoder(cfg, num_block_types=num_blocks)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    bad = [k for k in missing if k.startswith(("voxel_encoder", "clip_proj"))]
    assert not bad, f"trained weights failed to load: {bad[:5]}"
    model.to(device).eval()
    return model, block_mapping, cfg
