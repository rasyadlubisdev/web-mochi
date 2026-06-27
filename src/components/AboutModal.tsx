"use client";

import { X, Search, Image, Box } from "lucide-react";

export function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4 bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="fade-up w-full max-w-2xl max-h-[88vh] overflow-y-auto rounded-2xl border border-[var(--color-border)] bg-[var(--color-panel)] shadow-2xl p-7"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <h2 className="text-xl font-semibold text-white">How this demo works</h2>
          <button onClick={onClose} className="h-8 w-8 grid place-items-center rounded-lg border border-[var(--color-border)] text-white/60 hover:text-white">
            <X size={14} />
          </button>
        </div>

        <p className="mt-4 text-sm text-white/70 leading-relaxed">
          This is a demo of <strong className="text-white">cross-modal retrieval</strong> between text, images and
          3D Minecraft voxel schematics, the final project for the <em>Retrieval Information</em> course. It runs the
          actual <strong className="text-white">trained tri-modal model</strong>: a frozen
          <span className="text-[var(--color-text)]"> TinyCLIP</span> text + image encoder and a
          <span className="text-[var(--color-voxel)]"> 3D-CNN</span> voxel encoder projected into one shared 256-d space,
          trained with symmetric InfoNCE. Every build&apos;s voxel embedding is precomputed; your query is embedded live and ranked
          by cosine similarity.
        </p>

        <div className="mt-5 space-y-4 text-sm">
          <Section
            icon={<Search size={15} />}
            title="Text to Build"
            color="var(--color-text)"
            body="Your free-text query is embedded by the model's CLIP text encoder and ranked against every build's voxel embedding in the shared space (text to voxel, the model's main retrieval task)."
          />
          <Section
            icon={<Image size={15} />}
            title="Image to Build"
            color="var(--color-text)"
            body="Upload a photo or render; the model's TinyCLIP image encoder embeds it and ranks builds by similarity to their voxel embeddings (image to voxel)."
          />
          <Section
            icon={<Box size={15} />}
            title="Schematic to Build"
            color="var(--color-voxel)"
            body="Upload a .schem/.schematic; it's voxelised and embedded by the 3D-CNN voxel encoder, then matched against the gallery's voxel embeddings (voxel to voxel)."
          />
        </div>

        <div className="mt-5 text-xs text-white/40">
          Dataset: 8,328 Minecraft schematics (minecraft-schematics-mvm). Embeddings are produced by the user&apos;s trained
          checkpoint served from a local PyTorch inference sidecar.
        </div>
      </div>
    </div>
  );
}

function Section({ icon, title, body, color }: { icon: React.ReactNode; title: string; body: string; color: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-panel-2)] p-4">
      <div className="flex items-center gap-2 font-medium mb-1" style={{ color }}>
        {icon}
        {title}
      </div>
      <p className="text-white/65 leading-relaxed">{body}</p>
    </div>
  );
}
