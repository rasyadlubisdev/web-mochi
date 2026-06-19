"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import type { GalleryItemRaw } from "@/lib/types";
import { decodeGrid } from "@/lib/voxel";
import { compactNum, scorePct } from "@/lib/format";

const VoxelViewer = dynamic(() => import("./VoxelViewer").then((m) => m.VoxelViewer), {
  ssr: false,
  loading: () => <div className="h-full w-full skeleton rounded-lg" />,
});

const PrismarineViewer = dynamic(() => import("./PrismarineViewer").then((m) => m.PrismarineViewer), {
  ssr: false,
  loading: () => <div className="h-full w-full skeleton rounded-lg" />,
});

export function BuildDetail({
  item,
  score,
  onClose,
}: {
  item: GalleryItemRaw;
  score?: number;
  onClose: () => void;
}) {
  const [grid, setGrid] = useState<Uint8Array | null>(null);

  useEffect(() => {
    let alive = true;
    decodeGrid(item.voxels).then((g) => alive && setGrid(g));
    return () => {
      alive = false;
    };
  }, [item.voxels]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center p-4 bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="fade-up w-full max-w-4xl max-h-[90vh] overflow-hidden rounded-2xl border border-[var(--color-border)] bg-[var(--color-panel)] shadow-2xl flex flex-col md:flex-row"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 3D preview */}
        <div className="relative md:w-1/2 aspect-square md:aspect-auto md:h-[520px] bg-[#0d0d16]">
          <PrismarineViewer
            id={item.id}
            className="h-full w-full"
            fallback={grid ? <VoxelViewer grid={grid} className="h-full w-full" /> : undefined}
          />
          <span className="absolute bottom-3 left-3 text-[11px] text-white/40 mono">
            drag to orbit · textured · {item.dims.join("×")} blocks
          </span>
        </div>

        {/* metadata */}
        <div className="md:w-1/2 p-6 overflow-y-auto">
          <div className="flex items-start justify-between gap-3">
            <div>
              <span className="text-xs text-[var(--color-voxel)]">{item.category}</span>
              <h2 className="text-xl font-semibold text-white mt-0.5">{item.title}</h2>
              {item.user && <p className="text-sm text-white/50 mt-1">by {item.user}{item.year ? ` · ${item.year}` : ""}</p>}
            </div>
            <button
              onClick={onClose}
              className="shrink-0 h-8 w-8 grid place-items-center rounded-lg border border-[var(--color-border)] text-white/60 hover:text-white hover:border-white/30"
            >
              ✕
            </button>
          </div>

          {score != null && (
            <div className="mt-4 rounded-lg border border-[var(--color-voxel)]/30 bg-[var(--color-voxel)]/5 px-3 py-2">
              <span className="text-xs text-white/60">cosine similarity to query</span>
              <div className="text-lg font-semibold mono text-[var(--color-voxel)]">{scorePct(score)}</div>
            </div>
          )}

          {item.description && (
            <p className="mt-4 text-sm text-white/70 leading-relaxed">{item.description}</p>
          )}

          {item.tags.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-1.5">
              {item.tags.map((t) => (
                <span key={t} className="text-[11px] px-2 py-0.5 rounded-full bg-[var(--color-panel-2)] border border-[var(--color-border)] text-white/60">
                  {t}
                </span>
              ))}
            </div>
          )}

          <div className="mt-5 grid grid-cols-3 gap-2 text-center">
            {[
              ["💎 Diamonds", item.diamonds],
              ["⬇ Downloads", item.downloads],
              ["👁 Views", item.views],
            ].map(([label, val]) => (
              <div key={label as string} className="rounded-lg bg-[var(--color-panel-2)] py-2">
                <div className="text-sm font-semibold mono text-white">{compactNum(val as number)}</div>
                <div className="text-[10px] text-white/45">{label}</div>
              </div>
            ))}
          </div>

          {item.url && (
            <a
              href={item.url}
              target="_blank"
              rel="noreferrer"
              className="mt-5 inline-flex items-center gap-2 text-sm px-4 py-2 rounded-lg bg-[var(--color-voxel)] text-black font-medium hover:bg-emerald-400 transition-colors"
            >
              View on Planet Minecraft ↗
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
