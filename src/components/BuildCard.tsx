"use client";

import type { GalleryItemRaw } from "@/lib/types";
import { compactNum, scoreFraction, scorePct } from "@/lib/format";
import { VoxelThumb } from "./VoxelThumb";

const CAT_COLORS: Record<string, string> = {
  "Land Structure Map": "#10b981",
  "3D Art Map": "#8b5cf6",
  "Redstone Device Map": "#ef4444",
  "Air Structure Map": "#38bdf8",
  "Complex Map": "#f59e0b",
  "Pixel Art Map": "#ec4899",
  "Water Structure Map": "#0ea5e9",
  "Piston Map": "#f97316",
};
function catColor(cat: string): string {
  return CAT_COLORS[cat] ?? "#94a3b8";
}

export function BuildCard({
  item,
  score,
  rank,
  onOpen,
}: {
  item: GalleryItemRaw;
  score?: number;
  rank?: number;
  onOpen: () => void;
}) {
  const c = catColor(item.category);

  return (
    <button
      onClick={onOpen}
      className="group fade-up text-left rounded-xl overflow-hidden border border-[var(--color-border)] bg-[var(--color-panel)] hover:border-[var(--color-voxel)]/60 hover:shadow-lg hover:shadow-emerald-500/5 transition-all duration-200"
    >
      <div className="relative aspect-[4/3] bg-[var(--color-panel-2)] overflow-hidden">
        <VoxelThumb
          id={item.id}
          alt={item.title}
          className="absolute inset-0 h-full w-full group-hover:scale-105 transition-transform duration-300"
        />

        {rank != null && (
          <span className="absolute top-2 left-2 mono text-[11px] px-1.5 py-0.5 rounded bg-black/60 backdrop-blur text-white/90">
            #{rank}
          </span>
        )}
        {score != null && (
          <span
            className="absolute top-2 right-2 mono text-[11px] px-1.5 py-0.5 rounded text-white font-medium"
            style={{ background: c }}
          >
            {scorePct(score)}
          </span>
        )}

        {score != null && (
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/40">
            <div className="h-full transition-all" style={{ width: `${scoreFraction(score) * 100}%`, background: c }} />
          </div>
        )}
      </div>

      <div className="p-3">
        <div className="flex items-center gap-1.5 mb-1">
          <span className="h-2 w-2 rounded-full shrink-0" style={{ background: c }} />
          <span className="text-[11px] text-white/50 truncate">{item.category}</span>
        </div>
        <h3 className="text-sm font-medium text-white/90 truncate" title={item.title}>
          {item.title}
        </h3>
        <div className="mt-2 flex items-center gap-3 text-[11px] text-white/45 mono">
          <span title="diamonds">💎 {compactNum(item.diamonds)}</span>
          <span title="downloads">⬇ {compactNum(item.downloads)}</span>
          <span title="views">👁 {compactNum(item.views)}</span>
        </div>
      </div>
    </button>
  );
}
