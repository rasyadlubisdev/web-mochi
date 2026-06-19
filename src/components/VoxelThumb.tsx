"use client";

import { useEffect, useRef, useState } from "react";
import { decodeGrid, nonAirVoxels } from "@/lib/voxel";
import { blockColorRGB } from "@/lib/blocks";

// Multiply an [r,g,b] (0..1) by a brightness factor → CSS rgb() string.
function shade([r, g, b]: [number, number, number], f: number): string {
  const c = (v: number) => Math.round(Math.min(1, v * f) * 255);
  return `rgb(${c(r)},${c(g)},${c(b)})`;
}

/**
 * Lightweight isometric preview of a voxel build, painted once on a 2D canvas
 * straight from the local voxel grid. This replaces the old Planet Minecraft
 * thumbnail <img> — no external fetch, and (unlike a WebGL viewer) cheap enough
 * to mount across the whole browse grid. Renders lazily once scrolled into view.
 */
export function VoxelThumb({ voxels, className }: { voxels: string; className?: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = wrap.current;
    if (!el || visible) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    let alive = true;
    decodeGrid(voxels).then((grid) => {
      if (alive) paint(canvas.current, wrap.current, grid);
    });
    return () => {
      alive = false;
    };
  }, [visible, voxels]);

  return (
    <div ref={wrap} className={className}>
      <canvas ref={canvas} className="block h-full w-full" />
    </div>
  );
}

/** Corner-on (+x, +y, +z) isometric painter's render of all non-air voxels. */
function paint(cv: HTMLCanvasElement | null, wrap: HTMLDivElement | null, grid: Uint8Array): void {
  if (!cv || !wrap) return;
  const W = wrap.clientWidth || 240;
  const H = wrap.clientHeight || 180;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  const vox = nonAirVoxels(grid);
  if (vox.length === 0) return;

  // Unit-scale projected bounds (account for each cube's full 1×1×1 footprint).
  let minPx = Infinity, maxPx = -Infinity, minPy = Infinity, maxPy = -Infinity;
  for (const v of vox) {
    const px = v.x - v.z;
    const py = (v.x + v.z) / 2 - v.y;
    if (px - 1 < minPx) minPx = px - 1;
    if (px + 1 > maxPx) maxPx = px + 1;
    if (py - 1 < minPy) minPy = py - 1;
    if (py + 1 > maxPy) maxPy = py + 1;
  }
  const unitW = maxPx - minPx || 1;
  const unitH = maxPy - minPy || 1;
  const pad = 0.86;
  const s = Math.min((W * pad) / unitW, (H * pad) / unitH);
  const ox = -minPx * s + (W - unitW * s) / 2;
  const oy = -minPy * s + (H - unitH * s) / 2;

  // Back-to-front along the (+x, +y, +z) view direction.
  vox.sort((a, b) => a.x + a.y + a.z - (b.x + b.y + b.z));

  const proj = (cx: number, cy: number, cz: number): [number, number] => [
    ox + (cx - cz) * s,
    oy + ((cx + cz) / 2 - cy) * s,
  ];
  const quad = (
    p1: [number, number],
    p2: [number, number],
    p3: [number, number],
    p4: [number, number],
    fill: string,
  ) => {
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.lineTo(p3[0], p3[1]);
    ctx.lineTo(p4[0], p4[1]);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  };

  for (const { x, y, z, b } of vox) {
    const rgb = blockColorRGB(b);
    // top (+y)
    quad(proj(x, y + 1, z), proj(x + 1, y + 1, z), proj(x + 1, y + 1, z + 1), proj(x, y + 1, z + 1), shade(rgb, 1.0));
    // right (+x)
    quad(proj(x + 1, y, z), proj(x + 1, y + 1, z), proj(x + 1, y + 1, z + 1), proj(x + 1, y, z + 1), shade(rgb, 0.78));
    // front (+z)
    quad(proj(x, y, z + 1), proj(x + 1, y, z + 1), proj(x + 1, y + 1, z + 1), proj(x, y + 1, z + 1), shade(rgb, 0.6));
  }
}
