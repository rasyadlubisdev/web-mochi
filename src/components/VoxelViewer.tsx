"use client";

import type { ReactNode } from "react";
import { PrismarineViewer } from "./PrismarineViewer";

/**
 * Orbitable, **textured** 3D preview of a single build. Thin wrapper over
 * {@link PrismarineViewer} (the shared atlas + element mesher), so every preview —
 * gallery detail, uploads, and this viewer — share one rendering path and look
 * identical. Pass either a gallery `id` (its raw state-id grid is fetched) or a
 * raw 32^3 `Uint16Array` of Minecraft block-state ids.
 *
 * (For the flat-colour, compact-id editor preview see {@link Voxels}/VoxelBuilder.)
 */
export function VoxelViewer({
  id,
  grid,
  className,
  fallback,
}: {
  id?: string;
  grid?: Uint16Array;
  className?: string;
  fallback?: ReactNode;
}) {
  return <PrismarineViewer id={id} grid={grid} className={className} fallback={fallback} />;
}
