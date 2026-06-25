// Voxel → Build retrieval ("more like this").
// Accepts a user-built 32^3 grid, runs the same bbox-crop + NN-resize as the
// training pipeline, extracts its structural feature vector, and ranks gallery
// builds by cosine similarity of those descriptors.

import { NextRequest, NextResponse } from "next/server";
import { getGallery } from "@/lib/gallery";
import { base64ToBytes, bboxCropResize, voxelFeatures, VOXELS } from "@/lib/voxel";
import { topK } from "@/lib/similarity";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const t0 = performance.now();
  let body: { grid?: string; k?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!body.grid) return NextResponse.json({ error: "Missing grid" }, { status: 400 });
  const k = Math.min(Math.max(body.k ?? 24, 1), 100);

  let raw: Uint8Array;
  try {
    raw = base64ToBytes(body.grid);
  } catch {
    return NextResponse.json({ error: "grid must be base64-encoded uint8" }, { status: 400 });
  }
  if (raw.length !== VOXELS) {
    return NextResponse.json(
      { error: `grid must be ${VOXELS} bytes (32^3), got ${raw.length}` },
      { status: 400 },
    );
  }

  let nonAir = 0;
  for (let i = 0; i < raw.length; i++) if (raw[i] !== 0) nonAir++;
  if (nonAir < 1) {
    return NextResponse.json({ error: "Build is empty — place some blocks first" }, { status: 400 });
  }

  const { grid, dims } = bboxCropResize(raw);
  const queryFeat = voxelFeatures(grid, dims);

  const index = await getGallery();
  // features are L2-normalised → topK uses dot == cosine and returns the top-k.
  const ranked = topK(queryFeat, index.builds.map((b) => b.features), k);
  const results = ranked.map((r) => ({ id: index.builds[r.index].raw.id, score: r.score }));

  return NextResponse.json({
    mode: "voxel",
    results,
    stats: { nonAir, dims },
    tookMs: Math.round(performance.now() - t0),
  });
}
