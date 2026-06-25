// Schematic → Build retrieval.
//
// Accepts an uploaded .schem / .schematic file, parses it (prismarine-schematic),
// voxelises it into a 32^3 grid of 1.16.4 block-state IDs (uniform-scaled to keep
// true proportions), then:
//   • returns the raw grid so the client can render a textured 3D preview, and
//   • remaps it into the gallery's compact block space, extracts the same
//     structural descriptor as the dataset, and ranks builds by cosine similarity.

import { NextRequest, NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { Schematic } from "prismarine-schematic";
import mcDataLoader from "minecraft-data";
import { Vec3 } from "vec3";
import { getGallery } from "@/lib/gallery";
import { bboxCropResize, voxelFeatures, idx, VOXELS } from "@/lib/voxel";
import { topK } from "@/lib/similarity";
import { GRID } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VERSION = "1.16.4";
const MAX_FILE = 25 * 1024 * 1024; // 25 MB
const MAX_VOLUME = 48_000_000; // reject absurdly large schematics (parsed fully in memory)

const mc = mcDataLoader(VERSION);

let mappingCache: Record<string, number> | null = null;
function getMapping(): Record<string, number> {
  if (mappingCache) return mappingCache;
  const p = path.join(process.cwd(), "public", "data", "block_mapping.json");
  const m = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, number>;
  mappingCache = m;
  return m;
}

function nameToStateId(name: string): number {
  const short = name.includes(":") ? name.split(":")[1] : name;
  const b = mc.blocksByName[short];
  return b && typeof b.defaultState === "number" ? b.defaultState : 1; // unknown → stone, so it stays visible
}

export async function POST(req: NextRequest) {
  const t0 = performance.now();

  let buffer: Buffer;
  let filename = "schematic";
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!file || typeof file === "string") {
      return NextResponse.json({ error: "No file uploaded (expected form field 'file')" }, { status: 400 });
    }
    if (file.size > MAX_FILE) {
      return NextResponse.json({ error: "File too large (max 25 MB)" }, { status: 400 });
    }
    filename = file.name || filename;
    buffer = Buffer.from(await file.arrayBuffer());
  } catch {
    return NextResponse.json({ error: "Invalid upload" }, { status: 400 });
  }

  let schem: {
    start: () => { x: number; y: number; z: number };
    end: () => { x: number; y: number; z: number };
    getBlock: (pos: unknown) => { name?: string } | null;
    version?: unknown;
  };
  try {
    try {
      schem = await Schematic.read(buffer);
    } catch {
      schem = await Schematic.read(buffer, VERSION); // legacy .schematic needs an explicit version
    }
  } catch (err) {
    return NextResponse.json(
      {
        error: "Could not parse schematic",
        detail: err instanceof Error ? err.message : String(err),
        hint: "Supported: WorldEdit .schem (Sponge) and legacy MCEdit .schematic.",
      },
      { status: 422 },
    );
  }

  const start = schem.start();
  const end = schem.end();
  const W = end.x - start.x + 1;
  const H = end.y - start.y + 1;
  const L = end.z - start.z + 1;
  if (W <= 0 || H <= 0 || L <= 0 || W * H * L > MAX_VOLUME) {
    return NextResponse.json({ error: "Schematic dimensions out of range" }, { status: 422 });
  }

  // uniform downscale so the longest axis fills the 32-grid (preserves proportions), centred
  const maxD = Math.max(W, H, L);
  const s = GRID / maxD;
  const nw = Math.max(1, Math.min(GRID, Math.round(W * s)));
  const nh = Math.max(1, Math.min(GRID, Math.round(H * s)));
  const nl = Math.max(1, Math.min(GRID, Math.round(L * s)));
  const ox = (GRID - nw) >> 1;
  const oy = (GRID - nh) >> 1;
  const oz = (GRID - nl) >> 1;

  const raw = new Uint16Array(VOXELS);
  let nonAir = 0;
  for (let tx = 0; tx < nw; tx++) {
    const sx = Math.min(W - 1, Math.floor((tx * W) / nw));
    for (let ty = 0; ty < nh; ty++) {
      const sy = Math.min(H - 1, Math.floor((ty * H) / nh));
      for (let tz = 0; tz < nl; tz++) {
        const sz = Math.min(L - 1, Math.floor((tz * L) / nl));
        const block = schem.getBlock(new Vec3(start.x + sx, start.y + sy, start.z + sz));
        const name = block?.name;
        if (!name || name === "air" || name === "cave_air" || name === "void_air") continue;
        raw[idx(ox + tx, oy + ty, oz + tz)] = nameToStateId(name);
        nonAir++;
      }
    }
  }

  if (nonAir === 0) {
    return NextResponse.json({ error: "Schematic appears to be empty (all air)" }, { status: 422 });
  }

  // preview payload: gzip(uint16 LE) → base64 (matches public/data/raw/*.bin encoding)
  const gz = zlib.gzipSync(Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength), { level: 9 });
  const voxels = gz.toString("base64");

  // search: remap to compact space → crop/resize → structural features → top-k
  const mapping = getMapping();
  const compact = new Uint8Array(VOXELS);
  for (let i = 0; i < VOXELS; i++) {
    const r = raw[i];
    compact[i] = r === 0 ? 0 : mapping[r] ?? 1;
  }
  const { grid, dims } = bboxCropResize(compact);
  const queryFeat = voxelFeatures(grid, dims);

  const index = await getGallery();
  const ranked = topK(queryFeat, index.builds.map((b) => b.features), 48);
  const results = ranked.map((r) => ({ id: index.builds[r.index].raw.id, score: r.score }));

  return NextResponse.json({
    mode: "voxel",
    filename,
    voxels,
    results,
    stats: { dims: [W, H, L], blocks: nonAir, version: String(schem.version ?? VERSION) },
    tookMs: Math.round(performance.now() - t0),
  });
}
