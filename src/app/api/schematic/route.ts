// Schematic → Build retrieval via the trained tri-modal model.
//
// Parses an uploaded .schem/.schematic, voxelises it into a 32^3 grid, and:
//   • returns the real-block-state grid so the client renders a 3D preview, and
//   • maps blocks into the model's name2id vocab and asks the Python sidecar to
//     embed it with the PointBERT voxel encoder + rank the gallery (voxel→voxel).

import { NextRequest, NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { Schematic } from "prismarine-schematic";
import { Vec3 } from "vec3";
import { GRID } from "@/lib/types";
import { MODEL_SERVER, sidecarDownResponse } from "@/lib/modelServer";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const VERSION = "1.16.4";
const VOXELS = GRID * GRID * GRID;
const MAX_FILE = 25 * 1024 * 1024;
const MAX_VOLUME = 48_000_000;

const idx = (x: number, y: number, z: number) => x * GRID * GRID + y * GRID + z;

let nameToStateId: Record<string, number> | null = null;
let name2id: Record<string, number> | null = null;
function loadMaps() {
  if (!nameToStateId) nameToStateId = JSON.parse(fs.readFileSync(path.join(process.cwd(), "public", "data", "name2stateid.json"), "utf8"));
  if (!name2id) name2id = JSON.parse(fs.readFileSync(path.join(process.cwd(), "public", "data", "name2id.json"), "utf8"));
  return { nameToStateId: nameToStateId!, name2id: name2id! };
}

const isAir = (n?: string) => !n || n === "air" || n === "cave_air" || n === "void_air";

export async function POST(req: NextRequest) {
  const t0 = performance.now();

  let buffer: Buffer;
  let filename = "schematic";
  try {
    const form = await req.formData();
    const file = form.get("file");
    if (!file || typeof file === "string") return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    if (file.size > MAX_FILE) return NextResponse.json({ error: "File too large (max 25 MB)" }, { status: 400 });
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
      schem = await Schematic.read(buffer, VERSION);
    }
  } catch (err) {
    return NextResponse.json(
      { error: "Could not parse schematic", detail: err instanceof Error ? err.message : String(err),
        hint: "Supported: WorldEdit .schem (Sponge) and legacy MCEdit .schematic." },
      { status: 422 },
    );
  }

  const start = schem.start();
  const end = schem.end();
  const W = end.x - start.x + 1, H = end.y - start.y + 1, L = end.z - start.z + 1;
  if (W <= 0 || H <= 0 || L <= 0 || W * H * L > MAX_VOLUME) {
    return NextResponse.json({ error: "Schematic dimensions out of range" }, { status: 422 });
  }

  const { nameToStateId: n2s, name2id: n2i } = loadMaps();

  // uniform downscale → longest axis fills the 32-grid (preserves proportions), centred
  const maxD = Math.max(W, H, L);
  const s = GRID / maxD;
  const nw = Math.max(1, Math.min(GRID, Math.round(W * s)));
  const nh = Math.max(1, Math.min(GRID, Math.round(H * s)));
  const nl = Math.max(1, Math.min(GRID, Math.round(L * s)));
  const ox = (GRID - nw) >> 1, oy = (GRID - nh) >> 1, oz = (GRID - nl) >> 1;

  const raw = new Uint16Array(VOXELS);   // 1.16.4 state ids (preview)
  const vidx = new Uint16Array(VOXELS);  // model name2id indices (search)
  let nonAir = 0;
  for (let tx = 0; tx < nw; tx++) {
    const sx = Math.min(W - 1, Math.floor((tx * W) / nw));
    for (let ty = 0; ty < nh; ty++) {
      const sy = Math.min(H - 1, Math.floor((ty * H) / nh));
      for (let tz = 0; tz < nl; tz++) {
        const sz = Math.min(L - 1, Math.floor((tz * L) / nl));
        const name = schem.getBlock(new Vec3(start.x + sx, start.y + sy, start.z + sz))?.name;
        if (isAir(name)) continue;
        const key = `minecraft:${name}`;
        const gi = idx(ox + tx, oy + ty, oz + tz);
        raw[gi] = n2s[key] ?? n2s[name as string] ?? 1;
        vidx[gi] = n2i[key] ?? 1; // unknown non-air → <rare>
        nonAir++;
      }
    }
  }
  if (nonAir === 0) return NextResponse.json({ error: "Schematic appears to be empty (all air)" }, { status: 422 });

  const voxels = zlib.gzipSync(Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength), { level: 9 }).toString("base64");
  const gridB64 = zlib.gzipSync(Buffer.from(vidx.buffer, vidx.byteOffset, vidx.byteLength), { level: 9 }).toString("base64");

  let results: unknown;
  try {
    const res = await fetch(`${MODEL_SERVER}/search/voxel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid: gridB64, k: 48 }),
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    results = data.results;
  } catch (e) {
    return NextResponse.json(sidecarDownResponse(e instanceof Error ? e.message : String(e)), { status: 503 });
  }

  return NextResponse.json({
    mode: "voxel",
    method: "model:voxel->voxel",
    filename,
    voxels,
    results,
    stats: { dims: [W, H, L], blocks: nonAir, version: String(schem.version ?? VERSION) },
    tookMs: Math.round(performance.now() - t0),
  });
}
