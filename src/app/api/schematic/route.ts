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
import mcDataLoader from "minecraft-data";
import prismarineBlock from "prismarine-block";
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
function loadNameToStateId(): Record<string, number> {
  if (!nameToStateId) {
    nameToStateId = JSON.parse(fs.readFileSync(path.join(process.cwd(), "public", "data", "name2stateid.json"), "utf8"));
  }
  return nameToStateId!;
}

// 1.16.4 block factory + data, used to resolve a schematic block's name AND
// properties (facing/half/axis/shape…) to its full state id — so the preview and
// the model see real, orientation-carrying blocks, not just each name's default.
type PBlock = { name: string; getProperties?: () => Record<string, unknown> };
let blockFactory: ReturnType<typeof prismarineBlock> | null = null;
let mcData: ReturnType<typeof mcDataLoader> | null = null;
function getBlockFactory() {
  if (!blockFactory) blockFactory = prismarineBlock(VERSION);
  return blockFactory;
}
function getMcData() {
  if (!mcData) mcData = mcDataLoader(VERSION);
  return mcData;
}

const isAir = (n?: string) => !n || n === "air" || n === "cave_air" || n === "void_air";

/** Resolve a parsed schematic block → full 1.16.4 state id (0 = air). */
function resolveStateId(block: PBlock | null): number {
  const name = block?.name;
  if (isAir(name)) return 0;
  const bd = getMcData().blocksByName[name as string];
  if (bd) {
    try {
      const props = typeof block!.getProperties === "function" ? block!.getProperties() : {};
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const sid = (getBlockFactory() as any).fromProperties(bd.id, props, 0)?.stateId;
      if (typeof sid === "number") return sid;
    } catch {
      /* fall through to defaults */
    }
    if (typeof bd.defaultState === "number") return bd.defaultState;
  }
  const n2s = loadNameToStateId();
  return n2s[`minecraft:${name}`] ?? n2s[name as string] ?? 1;
}

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

  // uniform downscale → longest axis fills the 32-grid (preserves proportions), centred
  const maxD = Math.max(W, H, L);
  const s = GRID / maxD;
  const nw = Math.max(1, Math.min(GRID, Math.round(W * s)));
  const nh = Math.max(1, Math.min(GRID, Math.round(H * s)));
  const nl = Math.max(1, Math.min(GRID, Math.round(L * s)));
  const ox = (GRID - nw) >> 1, oy = (GRID - nh) >> 1, oz = (GRID - nl) >> 1;

  // Build a 32^3 grid of REAL block-state ids; the sidecar remaps it into the
  // model's compact block space (block_mapping) before encoding.
  const raw = new Uint16Array(VOXELS);
  let nonAir = 0;
  for (let tx = 0; tx < nw; tx++) {
    const sx = Math.min(W - 1, Math.floor((tx * W) / nw));
    for (let ty = 0; ty < nh; ty++) {
      const sy = Math.min(H - 1, Math.floor((ty * H) / nh));
      for (let tz = 0; tz < nl; tz++) {
        const sz = Math.min(L - 1, Math.floor((tz * L) / nl));
        const block = schem.getBlock(new Vec3(start.x + sx, start.y + sy, start.z + sz)) as PBlock | null;
        const sid = resolveStateId(block);
        if (sid === 0) continue;
        raw[idx(ox + tx, oy + ty, oz + tz)] = sid;
        nonAir++;
      }
    }
  }
  if (nonAir === 0) return NextResponse.json({ error: "Schematic appears to be empty (all air)" }, { status: 422 });

  const voxels = zlib.gzipSync(Buffer.from(raw.buffer, raw.byteOffset, raw.byteLength), { level: 9 }).toString("base64");

  let results: unknown;
  try {
    const res = await fetch(`${MODEL_SERVER}/search/voxel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid: voxels, k: 48 }),
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
