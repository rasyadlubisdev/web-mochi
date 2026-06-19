// Voxel grid helpers — shared by client (3D builder/viewer) and server (search).
//
// Grids are flat Uint8 arrays of length 32^3 in C-order matching the Python
// export: flatIndex = ax * 1024 + ay * 32 + az, where we treat
//   ax = X, ay = Y (up), az = Z.

import { GRID } from "./types";

export const VOXELS = GRID * GRID * GRID; // 32768

export function idx(x: number, y: number, z: number): number {
  return x * GRID * GRID + y * GRID + z;
}

// --- base64 <-> bytes (isomorphic) -----------------------------------------

export function bytesToBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== "undefined") return Buffer.from(bytes).toString("base64");
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

export function base64ToBytes(b64: string): Uint8Array {
  if (typeof Buffer !== "undefined") return new Uint8Array(Buffer.from(b64, "base64"));
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// --- gzip (native DecompressionStream / CompressionStream) ------------------

async function streamThrough(bytes: Uint8Array, stream: ReadableWritablePair): Promise<Uint8Array> {
  const blob = new Blob([bytes as BlobPart]);
  const ab = await new Response(blob.stream().pipeThrough(stream)).arrayBuffer();
  return new Uint8Array(ab);
}

/** Decode base64( gzip( uint8[32^3] ) ) → Uint8Array(32768). */
export async function decodeGrid(b64: string): Promise<Uint8Array> {
  const compressed = base64ToBytes(b64);
  return streamThrough(compressed, new DecompressionStream("gzip"));
}

/** Encode a Uint8Array grid → base64( gzip(...) ). */
export async function encodeGrid(grid: Uint8Array): Promise<string> {
  const gz = await streamThrough(grid, new CompressionStream("gzip"));
  return bytesToBase64(gz);
}

// --- preprocessing: bbox crop + nearest-neighbour resize to 32^3 ------------

/**
 * Mirror the training pipeline: crop the non-air bounding box, then NN-resize
 * back to 32^3. Returns the canonical grid plus the original cropped extents.
 */
export function bboxCropResize(grid: Uint8Array): { grid: Uint8Array; dims: [number, number, number] } {
  let minX = GRID, minY = GRID, minZ = GRID, maxX = -1, maxY = -1, maxZ = -1;
  for (let x = 0; x < GRID; x++) {
    for (let y = 0; y < GRID; y++) {
      for (let z = 0; z < GRID; z++) {
        if (grid[idx(x, y, z)] !== 0) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (z < minZ) minZ = z;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
          if (z > maxZ) maxZ = z;
        }
      }
    }
  }
  if (maxX < 0) return { grid: new Uint8Array(VOXELS), dims: [0, 0, 0] };

  const cw = maxX - minX + 1;
  const ch = maxY - minY + 1;
  const cd = maxZ - minZ + 1;
  const out = new Uint8Array(VOXELS);
  for (let x = 0; x < GRID; x++) {
    const sx = minX + Math.min(cw - 1, Math.floor((x * cw) / GRID));
    for (let y = 0; y < GRID; y++) {
      const sy = minY + Math.min(ch - 1, Math.floor((y * ch) / GRID));
      for (let z = 0; z < GRID; z++) {
        const sz = minZ + Math.min(cd - 1, Math.floor((z * cd) / GRID));
        out[idx(x, y, z)] = grid[idx(sx, sy, sz)];
      }
    }
  }
  return { grid: out, dims: [cw, ch, cd] };
}

// --- iteration helper for rendering -----------------------------------------

export interface Voxel {
  x: number;
  y: number;
  z: number;
  b: number;
}

/** Collect non-air voxels for instanced rendering. */
export function nonAirVoxels(grid: Uint8Array): Voxel[] {
  const out: Voxel[] = [];
  for (let x = 0; x < GRID; x++)
    for (let y = 0; y < GRID; y++)
      for (let z = 0; z < GRID; z++) {
        const b = grid[idx(x, y, z)];
        if (b !== 0) out.push({ x, y, z, b });
      }
  return out;
}

// --- structural feature vector ----------------------------------------------
//
// A handcrafted descriptor used as a stand-in for the trained VoxelEncoder. It
// is computed identically for gallery builds and user queries, so cosine
// similarity over these vectors gives a meaningful "structurally similar"
// ranking (Build → Build retrieval). Groups are L2-normalised independently,
// scaled by a weight, concatenated, then the whole vector is L2-normalised.

const Y_BINS = 8;
const AX_BINS = 8;

const GROUP_WEIGHTS = {
  hist: 1.0, // material / block palette
  fill: 0.5, // overall density
  aspect: 1.1, // tall vs flat vs wide
  vprofile: 1.0, // vertical mass distribution
  xprofile: 0.7,
  zprofile: 0.7,
  symmetry: 0.6,
};

function l2norm(v: number[]): void {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
}

export function voxelFeatures(grid: Uint8Array, dims: [number, number, number]): Float32Array {
  const hist = new Array(256).fill(0);
  const vprofile = new Array(Y_BINS).fill(0);
  const xprofile = new Array(AX_BINS).fill(0);
  const zprofile = new Array(AX_BINS).fill(0);
  let nonAir = 0;

  for (let x = 0; x < GRID; x++)
    for (let y = 0; y < GRID; y++)
      for (let z = 0; z < GRID; z++) {
        const b = grid[idx(x, y, z)];
        if (b === 0) continue;
        nonAir++;
        hist[b]++;
        vprofile[Math.min(Y_BINS - 1, (y * Y_BINS) >> 5)]++;
        xprofile[Math.min(AX_BINS - 1, (x * AX_BINS) >> 5)]++;
        zprofile[Math.min(AX_BINS - 1, (z * AX_BINS) >> 5)]++;
      }

  // mirror symmetry along X and Z (fraction of voxels whose mirror is also set)
  let symX = 0, symZ = 0;
  if (nonAir > 0) {
    for (let x = 0; x < GRID; x++)
      for (let y = 0; y < GRID; y++)
        for (let z = 0; z < GRID; z++) {
          if (grid[idx(x, y, z)] !== 0) {
            if (grid[idx(GRID - 1 - x, y, z)] !== 0) symX++;
            if (grid[idx(x, y, GRID - 1 - z)] !== 0) symZ++;
          }
        }
    symX /= nonAir;
    symZ /= nonAir;
  }

  const fillRatio = nonAir / VOXELS;
  const dmag = Math.hypot(dims[0], dims[1], dims[2]) || 1;
  const aspect = [dims[0] / dmag, dims[1] / dmag, dims[2] / dmag];

  // normalise each group independently
  l2norm(hist);
  l2norm(vprofile);
  l2norm(xprofile);
  l2norm(zprofile);

  const feat: number[] = [];
  const push = (g: number[], w: number) => {
    for (const x of g) feat.push(x * w);
  };
  push(hist, GROUP_WEIGHTS.hist);
  push([fillRatio], GROUP_WEIGHTS.fill);
  push(aspect, GROUP_WEIGHTS.aspect);
  push(vprofile, GROUP_WEIGHTS.vprofile);
  push(xprofile, GROUP_WEIGHTS.xprofile);
  push(zprofile, GROUP_WEIGHTS.zprofile);
  push([symX, symZ], GROUP_WEIGHTS.symmetry);

  l2norm(feat);
  return new Float32Array(feat);
}
