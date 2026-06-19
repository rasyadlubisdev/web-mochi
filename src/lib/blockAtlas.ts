// Client-side textured voxel renderer support.
//
// Loads the prebuilt block atlas (texture PNG + per-state face-UV map produced
// by scripts/build_voxel_assets.mjs) and meshes a raw 32^3 grid of real
// Minecraft block-state IDs into textured geometry, with greedy face culling,
// grass/foliage tinting, and a flat-colour fallback for unresolved states.

import * as THREE from "three";
import { GRID } from "./types";
import { blockColorRGB } from "./blocks";

const VOXELS = GRID * GRID * GRID;
const ridx = (x: number, y: number, z: number) => x * GRID * GRID + y * GRID + z;

// Foliage/grass biome tint (plains-ish) applied to faces flagged tintindex.
const TINT: [number, number, number] = [0.49, 0.74, 0.35];

type Rect = [number, number, number, number]; // [u, v, su, sv] in atlas UV space
interface BlockEntry {
  faces: Partial<Record<FaceName, Rect>>;
  tint?: Partial<Record<FaceName, boolean>>;
  opaque?: boolean;
  translucent?: boolean;
}
interface AtlasData {
  version: string;
  atlas: string;
  tile: number;
  blocks: Record<number, BlockEntry>;
}

type FaceName = "up" | "down" | "north" | "south" | "east" | "west";

const FACE_DEFS: { name: FaceName; dir: [number, number, number]; corners: [number, number, number][] }[] = [
  { name: "east", dir: [1, 0, 0], corners: [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]] },
  { name: "west", dir: [-1, 0, 0], corners: [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]] },
  { name: "up", dir: [0, 1, 0], corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]] },
  { name: "down", dir: [0, -1, 0], corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]] },
  { name: "south", dir: [0, 0, 1], corners: [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]] },
  { name: "north", dir: [0, 0, -1], corners: [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]] },
];
// local UV per corner (top-left, bottom-left, bottom-right, top-right) — upright
const LUV: [number, number][] = [[0, 0], [0, 1], [1, 1], [1, 0]];

// --- cached singletons ------------------------------------------------------

let atlasPromise: Promise<AtlasData> | null = null;
let texturePromise: Promise<THREE.Texture> | null = null;

export function loadAtlas(): Promise<AtlasData> {
  if (!atlasPromise) {
    atlasPromise = fetch("/data/block_atlas.json").then((r) => {
      if (!r.ok) throw new Error("block_atlas.json missing — run scripts/build_voxel_assets.mjs");
      return r.json();
    });
  }
  return atlasPromise;
}

export async function loadAtlasTexture(): Promise<THREE.Texture> {
  if (!texturePromise) {
    texturePromise = loadAtlas().then(
      (a) =>
        new Promise<THREE.Texture>((resolve, reject) => {
          new THREE.TextureLoader().load(
            a.atlas,
            (tex) => {
              tex.magFilter = THREE.NearestFilter;
              tex.minFilter = THREE.NearestFilter;
              tex.generateMipmaps = false;
              tex.flipY = false;
              tex.colorSpace = THREE.SRGBColorSpace;
              tex.needsUpdate = true;
              resolve(tex);
            },
            undefined,
            reject,
          );
        }),
    );
  }
  return texturePromise;
}

/** Fetch + gunzip a build's raw voxel grid → Uint16Array(32^3) of state IDs. */
export async function fetchVoxelGrid(id: string): Promise<Uint16Array> {
  const res = await fetch(`/data/raw/${id}.bin`);
  if (!res.ok) throw new Error(`raw voxels for ${id} not found`);
  const ds = new DecompressionStream("gzip");
  const ab = await new Response(res.body!.pipeThrough(ds)).arrayBuffer();
  return new Uint16Array(ab);
}

export interface MeshResult {
  textured: THREE.BufferGeometry | null;
  translucent: THREE.BufferGeometry | null;
  flat: THREE.BufferGeometry | null;
  /** centre of the build's non-air bounding box, in grid coords */
  center: THREE.Vector3;
  /** bounding-box extents [w, h, d] in blocks (true proportions) */
  size: THREE.Vector3;
}

/**
 * Mesh a raw grid into textured + flat-colour geometries. Geometry is centred
 * on the build's real bounding box so it renders at true proportions.
 */
export function buildVoxelMesh(grid: Uint16Array, atlas: AtlasData): MeshResult {
  const blocks = atlas.blocks;
  const at = (x: number, y: number, z: number): number =>
    x < 0 || y < 0 || z < 0 || x >= GRID || y >= GRID || z >= GRID ? 0 : grid[ridx(x, y, z)];
  const isOpaque = (id: number): boolean => (id === 0 ? false : blocks[id] ? blocks[id].opaque !== false : true);

  // textured (opaque/cutout) + translucent (glass…) + flat (unresolved) buffers
  const tp: number[] = [], tn: number[] = [], tu: number[] = [], tc: number[] = [];
  const gp: number[] = [], gn: number[] = [], gu: number[] = [], gc: number[] = [];
  const fp: number[] = [], fn: number[] = [], fc: number[] = [];

  let minX = GRID, minY = GRID, minZ = GRID, maxX = -1, maxY = -1, maxZ = -1;

  for (let x = 0; x < GRID; x++) {
    for (let y = 0; y < GRID; y++) {
      for (let z = 0; z < GRID; z++) {
        const id = grid[ridx(x, y, z)];
        if (id === 0) continue;
        if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
        if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;

        const entry = blocks[id];
        const translucent = !!entry?.translucent;
        for (const f of FACE_DEFS) {
          const [dx, dy, dz] = f.dir;
          const nid = at(x + dx, y + dy, z + dz);
          // cull faces hidden by an opaque neighbour, or shared between same translucent blocks
          if (isOpaque(nid) || (translucent && nid === id)) continue;

          const rect = entry?.faces[f.name];
          if (rect) {
            const [cr, cg, cb] = entry?.tint?.[f.name] ? TINT : [1, 1, 1];
            const [u, v, su, sv] = rect;
            const [p, n, uv, c] = translucent ? [gp, gn, gu, gc] : [tp, tn, tu, tc];
            for (const ci of [0, 1, 2, 0, 2, 3]) {
              const [ox, oy, oz] = f.corners[ci];
              p.push(x + ox, y + oy, z + oz);
              n.push(dx, dy, dz);
              const [lu, lv] = LUV[ci];
              uv.push(u + lu * su, v + lv * sv);
              c.push(cr, cg, cb);
            }
          } else {
            // unresolved block → flat hashed colour cube face
            const [cr, cg, cb] = blockColorRGB(((id % 250) + 2) | 0);
            for (const ci of [0, 1, 2, 0, 2, 3]) {
              const [ox, oy, oz] = f.corners[ci];
              fp.push(x + ox, y + oy, z + oz);
              fn.push(dx, dy, dz);
              fc.push(cr, cg, cb);
            }
          }
        }
      }
    }
  }

  if (maxX < 0) {
    return { textured: null, translucent: null, flat: null, center: new THREE.Vector3(GRID / 2, GRID / 2, GRID / 2), size: new THREE.Vector3(1, 1, 1) };
  }

  const center = new THREE.Vector3((minX + maxX + 1) / 2, (minY + maxY + 1) / 2, (minZ + maxZ + 1) / 2);
  const size = new THREE.Vector3(maxX - minX + 1, maxY - minY + 1, maxZ - minZ + 1);

  const mk = (pos: number[], nor: number[], col: number[], uv?: number[]): THREE.BufferGeometry | null => {
    if (pos.length === 0) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    if (uv) g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.translate(-center.x, -center.y, -center.z); // centre on bbox
    return g;
  };

  return { textured: mk(tp, tn, tc, tu), translucent: mk(gp, gn, gc, gu), flat: mk(fp, fn, fc), center, size };
}
