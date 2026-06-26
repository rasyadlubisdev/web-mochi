// Client-side textured voxel renderer (Cubane-quality model resolution).
//
// Loads the prebuilt block atlas (texture PNG + per-state ELEMENT models produced
// by scripts/build_voxel_assets.mjs, schema v2) and meshes a raw 32^3 grid of real
// Minecraft block-state IDs into textured geometry. Unlike a naive cube mesher,
// each block is rendered from its real model elements: stairs are L-shaped, slabs
// are half-height, crosses (flowers/grass) are X-shaped billboards, logs face the
// right way, furnaces show their front — every face mapped to its true atlas tile
// and in-tile sub-rect, with element/face rotations baked in, foliage tint, and
// full-cube face culling. A flat-colour fallback covers any unresolved states.

import * as THREE from "three";
import { GRID } from "./types";
import { blockColorRGB } from "./blocks";

const ridx = (x: number, y: number, z: number) => x * GRID * GRID + y * GRID + z;

// Foliage/grass biome tint (plains-ish) applied to faces flagged tintindex.
const TINT: [number, number, number] = [0.49, 0.74, 0.35];

type FaceName = "up" | "down" | "north" | "south" | "east" | "west";

interface AtlasFace {
  /** atlas tile rect in 0..1 UV space: [u, v, su, sv] (top-left + size) */
  box: [number, number, number, number];
  /** in-tile sub-rect in 0..16 Minecraft coords: [u0, v0, u1, v1] (default whole tile) */
  uv?: [number, number, number, number];
  /** face texture rotation in degrees: 0 | 90 | 180 | 270 */
  rot?: number;
  /** 1 → multiply by foliage/grass tint */
  tint?: number;
  /** face direction that, when its opaque neighbour is present, culls this face */
  cull?: FaceName;
}
interface AtlasElement {
  from: [number, number, number]; // 0..16
  to: [number, number, number]; // 0..16
  /** element rotation around an axis through `origin` */
  rot?: { origin: [number, number, number]; axis: "x" | "y" | "z"; angle: number; rescale?: boolean };
  faces: Partial<Record<FaceName, AtlasFace>>;
  shade?: number;
}
interface BlockEntry {
  elements: AtlasElement[];
  opaque?: boolean;
  /** single full 0..16 element → eligible for neighbour face culling */
  cube?: boolean;
  translucent?: boolean;
}
interface AtlasData {
  version: string;
  schema?: number;
  atlas: string;
  blocks: Record<number, BlockEntry>;
}

// Face geometry templates, in block-local 0..1 space. Each face lists its four
// corners (a,b,c,d, CCW seen from outside) keyed to the element's from/to box,
// plus the outward normal and the neighbour-direction used for culling. `0`=from,
// `1`=to on each axis. UV order matches Minecraft's [u0,v0,u1,v1] face mapping:
// (a)=u0,v0 top-left, (b)=u0,v1 bottom-left, (c)=u1,v1 bottom-right, (d)=u1,v0.
type Sel = 0 | 1;
interface FaceTemplate {
  name: FaceName;
  dir: [number, number, number];
  corners: [Sel, Sel, Sel][]; // a,b,c,d
}
const FACES: FaceTemplate[] = [
  // up: looking down -Y, u→+x, v→-z (north at top)
  { name: "up", dir: [0, 1, 0], corners: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]] },
  // down: looking up +Y, u→+x, v→+z
  { name: "down", dir: [0, -1, 0], corners: [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 0, 1]] },
  // north (-z): u→-x (east-left), v→-y
  { name: "north", dir: [0, 0, -1], corners: [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]] },
  // south (+z): u→+x, v→-y
  { name: "south", dir: [0, 0, 1], corners: [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]] },
  // west (-x): u→+z, v→-y
  { name: "west", dir: [-1, 0, 0], corners: [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]] },
  // east (+x): u→-z, v→-y
  { name: "east", dir: [1, 0, 0], corners: [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]] },
];

// Per-corner base UV in [u,v] tile-fraction order matching the corners above.
const CORNER_UV: [number, number][] = [
  [0, 0], // a top-left
  [0, 1], // b bottom-left
  [1, 1], // c bottom-right
  [1, 0], // d top-right
];

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

/** Decode a base64(gzip(uint16 LE)) raw grid (e.g. an uploaded schematic) → Uint16Array(32^3). */
export async function decodeRawVoxels(b64: string): Promise<Uint16Array> {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const ds = new DecompressionStream("gzip");
  const ab = await new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(ds)).arrayBuffer();
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

// Scratch buffer-of-arrays a mesh accumulates into.
interface Buf {
  p: number[];
  n: number[];
  u: number[];
  c: number[];
}
const newBuf = (): Buf => ({ p: [], n: [], u: [], c: [] });

// Rotate a block-local point (0..1) around an element-rotation axis.
function applyElementRotation(
  pt: [number, number, number],
  rot: NonNullable<AtlasElement["rot"]>,
): [number, number, number] {
  const ox = rot.origin[0] / 16, oy = rot.origin[1] / 16, oz = rot.origin[2] / 16;
  let x = pt[0] - ox, y = pt[1] - oy, z = pt[2] - oz;
  const a = (rot.angle * Math.PI) / 180;
  const cos = Math.cos(a), sin = Math.sin(a);
  if (rot.axis === "x") {
    [y, z] = [y * cos - z * sin, y * sin + z * cos];
  } else if (rot.axis === "y") {
    [x, z] = [x * cos + z * sin, -x * sin + z * cos];
  } else {
    [x, y] = [x * cos - y * sin, x * sin + y * cos];
  }
  // `rescale` stretches the two axes perpendicular to the rotation axis so the
  // rotated element still spans the block (used by crosses with angle 45).
  if (rot.rescale) {
    const s = 1 / Math.cos(a);
    if (rot.axis === "x") { y *= s; z *= s; }
    else if (rot.axis === "y") { x *= s; z *= s; }
    else { x *= s; y *= s; }
  }
  return [x + ox, y + oy, z + oz];
}

/**
 * Mesh a raw grid into textured + translucent + flat-colour geometries from each
 * block's real model elements. Geometry is centred on the build's bounding box so
 * it renders at true proportions.
 */
export function buildVoxelMesh(grid: Uint16Array, atlas: AtlasData): MeshResult {
  const blocks = atlas.blocks;
  const at = (x: number, y: number, z: number): number =>
    x < 0 || y < 0 || z < 0 || x >= GRID || y >= GRID || z >= GRID ? 0 : grid[ridx(x, y, z)];
  const isOpaque = (id: number): boolean =>
    id === 0 ? false : blocks[id] ? blocks[id].opaque !== false && blocks[id].cube !== false : true;

  const tex = newBuf();   // opaque / cutout (alphaTest)
  const trans = newBuf(); // translucent (glass, water…)
  const flat = newBuf();  // unresolved → hashed flat colour

  let minX = GRID, minY = GRID, minZ = GRID, maxX = -1, maxY = -1, maxZ = -1;

  for (let x = 0; x < GRID; x++) {
    for (let y = 0; y < GRID; y++) {
      for (let z = 0; z < GRID; z++) {
        const id = grid[ridx(x, y, z)];
        if (id === 0) continue;
        if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
        if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;

        const entry = blocks[id];
        if (!entry || !entry.elements?.length) {
          emitFlatCube(flat, x, y, z, blockColorRGB(((id % 250) + 2) | 0));
          continue;
        }

        const translucent = !!entry.translucent;
        const cube = !!entry.cube;
        const out = translucent ? trans : tex;

        for (const el of entry.elements) {
          // box corners in block-local 0..1, optionally element-rotated
          const lo: [number, number, number] = [el.from[0] / 16, el.from[1] / 16, el.from[2] / 16];
          const hi: [number, number, number] = [el.to[0] / 16, el.to[1] / 16, el.to[2] / 16];

          for (const ft of FACES) {
            const face = el.faces[ft.name];
            if (!face) continue;

            // full-cube face culling against an opaque (or same-translucent) neighbour
            if (cube && face.cull) {
              const [dx, dy, dz] = ft.dir;
              const nid = at(x + dx, y + dy, z + dz);
              if (isOpaque(nid) || (translucent && nid === id)) continue;
            }

            emitFace(out, x, y, z, lo, hi, el.rot, ft, face);
          }
        }
      }
    }
  }

  if (maxX < 0) {
    return {
      textured: null, translucent: null, flat: null,
      center: new THREE.Vector3(GRID / 2, GRID / 2, GRID / 2),
      size: new THREE.Vector3(1, 1, 1),
    };
  }

  const center = new THREE.Vector3((minX + maxX + 1) / 2, (minY + maxY + 1) / 2, (minZ + maxZ + 1) / 2);
  const size = new THREE.Vector3(maxX - minX + 1, maxY - minY + 1, maxZ - minZ + 1);

  const mk = (b: Buf, withUv: boolean): THREE.BufferGeometry | null => {
    if (b.p.length === 0) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(b.p, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(b.n, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(b.c, 3));
    if (withUv) g.setAttribute("uv", new THREE.Float32BufferAttribute(b.u, 2));
    g.translate(-center.x, -center.y, -center.z); // centre on bbox
    return g;
  };

  return { textured: mk(tex, true), translucent: mk(trans, true), flat: mk(flat, false), center, size };
}

// Emit one element face (two triangles) with textured UVs into a buffer.
function emitFace(
  out: Buf,
  bx: number, by: number, bz: number,
  lo: [number, number, number],
  hi: [number, number, number],
  elRot: AtlasElement["rot"],
  ft: FaceTemplate,
  face: AtlasFace,
) {
  const [u, v, su, sv] = face.box;
  // in-tile sub-rect in 0..1 of the tile (Minecraft uv is 0..16)
  const fu = face.uv ?? [0, 0, 16, 16];
  const u0 = fu[0] / 16, v0 = fu[1] / 16, u1 = fu[2] / 16, v1 = fu[3] / 16;

  // four corner positions (block-local 0..1), element-rotated if needed
  const pos: [number, number, number][] = ft.corners.map(([sx, sy, sz]) => {
    let p: [number, number, number] = [sx ? hi[0] : lo[0], sy ? hi[1] : lo[1], sz ? hi[2] : lo[2]];
    if (elRot) p = applyElementRotation(p, elRot);
    return p;
  });

  // per-corner UVs: pick u0/u1, v0/v1 by the corner's base UV, then rotate the
  // mapping by face.rot (90° steps) about the tile centre.
  const rot = ((face.rot ?? 0) % 360 + 360) % 360;
  const uvs: [number, number][] = CORNER_UV.map(([cu, cv]) => {
    let fU = cu ? u1 : u0;
    let fV = cv ? v1 : v0;
    return [fU, fV];
  });
  if (rot) rotateUv(uvs, u0, v0, u1, v1, rot);

  // final atlas UVs (flipY=false → v measured from top)
  const finalUv = uvs.map(([fU, fV]): [number, number] => [u + fU * su, v + fV * sv]);

  const [cr, cg, cb] = face.tint ? TINT : [1, 1, 1];
  const [nx, ny, nz] = ft.dir;

  // two triangles: a,b,c + a,c,d
  for (const ci of [0, 1, 2, 0, 2, 3]) {
    const [px, py, pz] = pos[ci];
    out.p.push(bx + px, by + py, bz + pz);
    out.n.push(nx, ny, nz);
    out.u.push(finalUv[ci][0], finalUv[ci][1]);
    out.c.push(cr, cg, cb);
  }
}

// Rotate the 4 corner UVs by 90° steps about the tile sub-rect centre.
function rotateUv(uvs: [number, number][], u0: number, v0: number, u1: number, v1: number, deg: number) {
  const cx = (u0 + u1) / 2, cy = (v0 + v1) / 2;
  const a = (deg * Math.PI) / 180;
  const cos = Math.cos(a), sin = Math.sin(a);
  for (const p of uvs) {
    const dx = p[0] - cx, dy = p[1] - cy;
    p[0] = cx + dx * cos - dy * sin;
    p[1] = cy + dx * sin + dy * cos;
  }
}

// Fallback: a flat-colour unit cube face set for unresolved states.
const CUBE_FACES: { dir: [number, number, number]; c: [number, number, number][] }[] = [
  { dir: [1, 0, 0], c: [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]] },
  { dir: [-1, 0, 0], c: [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]] },
  { dir: [0, 1, 0], c: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]] },
  { dir: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]] },
  { dir: [0, 0, 1], c: [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]] },
  { dir: [0, 0, -1], c: [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]] },
];
function emitFlatCube(out: Buf, x: number, y: number, z: number, color: [number, number, number]) {
  for (const f of CUBE_FACES) {
    for (const ci of [0, 1, 2, 0, 2, 3]) {
      const [ox, oy, oz] = f.c[ci];
      out.p.push(x + ox, y + oy, z + oz);
      out.n.push(f.dir[0], f.dir[1], f.dir[2]);
      out.c.push(color[0], color[1], color[2]);
    }
  }
}
