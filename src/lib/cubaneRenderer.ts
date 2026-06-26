// Cubane-powered grid renderer with the library's occlusion culling.
//
// This is the schematic-renderer engine (`Cubane`) ported into the web app, with
// the same inter-block face culling its `WorldMeshBuilder` uses — so the output
// matches the desktop renderer and the block-render glitches of a naive mesher
// (z-fighting on coincident faces, doubled/darkened glass, drawn-but-hidden
// interior faces) are gone.
//
// Per block we ask Cubane for: (1) the renderable mesh (full model geometry, real
// per-face textures, element/face rotations, biome tint) and (2) its occlusion
// flags — which of the six faces are full + opaque. When assembling the grid, a
// block face flush with a cell boundary is dropped if the neighbour in that
// direction is in the SAME render category (solid/transparent/water) and presents
// a full opaque face there. That removes hidden interior faces and culls
// glass-on-glass / water-on-water without ever hiding a solid behind glass.

import * as THREE from "three";
import { Cubane } from "./cubane/Cubane";
import { MaterialRegistry } from "./cubane/MaterialRegistry";
import { GRID } from "./types";

const ridx = (x: number, y: number, z: number) => x * GRID * GRID + y * GRID + z;

// Blocks that hold water without exposing a `waterlogged` property (mirrors
// WorldMeshBuilder.IMPLICIT_WATERLOGGED) — they render in the water cull category.
const IMPLICIT_WATERLOGGED = new Set([
  "kelp", "kelp_plant", "seagrass", "tall_seagrass", "bubble_column",
]);

type Category = "solid" | "transparent" | "water";
const RENDER_ORDER: Record<Category, number> = { solid: 0, transparent: 2, water: 3 };

// --- shared singletons (built once per page) --------------------------------

let cubanePromise: Promise<Cubane> | null = null;
let state2blockPromise: Promise<Record<number, string>> | null = null;

/** The shared Cubane engine: loads the vanilla pack once and builds its atlas. */
export function getCubane(): Promise<Cubane> {
  if (!cubanePromise) {
    cubanePromise = (async () => {
      const cubane = new Cubane({ autoRestore: false, showUnknownBlocks: false });
      const res = await fetch("/pack.zip");
      if (!res.ok) throw new Error("pack.zip missing from /public — copy schematic-renderer/test/public/pack.zip");
      const blob = await res.blob();
      const assetLoader = cubane.getAssetLoader();
      await assetLoader.loadResourcePack(blob);
      await assetLoader.buildTextureAtlas();
      return cubane;
    })().catch((e) => {
      cubanePromise = null; // allow retry on transient failure
      throw e;
    });
  }
  return cubanePromise;
}

/** stateId → "minecraft:name[props]" map (built at asset-build time). */
function getState2Block(): Promise<Record<number, string>> {
  if (!state2blockPromise) {
    state2blockPromise = fetch("/data/state2block.json").then((r) => {
      if (!r.ok) throw new Error("state2block.json missing — run scripts/build_voxel_assets.mjs");
      return r.json();
    });
  }
  return state2blockPromise;
}

export interface GridMesh {
  /** Merged, material-grouped meshes, centred at the origin (model only). */
  group: THREE.Group;
  /** centre of the build's non-air bounding box, in grid coords */
  center: THREE.Vector3;
  /** bounding-box extents [w, h, d] in blocks (true proportions) */
  size: THREE.Vector3;
}

interface MeshData {
  geometry: THREE.BufferGeometry; // block-local [-0.5, 0.5], indexed
  material: THREE.Material;
}

interface PaletteEntry {
  meshData: MeshData[];
  /** 6-bit full-opaque-face mask: 0=west 1=east 2=down 3=up 4=north 5=south */
  occlusion: number;
  category: Category;
}

// Flatten a Cubane block Object3D into block-local { geometry, material } pairs,
// mirroring WorldMeshBuilder.extractAllMeshData.
function extractMeshData(root: THREE.Object3D): MeshData[] {
  const out: MeshData[] = [];
  root.updateMatrixWorld(true);
  const rootInverse = new THREE.Matrix4().copy(root.matrixWorld).invert();
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || !child.geometry || !child.material || !child.visible || child === root) {
      return;
    }
    const material = Array.isArray(child.material) ? child.material[0] : child.material;
    if (!material || !(material instanceof THREE.Material)) return;
    let geometry = child.geometry.clone();
    geometry.applyMatrix4(new THREE.Matrix4().copy(child.matrixWorld).multiply(rootInverse));
    if (!geometry.index) geometry = toIndexed(geometry);
    if (geometry.attributes.position && geometry.attributes.position.count > 0) {
      out.push({ geometry, material });
    } else {
      geometry.dispose();
    }
  });
  return out;
}

function toIndexed(g: THREE.BufferGeometry): THREE.BufferGeometry {
  const n = g.attributes.position.count;
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

// Remove the `waterlogged` property token from a block string ("…[…]" form).
function stripWaterlogged(bs: string): string {
  const m = bs.match(/^(minecraft:[a-z0-9_]+)(?:\[(.*)\])?$/);
  if (!m || !m[2]) return bs;
  const props = m[2].split(",").filter((kv) => !kv.startsWith("waterlogged="));
  return props.length ? `${m[1]}[${props.join(",")}]` : m[1];
}

function categoryOf(name: string, props: Record<string, string>): Category {
  if (name.includes("water") || name.includes("lava")) return "water";
  if (props.waterlogged === "true" || IMPLICIT_WATERLOGGED.has(name)) return "water";
  if (name.includes("glass") || name.includes("leaves") || name.includes("ice") || name === "barrier") {
    return "transparent";
  }
  return "solid";
}

// Replicate WorldMeshBuilder.computeOcclusionFlags: a face bit is set when the
// block presents a full (≈16×16) opaque face on that side. Water self-occludes.
function occlusionFlagsFrom(
  optData: { isCube?: boolean; cullableFaces?: Map<string, Array<{ material?: THREE.Material; elementBounds?: [number[], number[]] }>> },
  category: Category,
): number {
  if (category === "water") return 0b111111;
  if (!optData || !optData.cullableFaces) return 0;
  const bit: Record<string, number> = { west: 0, east: 1, down: 2, up: 3, north: 4, south: 5 };
  let flags = 0;
  for (const [dir, faces] of optData.cullableFaces.entries()) {
    const b = bit[dir];
    if (b === undefined || !Array.isArray(faces)) continue;
    let opaque = true, full = false;
    for (const face of faces) {
      const m = face.material as THREE.Material & { transparent?: boolean; opacity?: number };
      if (m && m.transparent && (m.opacity ?? 1) < 1) { opaque = false; break; }
      if (face.elementBounds) {
        const [min, max] = face.elementBounds;
        let w = 0, h = 0;
        if (dir === "up" || dir === "down") { w = max[0] - min[0]; h = max[2] - min[2]; }
        else if (dir === "north" || dir === "south") { w = max[0] - min[0]; h = max[1] - min[1]; }
        else { w = max[1] - min[1]; h = max[2] - min[2]; }
        if (w > 15.9 && h > 15.9) full = true;
      } else if (optData.isCube) {
        full = true;
      }
    }
    if (opaque && full) flags |= 1 << b;
  }
  return flags;
}

// Opposite-face bit of the neighbour in outward direction (dx,dy,dz).
function neighbourBit(dx: number, dy: number, dz: number): number {
  if (dx === 1) return 0;   // neighbour's west face faces me
  if (dx === -1) return 1;  // east
  if (dy === 1) return 2;   // down
  if (dy === -1) return 3;  // up
  if (dz === 1) return 4;   // north
  if (dz === -1) return 5;  // south
  return -1;
}

interface Bucket {
  material: THREE.Material;
  pos: number[];
  nrm: number[];
  uv: number[];
  renderOrder: number;
}

const FLUSH_EPS = 0.01;

/**
 * Mesh a 32^3 grid of block-state ids into a textured THREE.Group via Cubane,
 * with WorldMeshBuilder-style inter-block face culling. Centred on the build's
 * bounding box (true proportions). Model only — no bounding box, no ground plane.
 */
export async function buildGridMesh(grid: Uint16Array): Promise<GridMesh> {
  const [cubane, s2b] = await Promise.all([getCubane(), getState2Block()]);

  // Resolve every unique block once (geometry + occlusion + category).
  const palette = new Map<string, PaletteEntry | null>();
  const ensure = async (blockString: string): Promise<PaletteEntry | null> => {
    if (palette.has(blockString)) return palette.get(blockString)!;
    let entry: PaletteEntry | null = null;
    try {
      // Drop the `waterlogged` property: Cubane wraps waterlogged blocks in a full
      // water cube, and the dataset marks many dry blocks (roof stairs, panes…)
      // waterlogged, which would tint whole structures blue. Real `water` blocks
      // are unaffected (they aren't waterlogged variants).
      const clean = stripWaterlogged(blockString);
      const m = clean.match(/^minecraft:([a-z0-9_]+)(?:\[(.*)\])?$/);
      const name = m ? m[1] : clean.replace("minecraft:", "");
      const props: Record<string, string> = {};
      if (m && m[2]) for (const kv of m[2].split(",")) { const [k, v] = kv.split("="); props[k] = v; }
      const category = categoryOf(name, props);

      const obj = await cubane.getBlockMesh(clean, "plains", true);
      const meshData = obj ? extractMeshData(obj) : [];
      if (meshData.length) {
        const optData = await cubane.getBlockOptimizationData(clean, "plains", true);
        entry = { meshData, occlusion: occlusionFlagsFrom(optData, category), category };
      }
    } catch {
      entry = null;
    }
    palette.set(blockString, entry);
    return entry;
  };

  // First pass: gather presence + bbox, ensure all unique blocks are resolved.
  let minX = GRID, minY = GRID, minZ = GRID, maxX = -1, maxY = -1, maxZ = -1;
  const unique = new Set<string>();
  for (let i = 0; i < grid.length; i++) {
    const id = grid[i];
    if (id === 0) continue;
    const bs = s2b[id];
    if (bs) unique.add(bs);
  }
  await Promise.all([...unique].map((bs) => ensure(bs)));

  // Helper: occlusion + category of the block at a voxel (0 / null when empty).
  const entryAt = (x: number, y: number, z: number): PaletteEntry | null => {
    if (x < 0 || y < 0 || z < 0 || x >= GRID || y >= GRID || z >= GRID) return null;
    const id = grid[ridx(x, y, z)];
    if (id === 0) return null;
    const bs = s2b[id];
    return bs ? palette.get(bs) ?? null : null;
  };

  const buckets = new Map<string, Bucket>();

  // Second pass: place each block, culling flush faces against same-category
  // neighbours that present a full opaque face.
  for (let x = 0; x < GRID; x++) {
    for (let y = 0; y < GRID; y++) {
      for (let z = 0; z < GRID; z++) {
        const id = grid[ridx(x, y, z)];
        if (id === 0) continue;
        const bs = s2b[id];
        if (!bs) continue;
        const entry = palette.get(bs);
        if (!entry) continue;

        if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
        if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;

        const ox = x + 0.5, oy = y + 0.5, oz = z + 0.5;

        for (const { geometry, material } of entry.meshData) {
          const pos = geometry.attributes.position.array as ArrayLike<number>;
          const nrm = geometry.attributes.normal?.array as ArrayLike<number> | undefined;
          const uvA = geometry.attributes.uv?.array as ArrayLike<number> | undefined;
          const index = geometry.index!;
          const ia = index.array;

          const shared = MaterialRegistry.getMaterial(material);
          let bucket = buckets.get(shared.uuid);
          if (!bucket) {
            bucket = { material: shared, pos: [], nrm: [], uv: [], renderOrder: RENDER_ORDER[entry.category] };
            buckets.set(shared.uuid, bucket);
          } else {
            bucket.renderOrder = Math.max(bucket.renderOrder, RENDER_ORDER[entry.category]);
          }

          for (let t = 0; t < ia.length; t += 3) {
            const i0 = ia[t], i1 = ia[t + 1], i2 = ia[t + 2];

            // Cull this triangle if it is a flush, axis-aligned face occluded by a
            // same-category neighbour (matches the worker's per-triangle test).
            if (nrm) {
              const dx = Math.round(nrm[i0 * 3]);
              const dy = Math.round(nrm[i0 * 3 + 1]);
              const dz = Math.round(nrm[i0 * 3 + 2]);
              if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) === 1) {
                const vx = pos[i0 * 3], vy = pos[i0 * 3 + 1], vz = pos[i0 * 3 + 2];
                let flush = false;
                if (dx === 1) flush = Math.abs(vx - 0.5) < FLUSH_EPS;
                else if (dx === -1) flush = Math.abs(vx + 0.5) < FLUSH_EPS;
                else if (dy === 1) flush = Math.abs(vy - 0.5) < FLUSH_EPS;
                else if (dy === -1) flush = Math.abs(vy + 0.5) < FLUSH_EPS;
                else if (dz === 1) flush = Math.abs(vz - 0.5) < FLUSH_EPS;
                else if (dz === -1) flush = Math.abs(vz + 0.5) < FLUSH_EPS;
                if (flush) {
                  const nb = entryAt(x + dx, y + dy, z + dz);
                  if (nb && nb.category === entry.category) {
                    const b = neighbourBit(dx, dy, dz);
                    if (b >= 0 && (nb.occlusion & (1 << b)) !== 0) continue; // culled
                  }
                }
              }
            }

            for (const i of [i0, i1, i2]) {
              bucket.pos.push(pos[i * 3] + ox, pos[i * 3 + 1] + oy, pos[i * 3 + 2] + oz);
              if (nrm) bucket.nrm.push(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]);
              if (uvA) bucket.uv.push(uvA[i * 2], uvA[i * 2 + 1]);
            }
          }
        }
      }
    }
  }

  // The palette's per-block source geometries were cloned just to read their
  // arrays during assembly; the buckets hold the kept vertices now, so free them.
  for (const entry of palette.values()) {
    if (entry) for (const md of entry.meshData) md.geometry.dispose();
  }

  const group = new THREE.Group();
  for (const b of buckets.values()) {
    if (b.pos.length === 0) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(b.pos, 3));
    if (b.nrm.length) g.setAttribute("normal", new THREE.Float32BufferAttribute(b.nrm, 3));
    if (b.uv.length) g.setAttribute("uv", new THREE.Float32BufferAttribute(b.uv, 2));
    if (!b.nrm.length) g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, b.material);
    mesh.renderOrder = b.renderOrder;
    mesh.frustumCulled = false;
    group.add(mesh);
  }

  if (maxX < 0) {
    return {
      group,
      center: new THREE.Vector3(GRID / 2, GRID / 2, GRID / 2),
      size: new THREE.Vector3(1, 1, 1),
    };
  }

  const center = new THREE.Vector3((minX + maxX + 1) / 2, (minY + maxY + 1) / 2, (minZ + maxZ + 1) / 2);
  const size = new THREE.Vector3(maxX - minX + 1, maxY - minY + 1, maxZ - minZ + 1);
  group.position.set(-center.x, -center.y, -center.z); // centre the model at origin

  return { group, center, size };
}
