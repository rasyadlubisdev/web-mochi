// Cubane-powered grid renderer.
//
// This is the real schematic-renderer engine (`Cubane`) ported into the web app.
// Instead of approximating block geometry, it resolves every block through the
// vanilla resource pack (pack.zip) exactly like the desktop renderer: full
// blockstate → variant/multipart → model → element resolution, per-face textures,
// element/face rotations, biome tint, and the in-browser texture atlas. Small
// blocks (levers, torches, buttons, rails, fences, stairs…) render with their
// true shape and position — not as mis-placed cubes.
//
// The web data is a 32^3 grid of real Minecraft block-state ids. We map each id to
// its block string ("minecraft:name[props]", from /data/state2block.json) and ask
// Cubane for that block's mesh (cached per unique block), then place + merge all
// of them by material into a compact THREE.Group framed at the build's true size.

import * as THREE from "three";
import { Cubane } from "./cubane/Cubane";
import { MaterialRegistry } from "./cubane/MaterialRegistry";
import { mergeBufferGeometries } from "./cubane/mergeBufferGeometries";
import { GRID } from "./types";

const ridx = (x: number, y: number, z: number) => x * GRID * GRID + y * GRID + z;

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
  geometry: THREE.BufferGeometry; // block-local [-0.5, 0.5], material applied
  material: THREE.Material;
}

// Flatten a Cubane block Object3D into world-baked { geometry, material } pairs,
// mirroring WorldMeshBuilder.extractAllMeshData. Geometry is returned in the
// block's local frame (the rootGroup sits at origin; child transforms are baked).
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
    const geometry = child.geometry.clone();
    geometry.applyMatrix4(new THREE.Matrix4().copy(child.matrixWorld).multiply(rootInverse));
    if (geometry.attributes.position && geometry.attributes.position.count > 0) {
      out.push({ geometry, material });
    } else {
      geometry.dispose();
    }
  });
  return out;
}

/**
 * Mesh a 32^3 grid of block-state ids into a textured, material-grouped THREE.Group
 * using the Cubane engine. The group is centred on the build's bounding box so it
 * renders at true proportions — model only, no bounding box and no ground plane.
 */
export async function buildGridMesh(grid: Uint16Array): Promise<GridMesh> {
  const [cubane, s2b] = await Promise.all([getCubane(), getState2Block()]);

  // Mesh each unique block once; place every instance by translating the geometry.
  const blockCache = new Map<string, MeshData[] | null>();
  const byMaterial = new Map<string, { material: THREE.Material; geos: THREE.BufferGeometry[] }>();

  let minX = GRID, minY = GRID, minZ = GRID, maxX = -1, maxY = -1, maxZ = -1;

  for (let x = 0; x < GRID; x++) {
    for (let y = 0; y < GRID; y++) {
      for (let z = 0; z < GRID; z++) {
        const id = grid[ridx(x, y, z)];
        if (id === 0) continue;
        const blockString = s2b[id];
        if (!blockString) continue;

        if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
        if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;

        let data = blockCache.get(blockString);
        if (data === undefined) {
          try {
            const obj = await cubane.getBlockMesh(blockString, "plains", true);
            data = obj ? extractMeshData(obj) : null;
          } catch {
            data = null;
          }
          blockCache.set(blockString, data);
        }
        if (!data) continue;

        for (const { geometry, material } of data) {
          const shared = MaterialRegistry.getMaterial(material);
          const g = geometry.clone();
          // Cubane geometry is centred in [-0.5, 0.5]; offset so the block fills cell.
          g.translate(x + 0.5, y + 0.5, z + 0.5);
          let bucket = byMaterial.get(shared.uuid);
          if (!bucket) {
            bucket = { material: shared, geos: [] };
            byMaterial.set(shared.uuid, bucket);
          }
          bucket.geos.push(g);
        }
      }
    }
  }

  const group = new THREE.Group();
  for (const { material, geos } of byMaterial.values()) {
    const merged = mergeBufferGeometries(geos);
    geos.forEach((g) => g.dispose());
    if (!merged.attributes.position || merged.attributes.position.count === 0) continue;
    const mesh = new THREE.Mesh(merged, material);
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
