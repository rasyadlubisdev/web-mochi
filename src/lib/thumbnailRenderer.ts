// Card-thumbnail renderer.
//
// The browse grid can show 200+ cards, so we can't mount a WebGL context per
// card. Instead a single shared offscreen Three.js renderer paints each build
// (using the exact same atlas + mesher as the detail PrismarineViewer) to a PNG
// data URL — queued one at a time, cached by id. The result is a thumbnail that
// matches the detail view: real textures and true proportions (no squish).

import * as THREE from "three";
import { buildVoxelMesh, fetchVoxelGrid, loadAtlas, loadAtlasTexture } from "./blockAtlas";

const W = 480;
const H = 360; // 4:3, matches the card image area

interface RState {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  group: THREE.Group;
  matTex: THREE.MeshStandardMaterial;
  matTrans: THREE.MeshStandardMaterial;
  matFlat: THREE.MeshStandardMaterial;
}

let state: RState | null = null;
const cache = new Map<string, string>();
const inflight = new Map<string, Promise<string>>();
let chain: Promise<unknown> = Promise.resolve();

async function ensureState(): Promise<RState> {
  if (state) return state;
  const tex = await loadAtlasTexture();
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H, false);
  renderer.setClearColor(0x000000, 0); // transparent → blends with the card panel

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(40, 70, 30);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x8b5cf6, 0.35);
  fill.position.set(-30, 25, -35);
  scene.add(fill);

  const group = new THREE.Group();
  scene.add(group);
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 2000);

  state = {
    renderer,
    scene,
    camera,
    group,
    matTex: new THREE.MeshStandardMaterial({ map: tex, vertexColors: true, roughness: 0.95, metalness: 0, alphaTest: 0.5 }),
    matTrans: new THREE.MeshStandardMaterial({
      map: tex,
      vertexColors: true,
      transparent: true,
      alphaTest: 0.05,
      depthWrite: false,
      roughness: 0.25,
      metalness: 0,
      side: THREE.DoubleSide,
    }),
    matFlat: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0 }),
  };
  return state;
}

async function renderOne(id: string): Promise<string> {
  const s = await ensureState();
  const [grid, atlas] = await Promise.all([fetchVoxelGrid(id), loadAtlas()]);
  const mesh = buildVoxelMesh(grid, atlas);

  const added: THREE.Mesh[] = [];
  const add = (geo: THREE.BufferGeometry | null, mat: THREE.Material, order = 0) => {
    if (!geo) return;
    const m = new THREE.Mesh(geo, mat);
    m.renderOrder = order;
    s.group.add(m);
    added.push(m);
  };
  add(mesh.textured, s.matTex);
  add(mesh.flat, s.matFlat);
  add(mesh.translucent, s.matTrans, 1);

  // frame on the build's bounding sphere — true proportions, fit into the 4:3 view
  const radius = 0.5 * mesh.size.length() || 8;
  const fov = (s.camera.fov * Math.PI) / 180;
  const aspect = W / H;
  const fitV = radius / Math.tan(fov / 2);
  const fitH = radius / (Math.tan(fov / 2) * aspect);
  const dist = 1.45 * Math.max(fitV, fitH);
  const az = Math.PI / 4;
  const el = Math.PI / 6;
  s.camera.position.set(dist * Math.cos(el) * Math.cos(az), dist * Math.sin(el), dist * Math.cos(el) * Math.sin(az));
  s.camera.lookAt(0, 0, 0);

  s.renderer.render(s.scene, s.camera);
  const url = s.renderer.domElement.toDataURL("image/png");

  for (const m of added) {
    s.group.remove(m);
    m.geometry.dispose();
  }
  cache.set(id, url);
  return url;
}

/** Get a thumbnail PNG data URL for a build, rendered once and cached. */
export function getThumbnail(id: string): Promise<string> {
  const hit = cache.get(id);
  if (hit) return Promise.resolve(hit);
  const pending = inflight.get(id);
  if (pending) return pending;

  const run = chain.then(() => renderOne(id));
  chain = run.catch(() => {}); // keep the queue alive past failures
  const p = run.finally(() => inflight.delete(id));
  inflight.set(id, p);
  return p;
}
