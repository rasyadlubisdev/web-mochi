// Build-time asset generator for the WHOLE dataset (all rows of data.parquet).
//
// Streams every row out of the parquet with DuckDB (the memory-bounded
// "iterrows" — a single native scan, never materialising the 8k×32^3 column in
// JS) and, per build, emits:
//   public/data/raw/<id>.bin     gzip(uint16 LE [32^3])  — real block-state grid (rendering)
//   public/data/block_atlas.json                          — stateId -> per-face atlas UVs
//   public/atlas/<version>.png                            — block texture atlas
//   public/data/gallery.json                              — lightweight metadata index (client + server)
//   public/data/features.bin   Float32 [count][featDim]   — structural voxel descriptors (voxel search)
//   public/data/_texts.json                               — per-build text (input to build_text_embeddings.mjs)
//
// Block remap / crop / resize / features mirror scripts/export_gallery.py and
// src/lib/voxel.ts exactly, so voxel-search stays consistent with the builder.
//
// Run with Node >= 18:  node scripts/build_voxel_assets.mjs [--limit N]

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const require = createRequire(import.meta.url);
const duckdb = require("duckdb");

const VERSION = "1.16.4";
const GRID = 32;
const VOXELS = GRID * GRID * GRID;
const MAX_BLOCK_TYPES = 256;
const MIN_NONAIR = 20;

const argv = process.argv.slice(2);
const LIMIT = (() => {
  const i = argv.indexOf("--limit");
  return i >= 0 ? parseInt(argv[i + 1], 10) : 0;
})();

const ROOT = process.cwd();
const PARQUET = path.join(ROOT, "public/data/data.parquet");
const PV_PUBLIC = path.join(ROOT, "node_modules/prismarine-viewer/public");
const OUT_RAW = path.join(ROOT, "public/data/raw");
const OUT_ATLAS_DIR = path.join(ROOT, "public/atlas");
const OUT_ATLAS_JSON = path.join(ROOT, "public/data/block_atlas.json");
const OUT_GALLERY = path.join(ROOT, "public/data/gallery.json");
const OUT_FEATURES = path.join(ROOT, "public/data/features.bin");
const OUT_TEXTS = path.join(ROOT, "public/data/_texts.json");
const OUT_MAPPING = path.join(ROOT, "public/data/block_mapping.json");
const OUT_STATE2BLOCK = path.join(ROOT, "public/data/state2block.json");

const FACES = ["up", "down", "north", "south", "east", "west"];
const TRANSLUCENT = /glass|^water$|^ice$|frosted_ice|slime_block|honey_block|nether_portal|barrier|^lava$/;

const mcData = require("minecraft-data")(VERSION);
const Block = require("prismarine-block")(VERSION);
const blocksStates = require(path.join(PV_PUBLIC, "blocksStates", `${VERSION}.json`));

// ── metadata helpers (mirror export_gallery.py) ───────────────────────────────
const clean = (s) => (typeof s === "string" ? s.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() : "");
function parseTags(raw) {
  if (Array.isArray(raw)) return raw.map((t) => String(t).trim()).filter(Boolean);
  if (typeof raw !== "string") return [];
  try {
    const v = JSON.parse(raw);
    if (Array.isArray(v)) return v.map((t) => String(t).trim()).filter(Boolean);
  } catch {
    /* not JSON */
  }
  return clean(raw).split(",").map((t) => t.trim()).filter(Boolean);
}
function buildText(row, tags) {
  const parts = [];
  for (const f of [row.title, row.subtitle, row.description]) if (clean(f)) parts.push(clean(f));
  if (tags.length) parts.push(tags.join(", "));
  return parts.join(" ");
}
function yearOf(date) {
  return typeof date === "string" && /^\d{4}/.test(date) ? parseInt(date.slice(0, 4), 10) : null;
}

// ── voxel preprocessing (mirror src/lib/voxel.ts) ─────────────────────────────
const idx = (x, y, z) => x * GRID * GRID + y * GRID + z;

function bboxCropResize(g) {
  let minX = GRID, minY = GRID, minZ = GRID, maxX = -1, maxY = -1, maxZ = -1;
  for (let x = 0; x < GRID; x++)
    for (let y = 0; y < GRID; y++)
      for (let z = 0; z < GRID; z++)
        if (g[idx(x, y, z)] !== 0) {
          if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
          if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
        }
  if (maxX < 0) return { grid: new Uint8Array(VOXELS), dims: [0, 0, 0] };
  const cw = maxX - minX + 1, ch = maxY - minY + 1, cd = maxZ - minZ + 1;
  const out = new Uint8Array(VOXELS);
  for (let x = 0; x < GRID; x++) {
    const sx = minX + Math.min(cw - 1, Math.floor((x * cw) / GRID));
    for (let y = 0; y < GRID; y++) {
      const sy = minY + Math.min(ch - 1, Math.floor((y * ch) / GRID));
      for (let z = 0; z < GRID; z++) {
        const sz = minZ + Math.min(cd - 1, Math.floor((z * cd) / GRID));
        out[idx(x, y, z)] = g[idx(sx, sy, sz)];
      }
    }
  }
  return { grid: out, dims: [cw, ch, cd] };
}

const Y_BINS = 8, AX_BINS = 8;
const GW = { hist: 1.0, fill: 0.5, aspect: 1.1, vprofile: 1.0, xprofile: 0.7, zprofile: 0.7, symmetry: 0.6 };
function l2(v) {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
}
function voxelFeatures(grid, dims) {
  const hist = new Array(256).fill(0), vprofile = new Array(Y_BINS).fill(0);
  const xprofile = new Array(AX_BINS).fill(0), zprofile = new Array(AX_BINS).fill(0);
  let nonAir = 0;
  for (let x = 0; x < GRID; x++)
    for (let y = 0; y < GRID; y++)
      for (let z = 0; z < GRID; z++) {
        const b = grid[idx(x, y, z)];
        if (b === 0) continue;
        nonAir++; hist[b]++;
        vprofile[Math.min(Y_BINS - 1, (y * Y_BINS) >> 5)]++;
        xprofile[Math.min(AX_BINS - 1, (x * AX_BINS) >> 5)]++;
        zprofile[Math.min(AX_BINS - 1, (z * AX_BINS) >> 5)]++;
      }
  let symX = 0, symZ = 0;
  if (nonAir > 0) {
    for (let x = 0; x < GRID; x++)
      for (let y = 0; y < GRID; y++)
        for (let z = 0; z < GRID; z++)
          if (grid[idx(x, y, z)] !== 0) {
            if (grid[idx(GRID - 1 - x, y, z)] !== 0) symX++;
            if (grid[idx(x, y, GRID - 1 - z)] !== 0) symZ++;
          }
    symX /= nonAir; symZ /= nonAir;
  }
  const fillRatio = nonAir / VOXELS;
  const dmag = Math.hypot(dims[0], dims[1], dims[2]) || 1;
  const aspect = [dims[0] / dmag, dims[1] / dmag, dims[2] / dmag];
  l2(hist); l2(vprofile); l2(xprofile); l2(zprofile);
  const feat = [];
  const push = (g, w) => { for (const x of g) feat.push(x * w); };
  push(hist, GW.hist); push([fillRatio], GW.fill); push(aspect, GW.aspect);
  push(vprofile, GW.vprofile); push(xprofile, GW.xprofile); push(zprofile, GW.zprofile);
  push([symX, symZ], GW.symmetry);
  l2(feat);
  return Float32Array.from(feat);
}
const FEAT_DIM = 256 + 1 + 3 + Y_BINS + AX_BINS + AX_BINS + 2;

// ── model resolution (prismarine-viewer prebuilt, FULLY-RESOLVED blockstates) ──
//
// prismarine-viewer ships `blocksStates/<version>.json` with each blockstate
// already resolved down to render-ready geometry: per variant/multipart, a model
// with ALL `elements` (from/to boxes, element rotations) and, per element face,
// the in-tile `uv` sub-rect (0–16), the atlas tile rect (`texture.{u,v,su,sv}` in
// 0–1 UV space), `tintindex`, `cullface` and face `rotation`. We carry that whole
// structure to the client so the runtime mesher draws REAL geometry (stairs are
// L-shaped, slabs are half-height, crosses are X-shaped, logs face the right way)
// with correct per-face textures — instead of a single-texture unit cube.

// Pick the model(s) that apply to a given set of block properties.
//   • variants → the single best-matching variant (exact props, else "", else first)
//   • multipart → every part whose `when` condition matches (fences, walls, panes…)
const takeModel = (v) => {
  const x = Array.isArray(v) ? v[0] : v; // weighted lists → first entry
  return x && x.model ? x.model : x;
};
function matchesWhen(when, props) {
  if (!when) return true;
  if (when.OR) return when.OR.some((c) => matchesWhen(c, props));
  return Object.entries(when).every(([k, val]) => {
    const have = String(props[k]);
    return String(val).split("|").includes(have);
  });
}
function resolveModels(entry, props) {
  if (!entry) return [];
  if (entry.variants) {
    const variants = entry.variants;
    const keys = Object.keys(variants);
    if (keys.length === 1) return [takeModel(variants[keys[0]])];
    // best variant: most property tokens matching `props`, none conflicting
    let best = null, bestScore = -1, empty = null;
    for (const key of keys) {
      if (key === "") { empty = variants[key]; continue; }
      const toks = key.split(",").map((t) => t.split("="));
      let score = 0, ok = true;
      for (const [k, val] of toks) {
        if (String(props[k]) === val) score++;
        else { ok = false; break; }
      }
      if (ok && score > bestScore) { bestScore = score; best = variants[key]; }
    }
    const chosen = best ?? empty ?? variants[keys[0]];
    return chosen ? [takeModel(chosen)] : [];
  }
  if (entry.multipart) {
    const models = [];
    for (const part of entry.multipart) {
      if (matchesWhen(part.when, props)) models.push(takeModel(part.apply));
    }
    return models;
  }
  return [];
}

// Translate one resolved element face into our compact carry format, or null.
function packFace(f) {
  const tex = f && f.texture;
  if (!tex || typeof tex.u !== "number") return null;
  const out = {
    // atlas tile rect in 0–1 UV space (top-left u,v + size su,sv)
    box: [tex.u, tex.v, tex.su, tex.sv],
  };
  // in-tile sub-rect in 0–16 Minecraft coords; default = whole tile [0,0,16,16]
  if (Array.isArray(f.uv)) out.uv = f.uv;
  if (f.rotation) out.rot = f.rotation;     // 0/90/180/270 texture rotation
  if (f.tintindex !== undefined) out.tint = 1;
  if (f.cullface) out.cull = f.cullface;     // face direction that culls this face
  return out;
}

// Translate one resolved model element (box + rotation + 6 faces) → compact form.
function packElement(el) {
  const faces = {};
  let any = false;
  for (const face of FACES) {
    const pf = packFace(el.faces && el.faces[face]);
    if (pf) { faces[face] = pf; any = true; }
  }
  if (!any) return null;
  const out = { from: el.from, to: el.to, faces };
  if (el.rotation) out.rot = el.rotation; // {origin, axis, angle, rescale}
  if (el.shade === false) out.shade = 0;
  return out;
}

function resolveState(id) {
  let block;
  try { block = Block.fromStateId(id, 0); } catch { return null; }
  if (!block || /(^|_)air$/.test(block.name)) return null;
  const entry = blocksStates[block.name];
  const props = typeof block.getProperties === "function" ? block.getProperties() : block._properties || {};
  const mcBlock = mcData.blocksByName[block.name];
  const translucent = TRANSLUCENT.test(block.name);
  // A block is a solid, face-culling cube only when its model is a single full
  // 0..16 element (handled at mesh time); flag opacity for neighbour culling.
  const opaque = !translucent && !!mcBlock && mcBlock.boundingBox === "block" && !mcBlock.transparent;

  const models = resolveModels(entry, props);
  const elements = [];
  for (const model of models) {
    for (const el of model?.elements ?? []) {
      const pe = packElement(el);
      if (pe) elements.push(pe);
    }
  }

  // Liquids (water/lava) ship an empty-element model but a valid `particle`
  // texture — synthesise a full translucent cube from it so they render as blue
  // water / orange lava instead of falling back to a random hashed colour.
  if (!elements.length) {
    const firstModel = models[0];
    const particle = firstModel?.textures?.particle;
    if (particle && typeof particle.u === "number" && /water|lava/.test(block.name)) {
      const box = [particle.u, particle.v, particle.su, particle.sv];
      const faces = {};
      for (const face of FACES) faces[face] = { box, uv: [0, 0, 16, 16], cull: face };
      return { elements: [{ from: [0, 0, 0], to: [16, 16, 16], faces }], opaque: false, cube: true, translucent: true };
    }
    return null;
  }

  // A "full cube" is one element spanning the whole block on every axis — used by
  // the mesher to cull hidden faces against opaque neighbours.
  const fullCube =
    elements.length === 1 &&
    elements[0].from[0] === 0 && elements[0].from[1] === 0 && elements[0].from[2] === 0 &&
    elements[0].to[0] === 16 && elements[0].to[1] === 16 && elements[0].to[2] === 16 &&
    !elements[0].rot;

  const out = { elements, opaque, cube: fullCube };
  if (translucent) out.translucent = true;
  return out;
}

// ── duckdb helpers ────────────────────────────────────────────────────────────
const pq = PARQUET.replace(/'/g, "''");
function allRows(conn, sql) {
  return new Promise((res, rej) => conn.all(sql, (e, r) => (e ? rej(e) : res(r))));
}

// ── main ──────────────────────────────────────────────────────────────────────
async function main() {
  if (!fs.existsSync(PARQUET)) throw new Error(`Missing ${PARQUET}`);
  fs.rmSync(OUT_RAW, { recursive: true, force: true });
  fs.mkdirSync(OUT_RAW, { recursive: true });
  fs.mkdirSync(OUT_ATLAS_DIR, { recursive: true });

  const db = new duckdb.Database(":memory:");
  const conn = db.connect();

  console.log("Computing top-254 block frequency mapping ...");
  const freq = await allRows(
    conn,
    `SELECT CAST(b AS INTEGER) b, count(*) c FROM (SELECT unnest(voxel_data) b FROM read_parquet('${pq}')) WHERE b <> 0 GROUP BY b ORDER BY c DESC LIMIT ${MAX_BLOCK_TYPES - 2}`,
  );
  const mapping = new Map([[0, 0]]);
  freq.forEach((r, i) => mapping.set(Number(r.b), i + 2)); // 0=air, 2.. = frequent, others -> 1
  console.log(`  mapped ${freq.length} frequent block states`);
  // persist so the schematic-search route can remap uploads into this same space
  fs.writeFileSync(OUT_MAPPING, JSON.stringify(Object.fromEntries(mapping)));

  const remap = (raw) => (raw === 0 ? 0 : mapping.get(raw) ?? 1);

  const items = [];
  const texts = [];
  const featRows = [];
  const usedStates = new Set();
  let n = 0, skipped = 0;

  const limitClause = LIMIT > 0 ? ` LIMIT ${LIMIT}` : "";
  const sql = `SELECT url, title, subtitle, description, tags, "user" AS usr, date,
       diamondCount, views, downloads,
       list_transform(voxel_data, x -> CAST(x AS INTEGER)) AS vd
       FROM read_parquet('${pq}')${limitClause}`;

  console.log("Streaming rows ...");
  for await (const row of conn.stream(sql)) {
    const vd = row.vd;
    if (!vd || vd.length !== VOXELS) { skipped++; continue; }

    // remap -> compact grid for features
    const compact = new Uint8Array(VOXELS);
    const rawU16 = new Uint16Array(VOXELS);
    for (let i = 0; i < VOXELS; i++) {
      const r = vd[i] | 0;
      rawU16[i] = r;
      compact[i] = remap(r);
      if (r !== 0) usedStates.add(r);
    }
    const { grid: resized, dims } = bboxCropResize(compact);
    let nonAir = 0;
    for (let i = 0; i < VOXELS; i++) if (resized[i] !== 0) nonAir++;
    if (nonAir < MIN_NONAIR) { skipped++; continue; }

    const id = `b${String(n).padStart(5, "0")}`;
    const tags = parseTags(row.tags).slice(0, 8);
    items.push({
      id,
      title: clean(row.title) || "Untitled Build",
      category: clean(row.subtitle) || "Other Map",
      description: clean(row.description).slice(0, 320),
      tags,
      user: clean(row.usr),
      url: typeof row.url === "string" ? row.url : null,
      img: null,
      diamonds: Number(row.diamondCount) || 0,
      views: Number(row.views) || 0,
      downloads: Number(row.downloads) || 0,
      year: yearOf(row.date),
      dims,
      fill: Math.round((nonAir / VOXELS) * 1e4) / 1e4,
    });
    texts.push(buildText(row, tags));
    featRows.push(voxelFeatures(resized, dims));

    const buf = Buffer.from(rawU16.buffer, rawU16.byteOffset, rawU16.byteLength);
    fs.writeFileSync(path.join(OUT_RAW, `${id}.bin`), zlib.gzipSync(buf, { level: 9 }));

    n++;
    if (n % 500 === 0) process.stdout.write(`\r  processed ${n} builds (${skipped} skipped)`);
  }
  console.log(`\nWrote ${n} builds (${skipped} skipped)`);

  // features.bin
  const feat = new Float32Array(n * FEAT_DIM);
  featRows.forEach((r, i) => feat.set(r, i * FEAT_DIM));
  fs.writeFileSync(OUT_FEATURES, Buffer.from(feat.buffer));

  // gallery.json (lightweight index) + texts
  fs.writeFileSync(
    OUT_GALLERY,
    JSON.stringify({
      meta: {
        grid: GRID,
        count: n,
        featDim: FEAT_DIM,
        maxBlockTypes: MAX_BLOCK_TYPES,
        source: "Planet Minecraft (rom1504/minecraft-schematics-dataset)",
        note: "Full dataset; voxels rendered client-side from per-build raw grids.",
      },
      items,
    }),
  );
  fs.writeFileSync(OUT_TEXTS, JSON.stringify(texts));

  // atlas — also include every block's default state so uploaded schematics
  // (which we render via default states) get textured, not just dataset states.
  for (const b of mcData.blocksArray) if (typeof b.defaultState === "number") usedStates.add(b.defaultState);

  const blocks = {};
  let resolved = 0;
  for (const id of usedStates) {
    const r = resolveState(id);
    if (r) { blocks[id] = r; resolved++; }
  }
  // schema v2: per-state element-based models (see resolveState) — the runtime
  // mesher renders real geometry from these, not single-texture unit cubes.
  fs.writeFileSync(OUT_ATLAS_JSON, JSON.stringify({ version: VERSION, schema: 2, atlas: `/atlas/${VERSION}.png`, blocks }));
  fs.copyFileSync(path.join(PV_PUBLIC, "textures", `${VERSION}.png`), path.join(OUT_ATLAS_DIR, `${VERSION}.png`));

  // state2block.json — stateId → "minecraft:name[prop=val,...]" for every state
  // that appears in the dataset (+ every block's default state, for uploads). The
  // client feeds these strings straight into the Cubane renderer, so it doesn't
  // need prismarine-block / minecraft-data bundled in the browser.
  //
  // The dataset uses 1.16.4 block names but the vendored resource pack is newer,
  // so the two blocks renamed since 1.16.4 are mapped to their current ids (the
  // only names from this dataset missing from the pack's blockstates).
  const RENAME = { grass: "short_grass", grass_path: "dirt_path" };
  const state2block = {};
  for (const id of usedStates) {
    let b;
    try { b = Block.fromStateId(id, 0); } catch { continue; }
    if (!b || /(^|_)air$/.test(b.name)) continue;
    const name = RENAME[b.name] ?? b.name;
    const props = typeof b.getProperties === "function" ? b.getProperties() : b._properties || {};
    const keys = Object.keys(props);
    const suffix = keys.length ? `[${keys.map((k) => `${k}=${props[k]}`).join(",")}]` : "";
    state2block[id] = `minecraft:${name}${suffix}`;
  }
  fs.writeFileSync(OUT_STATE2BLOCK, JSON.stringify(state2block));

  console.log(`features.bin: ${n}×${FEAT_DIM}  |  atlas states: ${resolved}/${usedStates.size}  |  gallery.json: ${(fs.statSync(OUT_GALLERY).size / 1024 / 1024).toFixed(1)} MB`);
  conn.close();
  db.close(() => {});
}

main().catch((e) => { console.error(e); process.exit(1); });
