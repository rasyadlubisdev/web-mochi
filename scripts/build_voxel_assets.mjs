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

// ── atlas UV resolution (prismarine-block + prebuilt blockstates) ─────────────
function pickVariant(entry, props) {
  if (!entry || !entry.variants) return null;
  const variants = entry.variants;
  const keys = Object.keys(variants);
  const take = (v) => {
    const x = Array.isArray(v) ? v[0] : v;
    return x && x.model ? x.model : x;
  };
  if (keys.length === 1) return take(variants[keys[0]]);
  const want = props || {};
  let fallback = null;
  for (const key of keys) {
    if (key === "") { fallback = fallback ?? variants[key]; continue; }
    if (key.split(",").every((t) => { const [k, v] = t.split("="); return String(want[k]) === v; })) return take(variants[key]);
  }
  return take(fallback ?? variants[keys[0]]);
}
function resolveState(id) {
  let block;
  try { block = Block.fromStateId(id, 0); } catch { return null; }
  if (!block || /(^|_)air$/.test(block.name)) return null;
  const entry = blocksStates[block.name];
  const props = typeof block.getProperties === "function" ? block.getProperties() : block._properties || {};
  const model = pickVariant(entry, props);
  const mcBlock = mcData.blocksByName[block.name];
  const translucent = TRANSLUCENT.test(block.name);
  const opaque = !translucent && !!mcBlock && mcBlock.boundingBox === "block" && !mcBlock.transparent;
  const out = { faces: {}, tint: {}, opaque };
  if (translucent) out.translucent = true;
  const el = model && model.elements && model.elements[0];
  for (const face of FACES) {
    const f = el && el.faces && el.faces[face];
    const tex = f && f.texture;
    if (tex && typeof tex.u === "number") {
      out.faces[face] = [tex.u, tex.v, tex.su, tex.sv];
      if (f.tintindex !== undefined) out.tint[face] = true;
    }
  }
  if (Object.keys(out.faces).length < 6) {
    const t = model && model.textures;
    const fb = t && (t.all || t.side || t.particle);
    const rect = fb && typeof fb.u === "number" ? [fb.u, fb.v, fb.su ?? 1 / 32, fb.sv ?? 1 / 32] : out.faces.up;
    if (rect) for (const face of FACES) if (!out.faces[face]) out.faces[face] = rect;
  }
  return Object.keys(out.faces).length ? out : null;
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

  // atlas
  const blocks = {};
  let resolved = 0;
  for (const id of usedStates) {
    const r = resolveState(id);
    if (r) { blocks[id] = r; resolved++; }
  }
  fs.writeFileSync(OUT_ATLAS_JSON, JSON.stringify({ version: VERSION, atlas: `/atlas/${VERSION}.png`, tile: 1 / 32, blocks }));
  fs.copyFileSync(path.join(PV_PUBLIC, "textures", `${VERSION}.png`), path.join(OUT_ATLAS_DIR, `${VERSION}.png`));

  console.log(`features.bin: ${n}×${FEAT_DIM}  |  atlas states: ${resolved}/${usedStates.size}  |  gallery.json: ${(fs.statSync(OUT_GALLERY).size / 1024 / 1024).toFixed(1)} MB`);
  conn.close();
  db.close(() => {});
}

main().catch((e) => { console.error(e); process.exit(1); });
