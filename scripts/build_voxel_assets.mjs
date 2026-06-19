// Build-time asset generator for the live textured 3D preview.
//
// Reads each gallery build's RAW voxel_data (real Minecraft block-state IDs, at
// their true proportions) straight from public/data/data.parquet, and emits:
//   public/data/raw/<id>.bin   gzip(uint16 LE [32^3])  — per-build block grid
//   public/data/block_atlas.json                       — stateId -> per-face atlas UVs
//   public/atlas/<version>.png                         — the block texture atlas
//
// The parquet's voxel_data is INT64 in a single 8k-row group, so a JS reader
// would have to expand the whole column (~273M values) into memory. DuckDB
// scans natively and only materialises the handful of rows we ask for.
//
// The atlas + per-face UV mapping are reused from prismarine-viewer's prebuilt
// 1.16.4 data (generated from minecraft-assets); block names/variants are
// resolved with prismarine-block. The browser renderer (three@0.172) then just
// samples the atlas — no prismarine runtime, no three.js version conflict.
//
// Run with Node >= 18:  node scripts/build_voxel_assets.mjs

import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";

const require = createRequire(import.meta.url);
const duckdb = require("duckdb");

const VERSION = "1.16.4";
const GRID = 32;
const VOXELS = GRID * GRID * GRID;
const URL_BATCH = 40; // builds per DuckDB query (keeps JS heap small)

const ROOT = process.cwd();
const PARQUET = path.join(ROOT, "public/data/data.parquet");
const GALLERY = path.join(ROOT, "public/data/gallery.json");
const PV_PUBLIC = path.join(ROOT, "node_modules/prismarine-viewer/public");
const OUT_RAW = path.join(ROOT, "public/data/raw");
const OUT_ATLAS_DIR = path.join(ROOT, "public/atlas");
const OUT_ATLAS_JSON = path.join(ROOT, "public/data/block_atlas.json");

const FACES = ["up", "down", "north", "south", "east", "west"];
const TRANSLUCENT = /glass|^water$|^ice$|frosted_ice|slime_block|honey_block|nether_portal|barrier|^lava$/;

const mcData = require("minecraft-data")(VERSION);
const Block = require("prismarine-block")(VERSION);
const blocksStates = require(path.join(PV_PUBLIC, "blocksStates", `${VERSION}.json`));

// --- variant / model resolution -------------------------------------------

function pickVariant(entry, props) {
  if (!entry || !entry.variants) return null;
  const variants = entry.variants;
  const keys = Object.keys(variants);
  // each variant value is `{ model: <model> }` (or an array of those); unwrap it
  const take = (v) => {
    const x = Array.isArray(v) ? v[0] : v;
    return x && x.model ? x.model : x;
  };
  if (keys.length === 1) return take(variants[keys[0]]);

  const want = props || {};
  let fallback = null;
  for (const key of keys) {
    if (key === "") {
      fallback = fallback ?? variants[key];
      continue;
    }
    const tokens = key.split(",");
    if (tokens.every((t) => {
      const [k, v] = t.split("=");
      return String(want[k]) === v;
    })) {
      return take(variants[key]);
    }
  }
  return take(fallback ?? variants[keys[0]]);
}

/** Resolve a state id to { faces: {face:[u,v,su,sv]}, tint:{face:true}, opaque }. */
function resolveState(id) {
  let block;
  try {
    block = Block.fromStateId(id, 0);
  } catch {
    return null;
  }
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
  // backfill any missing face from particle/all/side texture
  if (Object.keys(out.faces).length < 6) {
    const t = model && model.textures;
    const fb = t && (t.all || t.side || t.particle);
    const rect = fb && typeof fb.u === "number" ? [fb.u, fb.v, fb.su ?? 1 / 32, fb.sv ?? 1 / 32] : out.faces.up;
    if (rect) for (const face of FACES) if (!out.faces[face]) out.faces[face] = rect;
  }
  return Object.keys(out.faces).length ? out : null;
}

// --- duckdb helper ---------------------------------------------------------

function queryVoxels(conn, urls) {
  const list = urls.map((u) => `'${String(u).replace(/'/g, "''")}'`).join(",");
  const pq = PARQUET.replace(/'/g, "''");
  const sql = `SELECT url, voxel_data FROM read_parquet('${pq}') WHERE url IN (${list})`;
  return new Promise((resolve, reject) => {
    conn.all(sql, (err, rows) => (err ? reject(err) : resolve(rows)));
  });
}

// --- main ------------------------------------------------------------------

async function main() {
  if (!fs.existsSync(PARQUET)) throw new Error(`Missing ${PARQUET}`);
  const gallery = JSON.parse(fs.readFileSync(GALLERY, "utf8"));
  const items = gallery.items.filter((it) => typeof it.url === "string" && it.url);
  const byUrl = new Map(items.map((it) => [it.url, it]));
  console.log(`Gallery: ${gallery.items.length} builds (${items.length} with urls)`);

  fs.mkdirSync(OUT_RAW, { recursive: true });
  fs.mkdirSync(OUT_ATLAS_DIR, { recursive: true });

  const db = new duckdb.Database(":memory:");
  const conn = db.connect();

  const usedStates = new Set();
  let written = 0;
  const urls = items.map((it) => it.url);

  for (let i = 0; i < urls.length; i += URL_BATCH) {
    const batch = urls.slice(i, i + URL_BATCH);
    const rows = await queryVoxels(conn, batch);
    for (const row of rows) {
      const it = byUrl.get(row.url);
      const vd = row.voxel_data;
      if (!it || !vd || vd.length !== VOXELS) continue;
      const out = new Uint16Array(VOXELS);
      for (let k = 0; k < VOXELS; k++) {
        const sid = Number(vd[k]) | 0;
        out[k] = sid;
        if (sid !== 0) usedStates.add(sid);
      }
      const buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength);
      fs.writeFileSync(path.join(OUT_RAW, `${it.id}.bin`), zlib.gzipSync(buf, { level: 9 }));
      written++;
    }
    process.stdout.write(`\r  raw voxels: ${written}/${items.length}`);
  }
  console.log(`\nWrote ${written} raw voxel files`);

  const blocks = {};
  let resolved = 0;
  for (const id of usedStates) {
    const r = resolveState(id);
    if (r) {
      blocks[id] = r;
      resolved++;
    }
  }
  console.log(`Resolved ${resolved}/${usedStates.size} distinct block states to atlas UVs`);

  fs.writeFileSync(
    OUT_ATLAS_JSON,
    JSON.stringify({ version: VERSION, atlas: `/atlas/${VERSION}.png`, tile: 1 / 32, blocks }),
  );
  fs.copyFileSync(path.join(PV_PUBLIC, "textures", `${VERSION}.png`), path.join(OUT_ATLAS_DIR, `${VERSION}.png`));
  const sizeKB = (fs.statSync(OUT_ATLAS_JSON).size / 1024).toFixed(0);
  console.log(`Wrote block_atlas.json (${sizeKB} KB) + atlas png -> public/atlas/${VERSION}.png`);

  conn.close();
  db.close(() => {});
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
