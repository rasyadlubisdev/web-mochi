// Precompute MiniLM text embeddings for every build, so text-search covers the
// whole dataset without embedding thousands of texts at request time.
//
// Reads  public/data/_texts.json  (written by build_voxel_assets.mjs)
// Writes public/data/text-emb.bin  — Float32 [count][384], L2-normalised, aligned
//                                     with gallery.json item order.
//
// Run with Node >= 18 (after build_voxel_assets.mjs):
//   node scripts/build_text_embeddings.mjs

import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const TEXTS = path.join(ROOT, "public/data/_texts.json");
const OUT = path.join(ROOT, "public/data/text-emb.bin");
const MODEL = "Xenova/all-MiniLM-L6-v2";
const DIM = 384;
const BATCH = 64;

async function main() {
  if (!fs.existsSync(TEXTS)) throw new Error(`Missing ${TEXTS} — run build_voxel_assets.mjs first`);
  const texts = JSON.parse(fs.readFileSync(TEXTS, "utf8")).map((t) => (typeof t === "string" && t.trim() ? t : "minecraft build"));
  console.log(`Embedding ${texts.length} texts with ${MODEL} ...`);

  const { pipeline, env } = await import("@huggingface/transformers");
  env.cacheDir = path.join(ROOT, ".transformers-cache");
  env.allowLocalModels = true;
  const pipe = await pipeline("feature-extraction", MODEL);

  const out = new Float32Array(texts.length * DIM);
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const res = await pipe(batch, { pooling: "mean", normalize: true });
    const rows = res.tolist();
    for (let j = 0; j < rows.length; j++) out.set(Float32Array.from(rows[j]), (i + j) * DIM);
    process.stdout.write(`\r  ${Math.min(i + BATCH, texts.length)}/${texts.length}`);
  }
  fs.writeFileSync(OUT, Buffer.from(out.buffer));
  console.log(`\nWrote ${OUT} (${texts.length}×${DIM}, ${(fs.statSync(OUT).size / 1024 / 1024).toFixed(1)} MB)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
