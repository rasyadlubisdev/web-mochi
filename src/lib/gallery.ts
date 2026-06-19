// Server-only gallery index: loads the lightweight gallery.json metadata index
// plus the precomputed binaries built by scripts/build_voxel_assets.mjs and
// scripts/build_text_embeddings.mjs — structural voxel features (features.bin)
// and MiniLM text embeddings (text-emb.bin), both aligned with item order. No
// per-request voxel decoding or text embedding, so the whole dataset is served
// cheaply. Cached on globalThis so Next dev HMR reuses it.

import "server-only";
import fs from "node:fs";
import path from "node:path";
import type { BuildSummary, GalleryFile, GalleryItemRaw } from "./types";
import { TEXT_DIM } from "./text-embed";

export interface IndexedBuild {
  raw: GalleryItemRaw;
  summary: BuildSummary;
  features: Float32Array; // structural voxel descriptor (normalised)
}

interface GalleryIndex {
  builds: IndexedBuild[];
  meta: GalleryFile["meta"];
  textEmb: Float32Array[] | null; // aligned with builds; null until loaded
  textEmbReady: Promise<void> | null;
}

const GLOBAL_KEY = "__mc_gallery_index__";
const dataPath = (f: string) => path.join(process.cwd(), "public", "data", f);

function toSummary(it: GalleryItemRaw): BuildSummary {
  return it; // GalleryItemRaw is already the lightweight summary shape
}

/** Read a packed Float32 matrix → one Float32Array row per build. */
function readMatrix(file: string, count: number, dim: number): Float32Array[] {
  const buf = fs.readFileSync(file);
  const flat = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
  const rows: Float32Array[] = [];
  for (let i = 0; i < count; i++) rows.push(flat.slice(i * dim, (i + 1) * dim));
  return rows;
}

function buildIndex(): GalleryIndex {
  const data = JSON.parse(fs.readFileSync(dataPath("gallery.json"), "utf8")) as GalleryFile;
  const featDim = data.meta.featDim;
  const features = readMatrix(dataPath("features.bin"), data.items.length, featDim);

  const builds: IndexedBuild[] = data.items.map((it, i) => ({
    raw: it,
    summary: toSummary(it),
    features: features[i],
  }));

  return { builds, meta: data.meta, textEmb: null, textEmbReady: null };
}

/** Get (or create) the singleton gallery index. */
export async function getGallery(): Promise<GalleryIndex> {
  const g = globalThis as unknown as Record<string, GalleryIndex | undefined>;
  if (!g[GLOBAL_KEY]) g[GLOBAL_KEY] = buildIndex();
  return g[GLOBAL_KEY]!;
}

/**
 * Load the precomputed text embeddings (text-emb.bin). Runs at most once;
 * concurrent callers await the same promise.
 */
export async function ensureTextEmbeddings(): Promise<Float32Array[]> {
  const index = await getGallery();
  if (index.textEmb) return index.textEmb;
  if (!index.textEmbReady) {
    index.textEmbReady = (async () => {
      index.textEmb = readMatrix(dataPath("text-emb.bin"), index.builds.length, TEXT_DIM);
    })();
  }
  await index.textEmbReady;
  return index.textEmb!;
}
