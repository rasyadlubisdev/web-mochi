// Server-only gallery index: loads gallery.json once, decodes every voxel grid,
// precomputes structural voxel features, and (lazily) computes + caches text
// embeddings for all builds. Cached on globalThis so Next dev HMR reuses it.

import "server-only";
import fs from "node:fs";
import path from "node:path";
import type { BuildSummary, GalleryFile, GalleryItemRaw } from "./types";
import { decodeGrid, voxelFeatures } from "./voxel";
import { embedTexts, TEXT_DIM } from "./text-embed";

export interface IndexedBuild {
  raw: GalleryItemRaw;
  summary: BuildSummary;
  features: Float32Array; // structural voxel descriptor (normalised)
}

interface GalleryIndex {
  builds: IndexedBuild[];
  meta: GalleryFile["meta"];
  textEmb: Float32Array[] | null; // aligned with builds; null until warmed
  textEmbReady: Promise<void> | null;
}

const GLOBAL_KEY = "__mc_gallery_index__";

function toSummary(it: GalleryItemRaw): BuildSummary {
  const { voxels: _v, text: _t, ...rest } = it;
  return rest;
}

async function buildIndex(): Promise<GalleryIndex> {
  const file = path.join(process.cwd(), "public", "data", "gallery.json");
  const data = JSON.parse(fs.readFileSync(file, "utf8")) as GalleryFile;

  const builds: IndexedBuild[] = [];
  for (const it of data.items) {
    const grid = await decodeGrid(it.voxels);
    const features = voxelFeatures(grid, it.dims);
    builds.push({ raw: it, summary: toSummary(it), features });
  }

  return { builds, meta: data.meta, textEmb: null, textEmbReady: null };
}

/** Get (or create) the singleton gallery index. */
export async function getGallery(): Promise<GalleryIndex> {
  const g = globalThis as unknown as Record<string, Promise<GalleryIndex> | undefined>;
  if (!g[GLOBAL_KEY]) g[GLOBAL_KEY] = buildIndex();
  return g[GLOBAL_KEY]!;
}

// --- text embeddings (lazy + disk cache) ------------------------------------

function cachePath(count: number): string {
  return path.join(process.cwd(), ".transformers-cache", `gallery-text-emb-${count}.bin`);
}

function loadEmbCache(count: number): Float32Array[] | null {
  try {
    const p = cachePath(count);
    if (!fs.existsSync(p)) return null;
    const buf = fs.readFileSync(p);
    const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const rows: Float32Array[] = [];
    for (let i = 0; i < count; i++) rows.push(flat.slice(i * TEXT_DIM, (i + 1) * TEXT_DIM));
    return rows.length === count ? rows : null;
  } catch {
    return null;
  }
}

function saveEmbCache(count: number, rows: Float32Array[]): void {
  try {
    const flat = new Float32Array(count * TEXT_DIM);
    rows.forEach((r, i) => flat.set(r, i * TEXT_DIM));
    fs.mkdirSync(path.dirname(cachePath(count)), { recursive: true });
    fs.writeFileSync(cachePath(count), Buffer.from(flat.buffer));
  } catch {
    /* best-effort cache */
  }
}

/**
 * Ensure text embeddings for all gallery builds exist. Runs at most once;
 * concurrent callers await the same promise. Embeds in batches to bound memory.
 */
export async function ensureTextEmbeddings(): Promise<Float32Array[]> {
  const index = await getGallery();
  if (index.textEmb) return index.textEmb;
  if (!index.textEmbReady) {
    index.textEmbReady = (async () => {
      const cached = loadEmbCache(index.builds.length);
      if (cached) {
        index.textEmb = cached;
        return;
      }
      const texts = index.builds.map((b) => b.raw.text || b.raw.title);
      const out: Float32Array[] = [];
      const BATCH = 24;
      for (let i = 0; i < texts.length; i += BATCH) {
        out.push(...(await embedTexts(texts.slice(i, i + BATCH))));
      }
      index.textEmb = out;
      saveEmbCache(index.builds.length, out);
    })();
  }
  await index.textEmbReady;
  return index.textEmb!;
}
