// Server-only text embedding via transformers.js.
//
// Loads the SAME backbone the research project uses for its TextEncoder —
// sentence-transformers/all-MiniLM-L6-v2 — as an ONNX model through
// @huggingface/transformers, with mean pooling + L2 normalisation (the
// sentence-transformers default). This produces genuine 384-d semantic
// embeddings; cosine similarity between a query and gallery texts is real
// text↔text retrieval (the strongest unimodal baseline in the paper).
//
// NOTE: the trained model adds a learned 384→256 projection head on top of
// this backbone to align text with the voxel space. That head lives only in a
// trained checkpoint (not shipped), so the demo compares in the raw 384-d
// MiniLM space. See README for how to swap in the full DualEncoder.

import "server-only";
import path from "node:path";

export const TEXT_MODEL = "Xenova/all-MiniLM-L6-v2";
export const TEXT_DIM = 384;

type Extractor = (
  texts: string | string[],
  opts: { pooling: "mean"; normalize: boolean },
) => Promise<{ tolist: () => number[][]; dims: number[]; data: Float32Array }>;

let extractorPromise: Promise<Extractor> | null = null;

async function loadExtractor(): Promise<Extractor> {
  const { pipeline, env } = await import("@huggingface/transformers");
  // cache downloaded weights inside the project so repeat runs are offline
  env.cacheDir = path.join(process.cwd(), ".transformers-cache");
  env.allowLocalModels = true;
  const pipe = await pipeline("feature-extraction", TEXT_MODEL);
  return pipe as unknown as Extractor;
}

/** Lazily initialise (and cache) the embedding pipeline. */
export function getExtractor(): Promise<Extractor> {
  if (!extractorPromise) extractorPromise = loadExtractor();
  return extractorPromise;
}

/** Embed one string → L2-normalised Float32Array(384). */
export async function embedText(text: string): Promise<Float32Array> {
  const extractor = await getExtractor();
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return Float32Array.from(out.data);
}

/** Embed many strings → array of L2-normalised Float32Array(384). */
export async function embedTexts(texts: string[]): Promise<Float32Array[]> {
  const extractor = await getExtractor();
  const out = await extractor(texts, { pooling: "mean", normalize: true });
  const rows = out.tolist();
  return rows.map((r) => Float32Array.from(r));
}
