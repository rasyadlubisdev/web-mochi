// Cosine similarity + top-k ranking over float vectors.

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dotp = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dotp += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dotp / denom;
}

/**
 * Rank gallery vectors against a query and return the top-k indices + scores.
 * If vectors are already L2-normalised, dot == cosine (faster); set `normalized`.
 */
export function topK(
  query: Float32Array,
  gallery: Float32Array[],
  k: number,
  normalized = true,
): { index: number; score: number }[] {
  const sim = normalized ? dot : cosine;
  const scored = gallery.map((g, index) => ({ index, score: sim(query, g) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
