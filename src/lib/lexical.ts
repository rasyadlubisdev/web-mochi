// BM25 lexical index over build texts (server-only).
//
// Pairs with the dense MiniLM embeddings to form a hybrid retriever: dense
// captures meaning, BM25 captures exact terms (block/build names, "redstone",
// "pixel art", …) the embedding can blur. Built once over the whole corpus and
// cached; scoring a query is a cheap walk of the matching postings.

import "server-only";

// Light stoplist — drop ultra-common words + a few domain words that appear in
// almost every build text and so carry no discriminative signal.
const STOP = new Set([
  "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "it", "its",
  "this", "that", "these", "those", "as", "at", "by", "be", "are", "was", "i", "you", "your",
  "my", "we", "they", "have", "has", "will", "can", "all", "from", "but", "not", "so",
  "minecraft", "map", "build", "made", "make",
]);

function tokenize(s: string): string[] {
  const m = s.toLowerCase().match(/[a-z0-9]+/g);
  if (!m) return [];
  return m.filter((t) => t.length > 1 && !STOP.has(t));
}

export interface Bm25Index {
  /** BM25 score per document (length = corpus size); 0 where no query term matches. */
  scoreAll(query: string): Float32Array;
}

export function buildBm25(texts: string[], k1 = 1.5, b = 0.75): Bm25Index {
  const N = texts.length;
  const docLen = new Float32Array(N);
  const postings = new Map<string, { doc: number; tf: number }[]>();
  let total = 0;

  for (let d = 0; d < N; d++) {
    const toks = tokenize(texts[d]);
    docLen[d] = toks.length;
    total += toks.length;
    const tf = new Map<string, number>();
    for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const [t, f] of tf) {
      let p = postings.get(t);
      if (!p) postings.set(t, (p = []));
      p.push({ doc: d, tf: f });
    }
  }
  const avgdl = total / N || 1;

  return {
    scoreAll(query: string): Float32Array {
      const scores = new Float32Array(N);
      const qTerms = new Set(tokenize(query));
      for (const t of qTerms) {
        const p = postings.get(t);
        if (!p) continue;
        const idf = Math.log(1 + (N - p.length + 0.5) / (p.length + 0.5));
        for (const { doc, tf } of p) {
          const denom = tf + k1 * (1 - b + (b * docLen[doc]) / avgdl);
          scores[doc] += (idf * (tf * (k1 + 1))) / denom;
        }
      }
      return scores;
    },
  };
}
