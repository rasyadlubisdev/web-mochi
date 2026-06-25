// Text → Build retrieval (hybrid: dense semantic + lexical BM25).
//
// Dense: embeds the query with all-MiniLM-L6-v2 and takes cosine similarity to
// each build's precomputed text embedding. Lexical: BM25 over the build texts.
// The two are min-max fused so exact terms (build names, "redstone", "pagoda")
// rank correctly while meaning-based matching still works.

import { NextRequest, NextResponse } from "next/server";
import { getGallery, ensureTextEmbeddings, ensureLexical } from "@/lib/gallery";
import { embedText } from "@/lib/text-embed";
import { dot } from "@/lib/similarity";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// fusion weights (dense semantic vs lexical BM25), both normalised to [0,1]
const W_DENSE = 0.7;
const W_LEX = 0.3;

export async function POST(req: NextRequest) {
  const t0 = performance.now();
  let body: { query?: string; k?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const query = (body.query ?? "").trim();
  const k = Math.min(Math.max(body.k ?? 24, 1), 100);
  if (!query) return NextResponse.json({ error: "Empty query" }, { status: 400 });

  try {
    const index = await getGallery();
    const lexical = await ensureLexical();

    // Dense side (best-effort: if the model/embeddings are unavailable we still
    // serve lexical-only results instead of failing the request).
    let galleryEmb: Float32Array[] | null = null;
    let queryEmb: Float32Array | null = null;
    try {
      [galleryEmb, queryEmb] = await Promise.all([ensureTextEmbeddings(), embedText(query)]);
    } catch {
      galleryEmb = null;
      queryEmb = null;
    }

    const lexScores = lexical ? lexical.scoreAll(query) : null;
    let maxLex = 0;
    if (lexScores) for (let i = 0; i < lexScores.length; i++) if (lexScores[i] > maxLex) maxLex = lexScores[i];

    const hasDense = !!(galleryEmb && queryEmb);
    const hasLex = !!(lexScores && maxLex > 0);

    const scored = index.builds.map((b, i) => {
      const dense = hasDense ? Math.max(0, Math.min(1, dot(queryEmb!, galleryEmb![i]))) : 0;
      const lex = hasLex ? lexScores![i] / maxLex : 0;
      // when only one signal is present, use it alone (don't penalise by the missing weight)
      const score =
        hasDense && hasLex ? W_DENSE * dense + W_LEX * lex : hasDense ? dense : lex;
      return { id: b.raw.id, score };
    });
    scored.sort((a, b) => b.score - a.score);

    return NextResponse.json({
      mode: "text",
      query,
      method: hasDense && hasLex ? "hybrid" : hasDense ? "dense" : "lexical",
      results: scored.slice(0, k),
      tookMs: Math.round(performance.now() - t0),
    });
  } catch (err) {
    console.error("text search failed:", err);
    return NextResponse.json(
      {
        error: "Text search unavailable",
        detail: err instanceof Error ? err.message : String(err),
        hint: "First run downloads all-MiniLM-L6-v2 (~25MB) from the HuggingFace hub — check network access.",
      },
      { status: 503 },
    );
  }
}
