// Text → Build retrieval.
// Embeds the free-text query with all-MiniLM-L6-v2 and ranks gallery builds by
// cosine similarity against their (title+subtitle+description+tags) embeddings.

import { NextRequest, NextResponse } from "next/server";
import { getGallery, ensureTextEmbeddings } from "@/lib/gallery";
import { embedText } from "@/lib/text-embed";
import { dot } from "@/lib/similarity";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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
    const [galleryEmb, queryEmb] = await Promise.all([
      ensureTextEmbeddings(),
      embedText(query),
    ]);

    const scored = index.builds.map((b, i) => ({
      id: b.raw.id,
      score: dot(queryEmb, galleryEmb[i]), // both L2-normalised → cosine
    }));
    scored.sort((a, b) => b.score - a.score);

    return NextResponse.json({
      mode: "text",
      query,
      results: scored.slice(0, k),
      tookMs: Math.round(performance.now() - t0),
    });
  } catch (err) {
    console.error("text search failed:", err);
    return NextResponse.json(
      {
        error: "Text embedding model unavailable",
        detail: err instanceof Error ? err.message : String(err),
        hint: "First run downloads all-MiniLM-L6-v2 (~25MB) from the HuggingFace hub — check network access.",
      },
      { status: 503 },
    );
  }
}
