// Warm the text-embedding model + gallery embeddings so the first user query is fast.

import { NextResponse } from "next/server";
import { ensureTextEmbeddings, getGallery } from "@/lib/gallery";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const t0 = performance.now();
  try {
    const index = await getGallery();
    await ensureTextEmbeddings();
    return NextResponse.json({
      ready: true,
      builds: index.builds.length,
      tookMs: Math.round(performance.now() - t0),
    });
  } catch (err) {
    return NextResponse.json(
      { ready: false, detail: err instanceof Error ? err.message : String(err) },
      { status: 503 },
    );
  }
}
