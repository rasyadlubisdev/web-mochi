// Text → Build retrieval via the trained tri-modal model.
// Proxies the query to the Python sidecar, which embeds it with the model's
// CLIP text encoder and ranks the gallery's voxel embeddings (text→voxel).

import { NextRequest, NextResponse } from "next/server";
import { MODEL_SERVER, sidecarDownResponse } from "@/lib/modelServer";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let body: { query?: string; k?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  const query = (body.query ?? "").trim();
  const k = Math.min(Math.max(body.k ?? 48, 1), 100);
  if (!query) return NextResponse.json({ error: "Empty query" }, { status: 400 });

  try {
    const res = await fetch(`${MODEL_SERVER}/search/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k }),
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    return NextResponse.json({ ...data, query });
  } catch (e) {
    return NextResponse.json(sidecarDownResponse(e instanceof Error ? e.message : String(e)), { status: 503 });
  }
}
