// Image → Build retrieval via the trained tri-modal model.
// Forwards the uploaded image to the Python sidecar, which embeds it with the
// model's CLIP image encoder and ranks the gallery's voxel embeddings (image→voxel).

import { NextRequest, NextResponse } from "next/server";
import { MODEL_SERVER, sidecarDownResponse } from "@/lib/modelServer";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_FILE = 15 * 1024 * 1024;

export async function POST(req: NextRequest) {
  let file: File | null = null;
  try {
    const form = await req.formData();
    const f = form.get("file");
    if (f && typeof f !== "string") file = f;
  } catch {
    return NextResponse.json({ error: "Invalid upload" }, { status: 400 });
  }
  if (!file) return NextResponse.json({ error: "No image uploaded (field 'file')" }, { status: 400 });
  if (file.size > MAX_FILE) return NextResponse.json({ error: "Image too large (max 15 MB)" }, { status: 400 });

  try {
    const fd = new FormData();
    fd.append("file", file, file.name || "image");
    fd.append("k", "48");
    const res = await fetch(`${MODEL_SERVER}/search/image`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json(sidecarDownResponse(e instanceof Error ? e.message : String(e)), { status: 503 });
  }
}
