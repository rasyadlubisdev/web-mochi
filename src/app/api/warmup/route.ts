// Health check for the model sidecar so the UI can show "model ready".

import { NextResponse } from "next/server";
import { MODEL_SERVER, sidecarDownResponse } from "@/lib/modelServer";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const res = await fetch(`${MODEL_SERVER}/health`, { cache: "no-store" });
    const data = await res.json();
    return NextResponse.json({ ready: !!data.ok, ...data }, { status: res.ok ? 200 : 503 });
  } catch (e) {
    return NextResponse.json(sidecarDownResponse(e instanceof Error ? e.message : String(e)), { status: 503 });
  }
}
