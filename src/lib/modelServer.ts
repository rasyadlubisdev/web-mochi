// Base URL of the Python torch inference sidecar (model_server/serve.py).
// The Next API routes proxy to it; the browser never calls it directly.
export const MODEL_SERVER = process.env.MODEL_SERVER_URL || "http://127.0.0.1:8008";

export class ModelServerError extends Error {}

/** Shared error payload when the sidecar is unreachable. */
export function sidecarDownResponse(detail: string) {
  return {
    error: "Model server unavailable",
    detail,
    hint: "Start it: `conda activate mcmodel && python model_server/serve.py` (from the web project root).",
  };
}
