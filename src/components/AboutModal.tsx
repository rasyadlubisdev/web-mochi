"use client";

export function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4 bg-black/70 backdrop-blur-sm" onClick={onClose}>
      <div
        className="fade-up w-full max-w-2xl max-h-[88vh] overflow-y-auto rounded-2xl border border-[var(--color-border)] bg-[var(--color-panel)] shadow-2xl p-7"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <h2 className="text-xl font-semibold text-white">How this demo works</h2>
          <button onClick={onClose} className="h-8 w-8 grid place-items-center rounded-lg border border-[var(--color-border)] text-white/60 hover:text-white">✕</button>
        </div>

        <p className="mt-4 text-sm text-white/70 leading-relaxed">
          This is a demo of <strong className="text-white">cross-modal retrieval</strong> between natural language and
          3D Minecraft voxel schematics — the final project for the <em>Retrieval Information</em> course. The research
          model learns a shared embedding space with a CLIP-style dual encoder (a 3D-CNN <span className="text-[var(--color-voxel)]">VoxelEncoder</span> and
          a frozen <span className="text-[var(--color-text)]">all-MiniLM-L6-v2 TextEncoder</span>), trained with symmetric InfoNCE.
        </p>

        <div className="mt-5 space-y-4 text-sm">
          <Section
            title="🔍 Text → Build"
            color="var(--color-text)"
            body="Hybrid retrieval over the full dataset: a dense semantic score (your query embedded with all-MiniLM-L6-v2, cosine similarity to each build's title+subtitle+description+tags embedding) is fused with a lexical BM25 score. So 'cozy medieval house' matches on meaning, while exact terms like 'redstone' or 'AT-AT' still rank precisely."
          />
          <Section
            title="🧱 Build → Build"
            color="var(--color-voxel)"
            body="Your voxel structure is run through the same preprocessing as training (bounding-box crop + nearest-neighbour resize to 32³), then turned into a structural feature descriptor (block palette, fill, aspect ratio, vertical/horizontal mass profiles, symmetry). Cosine similarity over these descriptors finds structurally similar community builds."
          />
        </div>

        <div className="mt-5 rounded-lg border border-[var(--color-accent)]/30 bg-[var(--color-accent)]/5 p-4 text-sm text-white/70">
          <div className="font-medium text-[var(--color-accent)] mb-1">A note on fidelity</div>
          No trained checkpoint ships with this repo, so the demo runs the <strong className="text-white">real MiniLM text
          backbone</strong> for the text side and a <strong className="text-white">handcrafted structural descriptor</strong> as
          a stand-in for the trained VoxelEncoder. The UX, preprocessing, and retrieval flow mirror the real system; swapping
          in a trained <span className="mono">DualEncoder</span> checkpoint (via the documented inference adapter) upgrades both
          sides to the learned 256-d aligned space. See the project README.
        </div>

        <div className="mt-5 text-xs text-white/40">
          Dataset: 8,328 schematics scraped from Planet Minecraft (rom1504/minecraft-schematics-dataset). This demo indexes a
          curated, category-balanced subset.
        </div>
      </div>
    </div>
  );
}

function Section({ title, body, color }: { title: string; body: string; color: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-panel-2)] p-4">
      <div className="font-medium mb-1" style={{ color }}>{title}</div>
      <p className="text-white/65 leading-relaxed">{body}</p>
    </div>
  );
}
