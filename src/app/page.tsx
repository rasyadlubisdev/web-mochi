"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { GalleryFile, GalleryItemRaw } from "@/lib/types";
import { decodeRawVoxels } from "@/lib/blockAtlas";
import { BuildCard } from "@/components/BuildCard";
import { BuildDetail } from "@/components/BuildDetail";
import { AboutModal } from "@/components/AboutModal";

const PrismarineViewer = dynamic(() => import("@/components/PrismarineViewer").then((m) => m.PrismarineViewer), {
  ssr: false,
  loading: () => <div className="h-full w-full skeleton rounded-xl" />,
});

const EXAMPLES = [
  "medieval castle with towers",
  "cozy wooden cabin in the woods",
  "futuristic spaceship",
  "japanese pagoda temple",
  "pixel art character",
  "redstone computer",
  "modern glass house",
];

type Mode = "text" | "build";
type Scored = { id: string; score: number };
type ModelStatus = "idle" | "loading" | "ready" | "error";

export default function Home() {
  const [items, setItems] = useState<GalleryItemRaw[] | null>(null);
  const [meta, setMeta] = useState<GalleryFile["meta"] | null>(null);
  const [mode, setMode] = useState<Mode>("text");

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Scored[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const [selected, setSelected] = useState<{ item: GalleryItemRaw; score?: number } | null>(null);
  const [about, setAbout] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [visibleCount, setVisibleCount] = useState(60); // browse pagination

  // upload state (search by schematic file)
  const [uploadedGrid, setUploadedGrid] = useState<Uint16Array | null>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  // --- load gallery + warm the text model ---
  useEffect(() => {
    fetch("/data/gallery.json")
      .then((r) => r.json())
      .then((g: GalleryFile) => {
        setItems(g.items);
        setMeta(g.meta);
      })
      .catch(() => setError("Failed to load gallery data."));

    setModelStatus("loading");
    fetch("/api/warmup")
      .then((r) => (r.ok ? setModelStatus("ready") : setModelStatus("error")))
      .catch(() => setModelStatus("error"));
  }, []);

  const byId = useMemo(() => {
    const m = new Map<string, GalleryItemRaw>();
    items?.forEach((it) => m.set(it.id, it));
    return m;
  }, [items]);

  // --- text search ---
  const runText = useCallback(async (q: string) => {
    const query = q.trim();
    if (!query) return;
    setMode("text");
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k: 48 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.hint || data.error || "Search failed");
      setResults(data.results);
      setModelStatus("ready");
      setInfo(`${data.results.length} builds · ${data.tookMs}ms`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
      setResults(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // --- schematic upload search ---
  const runSchematic = useCallback(async (file: File) => {
    setMode("build");
    setLoading(true);
    setError(null);
    setUploadName(file.name);
    setUploadedGrid(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/schematic", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.hint ? `${data.error} — ${data.hint}` : data.error || "Upload failed");
      setUploadedGrid(await decodeRawVoxels(data.voxels));
      setResults(data.results);
      setInfo(
        `${data.results.length} builds · ${file.name} (${data.stats.dims.join("×")}, ${data.stats.blocks} blocks) · ${data.tookMs}ms`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setUploadedGrid(null);
      setUploadName(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const clearSearch = () => {
    setResults(null);
    setQuery("");
    setError(null);
    setInfo(null);
    setUploadedGrid(null);
    setUploadName(null);
    if (fileInput.current) fileInput.current.value = "";
  };

  const scoreById = useMemo(() => {
    const m = new Map<string, number>();
    results?.forEach((r) => m.set(r.id, r.score));
    return m;
  }, [results]);

  const displayed: { item: GalleryItemRaw; score?: number; rank?: number }[] = useMemo(() => {
    if (!items) return [];
    if (results) {
      return results
        .map((r, i) => ({ item: byId.get(r.id)!, score: r.score, rank: i + 1 }))
        .filter((x) => x.item);
    }
    // browse: highest-engagement first
    return [...items]
      .sort((a, b) => b.diamonds - a.diamonds)
      .map((item) => ({ item }));
  }, [items, results, byId]);

  return (
    <div className="min-h-screen">
      {/* header */}
      <header className="sticky top-0 z-30 border-b border-[var(--color-border)] bg-[var(--color-bg)]/80 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-[var(--color-voxel)] to-[var(--color-accent)] grid place-items-center text-lg">🧊</div>
            <div>
              <h1 className="text-sm font-semibold text-white leading-tight">MC-Retrieval</h1>
              <p className="text-[11px] text-white/45 leading-tight">Cross-modal Minecraft schematic search</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <ModelBadge status={modelStatus} />
            <button onClick={() => setAbout(true)} className="text-xs px-3 py-1.5 rounded-lg border border-[var(--color-border)] text-white/70 hover:text-white hover:border-white/30">
              How it works
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        {/* mode tabs */}
        <div className="flex items-center gap-2 mb-5">
          <TabButton active={mode === "text"} onClick={() => setMode("text")} icon="🔍" label="Search by Text" />
          <TabButton active={mode === "build"} onClick={() => setMode("build")} icon="🧱" label="Search by Building" />
        </div>

        {/* query area */}
        {mode === "text" ? (
          <section className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-panel)] p-5">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                runText(query);
              }}
              className="flex gap-2"
            >
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe a build — e.g. 'a tall medieval castle with stone towers'"
                className="flex-1 rounded-xl bg-[var(--color-panel-2)] border border-[var(--color-border)] px-4 py-3 text-sm text-white placeholder:text-white/35 outline-none focus:border-[var(--color-voxel)]/60"
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="px-5 py-3 rounded-xl bg-[var(--color-voxel)] text-black text-sm font-semibold hover:bg-emerald-400 disabled:opacity-40 transition-colors"
              >
                {loading ? "Searching…" : "Search"}
              </button>
            </form>
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="text-[11px] text-white/40 self-center">Try:</span>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => {
                    setQuery(ex);
                    runText(ex);
                  }}
                  className="text-[11px] px-2.5 py-1 rounded-full bg-[var(--color-panel-2)] border border-[var(--color-border)] text-white/60 hover:text-white hover:border-[var(--color-voxel)]/50"
                >
                  {ex}
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="grid lg:grid-cols-[1fr_280px] gap-4">
            {/* uploaded schematic preview */}
            <div className="relative rounded-2xl border border-[var(--color-border)] bg-[#0d0d16] overflow-hidden h-[440px]">
              {uploadedGrid ? (
                <PrismarineViewer grid={uploadedGrid} className="h-full w-full" />
              ) : (
                <label
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const f = e.dataTransfer.files?.[0];
                    if (f) runSchematic(f);
                  }}
                  className="h-full w-full grid place-items-center cursor-pointer text-center px-6 hover:bg-white/[0.02] transition-colors"
                >
                  <div>
                    <div className="text-4xl mb-3">📦</div>
                    <div className="text-sm text-white/80 font-medium">
                      {loading ? "Parsing schematic…" : "Drop a schematic here, or click to browse"}
                    </div>
                    <div className="text-[11px] text-white/40 mt-1">.schem (WorldEdit) · .schematic (MCEdit)</div>
                  </div>
                </label>
              )}
              {uploadedGrid && (
                <div className="absolute bottom-3 left-3 text-[11px] text-white/45 mono pointer-events-none">
                  drag to orbit · your uploaded schematic
                </div>
              )}
            </div>

            {/* upload controls */}
            <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-panel)] p-4 flex flex-col gap-4">
              <div>
                <div className="text-xs text-white/50 mb-2">Search by schematic</div>
                <p className="text-[12px] text-white/55 leading-relaxed">
                  Upload a Minecraft schematic — we voxelise it, show a 3D preview, and rank the most structurally similar community builds.
                </p>
              </div>

              <input
                ref={fileInput}
                type="file"
                accept=".schem,.schematic"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) runSchematic(f);
                }}
              />
              <button
                onClick={() => fileInput.current?.click()}
                disabled={loading}
                className="py-2.5 rounded-lg bg-[var(--color-voxel)] text-black text-sm font-semibold hover:bg-emerald-400 disabled:opacity-40"
              >
                {loading ? "Parsing…" : uploadedGrid ? "Upload another →" : "Choose file →"}
              </button>

              {uploadName && (
                <div className="text-[11px] text-white/55 mono break-all rounded-lg bg-[var(--color-panel-2)] px-3 py-2">
                  {uploadName}
                </div>
              )}

              <div className="mt-auto text-[11px] text-white/35 leading-relaxed">
                Supported: WorldEdit <span className="mono">.schem</span> and legacy MCEdit <span className="mono">.schematic</span>. Large builds are scaled to a 32³ grid with proportions preserved.
              </div>
            </div>
          </section>
        )}

        {/* status line */}
        <div className="mt-5 flex items-center justify-between min-h-[24px]">
          <div className="text-sm">
            {error ? (
              <span className="text-red-400">⚠ {error}</span>
            ) : results ? (
              <span className="text-white/70">
                {mode === "text" ? "Ranked by text similarity" : "Ranked by structural similarity"}
                {info && <span className="text-white/40 mono ml-2">· {info}</span>}
              </span>
            ) : items ? (
              <span className="text-white/50">
                Browsing {meta?.count ?? items.length} community builds — search above to rank by similarity
              </span>
            ) : (
              <span className="text-white/40">Loading gallery…</span>
            )}
          </div>
          {results && (
            <button onClick={clearSearch} className="text-xs text-white/50 hover:text-white">
              ✕ Clear results
            </button>
          )}
        </div>

        {/* results / browse grid */}
        <section className="mt-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {!items
            ? Array.from({ length: 10 }).map((_, i) => <div key={i} className="aspect-[4/3] rounded-xl skeleton" />)
            : (results ? displayed : displayed.slice(0, visibleCount)).map(({ item, score, rank }) => (
                <BuildCard
                  key={item.id}
                  item={item}
                  score={score}
                  rank={rank}
                  onOpen={() => setSelected({ item, score })}
                />
              ))}
        </section>

        {/* browse pagination */}
        {items && !results && visibleCount < displayed.length && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={() => setVisibleCount((v) => v + 60)}
              className="text-sm px-5 py-2.5 rounded-xl border border-[var(--color-border)] text-white/70 hover:text-white hover:border-[var(--color-voxel)]/50"
            >
              Load more ({displayed.length - visibleCount} left)
            </button>
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-7xl px-4 py-8 text-center text-xs text-white/30">
        MC-Retrieval demo · CLIP-style text↔voxel retrieval · data from Planet Minecraft
      </footer>

      {selected && <BuildDetail item={selected.item} score={selected.score} onClose={() => setSelected(null)} />}
      {about && <AboutModal onClose={() => setAbout(false)} />}
    </div>
  );
}

function TabButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: string; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition ${
        active
          ? "bg-[var(--color-panel)] border-[var(--color-voxel)]/60 text-white"
          : "border-transparent text-white/50 hover:text-white/80"
      }`}
    >
      <span>{icon}</span>
      {label}
    </button>
  );
}

function ModelBadge({ status }: { status: ModelStatus }) {
  const map = {
    idle: { c: "#64748b", t: "model idle" },
    loading: { c: "#f59e0b", t: "loading model…" },
    ready: { c: "#10b981", t: "model ready" },
    error: { c: "#ef4444", t: "model offline" },
  }[status];
  return (
    <span className="hidden sm:flex items-center gap-1.5 text-[11px] text-white/50 mono">
      <span className="h-2 w-2 rounded-full" style={{ background: map.c, boxShadow: `0 0 8px ${map.c}` }} />
      {map.t}
    </span>
  );
}
