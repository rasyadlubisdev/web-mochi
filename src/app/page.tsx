"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { GalleryFile, GalleryItemRaw } from "@/lib/types";
import { GRID } from "@/lib/types";
import { bytesToBase64, idx, VOXELS } from "@/lib/voxel";
import { blockColor } from "@/lib/blocks";
import { BuildCard } from "@/components/BuildCard";
import { BuildDetail } from "@/components/BuildDetail";
import { AboutModal } from "@/components/AboutModal";
import type { VoxelMap } from "@/components/VoxelBuilder";

const VoxelBuilder = dynamic(() => import("@/components/VoxelBuilder").then((m) => m.VoxelBuilder), {
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

const PALETTE = [
  { id: 2, name: "Stone" },
  { id: 4, name: "Grass" },
  { id: 14, name: "Wood" },
  { id: 13, name: "Leaves" },
  { id: 9, name: "Water" },
  { id: 11, name: "Sand" },
  { id: 10, name: "Brick" },
  { id: 15, name: "Quartz" },
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

  // builder state
  const [voxels, setVoxels] = useState<VoxelMap>({});
  const [block, setBlock] = useState(2);
  const [erase, setErase] = useState(false);

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

  // --- voxel search ---
  const runVoxel = useCallback(async () => {
    const keys = Object.keys(voxels);
    if (keys.length === 0) {
      setError("Place some blocks first, then search.");
      return;
    }
    setLoading(true);
    setError(null);
    const grid = new Uint8Array(VOXELS);
    for (const k of keys) {
      const [x, y, z] = k.split(",").map(Number);
      if (x < GRID && y < GRID && z < GRID) grid[idx(x, y, z)] = voxels[k];
    }
    try {
      const res = await fetch("/api/search/voxel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grid: bytesToBase64(grid), k: 48 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Search failed");
      setResults(data.results);
      setInfo(`${data.results.length} builds · ${keys.length} blocks → ${data.stats.dims.join("×")} · ${data.tookMs}ms`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }, [voxels]);

  const clearSearch = () => {
    setResults(null);
    setQuery("");
    setError(null);
    setInfo(null);
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
            <div className="relative rounded-2xl border border-[var(--color-border)] bg-[var(--color-panel)] overflow-hidden h-[440px]">
              <VoxelBuilder voxels={voxels} setVoxels={setVoxels} selectedBlock={block} erase={erase} className="h-full w-full" />
              <div className="absolute top-3 left-3 text-[11px] text-white/45 mono pointer-events-none">
                click ground/faces to build · {Object.keys(voxels).length} blocks
              </div>
            </div>

            {/* builder controls */}
            <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-panel)] p-4 flex flex-col gap-4">
              <div>
                <div className="text-xs text-white/50 mb-2">Block palette</div>
                <div className="grid grid-cols-4 gap-2">
                  {PALETTE.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setBlock(p.id);
                        setErase(false);
                      }}
                      title={p.name}
                      className={`aspect-square rounded-lg border-2 transition ${
                        block === p.id && !erase ? "border-white scale-105" : "border-transparent hover:border-white/40"
                      }`}
                      style={{ background: blockColor(p.id) }}
                    />
                  ))}
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => setErase(false)}
                  className={`flex-1 text-xs py-2 rounded-lg border ${!erase ? "border-[var(--color-voxel)] text-[var(--color-voxel)]" : "border-[var(--color-border)] text-white/60"}`}
                >
                  🧱 Build
                </button>
                <button
                  onClick={() => setErase(true)}
                  className={`flex-1 text-xs py-2 rounded-lg border ${erase ? "border-red-400 text-red-400" : "border-[var(--color-border)] text-white/60"}`}
                >
                  🧽 Erase
                </button>
              </div>

              <div>
                <div className="text-xs text-white/50 mb-2">Quick presets</div>
                <div className="grid grid-cols-2 gap-2">
                  {(["Tower", "House", "Pyramid", "Tree"] as const).map((name) => (
                    <button
                      key={name}
                      onClick={async () => {
                        const { PRESETS } = await import("@/components/VoxelBuilder");
                        setVoxels(PRESETS[name]());
                      }}
                      className="text-xs py-2 rounded-lg bg-[var(--color-panel-2)] border border-[var(--color-border)] text-white/70 hover:border-white/30"
                    >
                      {name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-auto flex flex-col gap-2">
                <button
                  onClick={() => setVoxels({})}
                  className="text-xs py-2 rounded-lg border border-[var(--color-border)] text-white/60 hover:text-white"
                >
                  Clear all
                </button>
                <button
                  onClick={runVoxel}
                  disabled={loading || Object.keys(voxels).length === 0}
                  className="py-2.5 rounded-lg bg-[var(--color-voxel)] text-black text-sm font-semibold hover:bg-emerald-400 disabled:opacity-40"
                >
                  {loading ? "Searching…" : "Find similar builds →"}
                </button>
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
            : displayed.map(({ item, score, rank }) => (
                <BuildCard
                  key={item.id}
                  item={item}
                  score={score}
                  rank={rank}
                  onOpen={() => setSelected({ item, score })}
                />
              ))}
        </section>
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
