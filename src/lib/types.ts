// Shared types for the MC-Retrieval web demo.

export const GRID = 32;

/**
 * A build's metadata as stored in public/data/gallery.json. This is a
 * lightweight index for the whole dataset — voxels are NOT inlined; they're
 * fetched per build from public/data/raw/<id>.bin and rendered client-side.
 */
export interface GalleryItemRaw {
  id: string;
  title: string;
  category: string;
  description: string;
  tags: string[];
  user: string;
  url: string | null;
  img: string | null;
  diamonds: number;
  views: number;
  downloads: number;
  year: number | null;
  dims: [number, number, number]; // original cropped extents (pre-resize)
  fill: number;
}

export interface GalleryFile {
  meta: {
    grid: number;
    count: number;
    featDim: number;
    maxBlockTypes: number;
    source: string;
    note: string;
  };
  items: GalleryItemRaw[];
}

/** What the client renders in a result card / detail view (no heavy embeddings). */
export interface BuildSummary {
  id: string;
  title: string;
  category: string;
  description: string;
  tags: string[];
  user: string;
  url: string | null;
  img: string | null;
  diamonds: number;
  views: number;
  downloads: number;
  year: number | null;
  dims: [number, number, number];
  fill: number;
}

/** Search APIs return only id + score; the client maps ids back to gallery items. */
export interface SearchResult {
  id: string;
  score: number; // cosine similarity in [-1, 1]
}

export type SearchMode = "text" | "voxel";

export interface TextSearchResponse {
  mode: "text";
  query: string;
  results: SearchResult[];
  tookMs: number;
}

export interface VoxelSearchResponse {
  mode: "voxel";
  results: SearchResult[];
  stats: { nonAir: number; dims: [number, number, number] };
  tookMs: number;
}
