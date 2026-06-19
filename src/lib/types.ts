// Shared types for the MC-Retrieval web demo.

export const GRID = 32;

/** A schematic as stored in public/data/gallery.json (voxels base64-gzipped). */
export interface GalleryItemRaw {
  id: string;
  title: string;
  category: string;
  description: string;
  tags: string[];
  text: string;
  user: string;
  url: string | null;
  img: string | null;
  diamonds: number;
  views: number;
  downloads: number;
  year: number | null;
  dims: [number, number, number]; // original cropped extents (pre-resize)
  fill: number;
  voxels: string; // base64( gzip( uint8[32^3] ) )
}

export interface GalleryFile {
  meta: {
    grid: number;
    count: number;
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

export interface SearchResult extends BuildSummary {
  score: number; // cosine similarity in [-1, 1]
  voxels: string; // base64-gzipped grid so the detail view can render it
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
