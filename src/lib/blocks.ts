// Deterministic colour palette for compact block IDs.
//
// The raw dataset uses legacy numeric Minecraft block IDs (no name mapping is
// shipped), so we cannot recover true material colours. Instead we assign each
// compact block ID (0=air, 1=rare, 2..=frequent) a stable, visually distinct
// colour. The most frequent IDs get hand-picked Minecraft-ish tones; the rest
// fall back to an HSL hash so every build renders with a consistent palette.

const BASE: Record<number, string> = {
  2: "#8d8d8d", // stone-ish grey (usually the most common block)
  3: "#8a5a2b", // dirt / wood brown
  4: "#5a8f3c", // grass green
  5: "#9c6b3f", // planks
  6: "#7a7a7a", // cobble
  7: "#caa472", // sandstone
  8: "#b9b9b9", // light grey
  9: "#3b6fb5", // blue / water
  10: "#c44b3a", // brick red
  11: "#d8c178", // sand
  12: "#43474d", // dark stone
  13: "#6f9b4a", // leaves
  14: "#a8732f", // log
  15: "#d4d4d4", // quartz / white
};

const cache = new Map<number, string>();

function hslHash(id: number): string {
  // golden-angle hue spread → well-separated hues for sequential IDs
  const hue = (id * 137.508) % 360;
  const sat = 45 + (id % 5) * 8; // 45–77%
  const light = 42 + ((id * 7) % 5) * 6; // 42–66%
  return `hsl(${hue.toFixed(0)} ${sat}% ${light}%)`;
}

/** CSS colour string for a compact block ID. */
export function blockColor(id: number): string {
  if (id <= 0) return "#000000"; // air (never rendered)
  if (BASE[id]) return BASE[id];
  const hit = cache.get(id);
  if (hit) return hit;
  const c = hslHash(id);
  cache.set(id, c);
  return c;
}

/** Same colour as an [r,g,b] tuple in 0..1 (for three.js instanced colours). */
export function blockColorRGB(id: number): [number, number, number] {
  const css = blockColor(id);
  if (css.startsWith("#")) {
    const n = parseInt(css.slice(1), 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }
  // hsl(h s% l%)
  const m = css.match(/hsl\(([\d.]+) ([\d.]+)% ([\d.]+)%\)/);
  if (!m) return [0.6, 0.6, 0.6];
  return hslToRgb(+m[1] / 360, +m[2] / 100, +m[3] / 100);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hk = (t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  return [hk(h + 1 / 3), hk(h), hk(h - 1 / 3)];
}
