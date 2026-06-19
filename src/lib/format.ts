export function compactNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1).replace(/\.0$/, "") + "k";
  return String(n);
}

/** Map a cosine score in [-1,1] to a 0..1 bar fraction (clamped). */
export function scoreFraction(score: number): number {
  return Math.max(0, Math.min(1, score));
}

export function scorePct(score: number): string {
  return (score * 100).toFixed(1) + "%";
}
