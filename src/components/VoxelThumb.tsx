"use client";

import { useEffect, useRef, useState } from "react";
import { getThumbnail } from "@/lib/thumbnailRenderer";

/**
 * Textured build thumbnail. Renders lazily (once scrolled into view) via the
 * shared offscreen Three.js renderer, so the card preview matches the detail
 * view — real Minecraft textures and true proportions, shown with object-contain
 * so it's never squished to the card box.
 */
export function VoxelThumb({ id, alt, className }: { id: string; alt?: string; className?: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = wrap.current;
    if (!el || visible) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: "300px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    let alive = true;
    getThumbnail(id)
      .then((u) => alive && setSrc(u))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [visible, id]);

  return (
    <div ref={wrap} className={className}>
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt={alt ?? ""} className="h-full w-full object-contain" />
      ) : failed ? (
        <div className="grid h-full w-full place-items-center text-3xl">🧊</div>
      ) : (
        <div className="skeleton h-full w-full" />
      )}
    </div>
  );
}
