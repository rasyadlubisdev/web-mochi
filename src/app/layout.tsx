import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MOCHI: Minecraft Object Comprehension & Hybrid Indexing",
  description:
    "Demo for cross-modal retrieval between natural language and 3D Minecraft voxel schematics. Search by text, image, or by uploading a schematic.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
