import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MC-Retrieval — Cross-Modal Minecraft Schematic Search",
  description:
    "Demo for cross-modal retrieval between natural language and 3D Minecraft voxel schematics. Search by text or by building a voxel structure.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
