# 🧊 MC-Retrieval — Web Demo

Demo interaktif **Next.js + TypeScript** untuk cross-modal retrieval teks ↔ voxel Minecraft.
Dua mode pencarian:

- **🔍 Search by Text** — ketik deskripsi bebas ("medieval castle with towers"); query di-embed
  dengan `all-MiniLM-L6-v2` dan di-rank terhadap teks tiap build via **cosine similarity** (semantic,
  bukan keyword/kategori).
- **🧱 Search by Building** — susun struktur voxel 3D langsung di browser; build di-normalisasi
  (bbox-crop + resize 32³) lalu dicari build yang **mirip secara struktur**.

Penjelasan arsitektur lengkap + flowchart Mermaid ada di [`../README.md`](../README.md).

---

## Menjalankan

```bash
npm install
npm run dev
# buka http://localhost:3000
```

Pada kunjungan pertama, halaman memanggil `/api/warmup` yang mengunduh model embedding teks
(`Xenova/all-MiniLM-L6-v2`, ONNX ~25 MB) dari HuggingFace hub lalu meng-cache-nya di
`.transformers-cache/`. Indikator status model di header berubah `loading… → ready`. Setelah warm,
query teks berlangsung ~2–5 ms.

> Butuh akses jaringan **sekali** untuk mengunduh model. Setelah ter-cache, demo berjalan offline.

Production build:

```bash
npm run build && npm start
```

---

## Struktur

```
web/
├── scripts/export_gallery.py     # parquet → public/data/gallery.json
├── public/data/gallery.json      # subset galeri (voxel gzip+base64 + metadata)
└── src/
    ├── app/
    │   ├── page.tsx               # UI utama (mode tabs, search, builder, hasil)
    │   ├── layout.tsx, globals.css
    │   └── api/
    │       ├── search/text/route.ts   # Text → Build (cosine MiniLM)
    │       ├── search/voxel/route.ts  # Build → Build (cosine fitur struktural)
    │       └── warmup/route.ts        # pra-muat model + embedding galeri
    ├── components/
    │   ├── VoxelBuilder.tsx       # editor voxel 3D (react-three-fiber)
    │   ├── VoxelViewer.tsx        # preview 3D orbit (detail build)
    │   ├── Voxels.tsx             # instanced renderer
    │   ├── BuildCard.tsx, BuildDetail.tsx, AboutModal.tsx
    └── lib/
        ├── voxel.ts              # decode gzip, preprocessing, fitur (isomorphic)
        ├── blocks.ts             # palet warna per block ID (isomorphic)
        ├── text-embed.ts         # transformers.js (server-only)
        ├── gallery.ts            # index + cache embedding (server-only)
        ├── similarity.ts         # cosine + topK
        ├── format.ts, types.ts
```

---

## Re-generate galeri dari parquet

Galeri di-`public/data/gallery.json` sudah disertakan. Untuk membuat ulang / mengubah jumlah build:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "numpy<2" pandas pyarrow scipy
python scripts/export_gallery.py --n 240 --out public/data/gallery.json
```

Skrip mengambil subset **seimbang antar kategori** (prioritas build dengan engagement tinggi),
menjalankan preprocessing voxel identik dengan pipeline training (top-254 block mapping → bbox crop
→ NN-resize 32³), dan menyimpan tiap grid sebagai `base64(gzip(uint8[32³]))` (~0.4 KB/build).

---

## Catatan fidelity

Demo **tidak** memuat checkpoint terlatih (belum ada di repo). Sisi teks memakai backbone MiniLM
**asli**; sisi voxel memakai **deskriptor struktural handcrafted** sebagai pengganti VoxelEncoder
terlatih. UX, preprocessing, dan alur retrieval mengikuti sistem sebenarnya. Untuk meng-upgrade ke
embedding lintas-modal 256-d yang terlatih, sajikan `DualEncoder` lewat backend Python dan arahkan
ulang `text-embed.ts` / fitur voxel ke sana — frontend tidak perlu diubah. Lihat
[`../README.md` §7](../README.md).

---

## Stack

Next.js 15 · React 18 · TypeScript · Tailwind CSS v4 · three.js + @react-three/fiber + drei ·
@huggingface/transformers (transformers.js).
