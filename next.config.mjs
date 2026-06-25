/** @type {import('next').NextConfig} */
const nextConfig = {
  // @huggingface/transformers pulls in onnxruntime-node / sharp — keep them as
  // real Node externals instead of bundling them into the server build.
  serverExternalPackages: [
    "@huggingface/transformers",
    "prismarine-schematic",
    "prismarine-nbt",
    "prismarine-block",
    "prismarine-world",
    "prismarine-registry",
    "prismarine-chunk",
    "minecraft-data",
    "vec3",
  ],
};

export default nextConfig;
