"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { GRID } from "@/lib/types";
import { Voxels } from "./Voxels";

/** Orbitable 3D preview of a single voxel build. */
export function VoxelViewer({ grid, className }: { grid: Uint8Array; className?: string }) {
  return (
    <div className={className}>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [GRID * 0.9, GRID * 0.75, GRID * 0.9], fov: 42 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={["#0d0d16"]} />
        <ambientLight intensity={0.6} />
        <directionalLight
          position={[20, 40, 25]}
          intensity={1.4}
          castShadow
          shadow-mapSize={[1024, 1024]}
        />
        <directionalLight position={[-25, 15, -20]} intensity={0.4} color="#8b5cf6" />
        <Voxels grid={grid} />
        <Environment preset="city" />
        <OrbitControls
          autoRotate
          autoRotateSpeed={0.8}
          enablePan={false}
          minDistance={GRID * 0.6}
          maxDistance={GRID * 2.2}
        />
      </Canvas>
    </div>
  );
}
