"use client";

import { useRef } from "react";
import { Canvas, ThreeEvent } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";
import { blockColor } from "@/lib/blocks";

export const BUILD_SIZE = 16; // editable grid is 16^3; server normalises scale

export type VoxelMap = Record<string, number>; // "x,y,z" -> blockId

const OFF = BUILD_SIZE / 2 - 0.5;
const key = (x: number, y: number, z: number) => `${x},${y},${z}`;
const inBounds = (n: number) => n >= 0 && n < BUILD_SIZE;

function world(x: number, y: number, z: number): [number, number, number] {
  return [x - OFF, y + 0.5, z - OFF];
}

interface BuilderProps {
  voxels: VoxelMap;
  setVoxels: (next: VoxelMap) => void;
  selectedBlock: number;
  erase: boolean;
}

function Scene({ voxels, setVoxels, selectedBlock, erase }: BuilderProps) {
  const down = useRef<[number, number]>([0, 0]);

  const isClick = (e: ThreeEvent<MouseEvent>) => {
    const [dx, dy] = down.current;
    return Math.hypot(e.nativeEvent.clientX - dx, e.nativeEvent.clientY - dy) < 6;
  };

  const place = (x: number, y: number, z: number) => {
    if (!inBounds(x) || !inBounds(y) || !inBounds(z)) return;
    setVoxels({ ...voxels, [key(x, y, z)]: selectedBlock });
  };
  const remove = (x: number, y: number, z: number) => {
    const next = { ...voxels };
    delete next[key(x, y, z)];
    setVoxels(next);
  };

  const onGround = (e: ThreeEvent<MouseEvent>) => {
    if (!isClick(e) || erase) return;
    e.stopPropagation();
    const gx = Math.round(e.point.x + OFF);
    const gz = Math.round(e.point.z + OFF);
    place(gx, 0, gz);
  };

  const onVoxel = (gx: number, gy: number, gz: number) => (e: ThreeEvent<MouseEvent>) => {
    if (!isClick(e)) return;
    e.stopPropagation();
    if (erase) {
      remove(gx, gy, gz);
      return;
    }
    const n = e.face?.normal ?? new THREE.Vector3(0, 1, 0);
    place(gx + Math.round(n.x), gy + Math.round(n.y), gz + Math.round(n.z));
  };

  const entries = Object.entries(voxels);

  return (
    <>
      <color attach="background" args={["#0d0d16"]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[18, 35, 22]} intensity={1.3} castShadow shadow-mapSize={[1024, 1024]} />
      <directionalLight position={[-22, 14, -18]} intensity={0.4} color="#8b5cf6" />

      {/* ground build plate */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        receiveShadow
        onPointerDown={(e) => (down.current = [e.nativeEvent.clientX, e.nativeEvent.clientY])}
        onClick={onGround}
      >
        <planeGeometry args={[BUILD_SIZE, BUILD_SIZE]} />
        <meshStandardMaterial color="#181826" transparent opacity={0.85} />
      </mesh>
      <gridHelper args={[BUILD_SIZE, BUILD_SIZE, "#3a3a52", "#23233a"]} position={[0, 0.01, 0]} />

      {/* placed blocks */}
      {entries.map(([k, b]) => {
        const [x, y, z] = k.split(",").map(Number);
        return (
          <mesh
            key={k}
            position={world(x, y, z)}
            castShadow
            receiveShadow
            onPointerDown={(e) => (down.current = [e.nativeEvent.clientX, e.nativeEvent.clientY])}
            onClick={onVoxel(x, y, z)}
          >
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial color={blockColor(b)} roughness={0.85} />
          </mesh>
        );
      })}

      <OrbitControls makeDefault enablePan={false} minDistance={8} maxDistance={48} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={["#ef4444", "#10b981", "#3b82f6"]} labelColor="white" />
      </GizmoHelper>
    </>
  );
}

export function VoxelBuilder(props: BuilderProps & { className?: string }) {
  const { className, ...scene } = props;
  return (
    <div className={className}>
      <Canvas shadows dpr={[1, 2]} camera={{ position: [BUILD_SIZE * 1.3, BUILD_SIZE * 1.1, BUILD_SIZE * 1.3], fov: 45 }}>
        <Scene {...scene} />
      </Canvas>
    </div>
  );
}

// --- preset builds for quick demos ------------------------------------------

export const PRESETS: Record<string, () => VoxelMap> = {
  Tower: () => {
    const v: VoxelMap = {};
    for (let y = 0; y < 14; y++)
      for (let x = 5; x <= 9; x++)
        for (let z = 5; z <= 9; z++)
          if (x === 5 || x === 9 || z === 5 || z === 9) v[`${x},${y},${z}`] = 2;
    // battlements
    for (let x = 5; x <= 9; x += 2) for (let z = 5; z <= 9; z += 2) v[`${x},14,${z}`] = 2;
    return v;
  },
  House: () => {
    const v: VoxelMap = {};
    for (let x = 4; x <= 11; x++)
      for (let z = 4; z <= 10; z++) {
        v[`${x},0,${z}`] = 5; // floor (planks)
        for (let y = 1; y <= 4; y++)
          if (x === 4 || x === 11 || z === 4 || z === 10) v[`${x},${y},${z}`] = 14; // log walls
      }
    // pitched roof
    for (let r = 0; r <= 3; r++)
      for (let x = 4 + r; x <= 11 - r; x++)
        for (let z = 4; z <= 10; z++) v[`${x},${5 + r},${z}`] = 10; // brick roof
    return v;
  },
  Pyramid: () => {
    const v: VoxelMap = {};
    for (let y = 0; y < 8; y++)
      for (let x = y; x < 16 - y; x++)
        for (let z = y; z < 16 - y; z++) v[`${x},${y},${z}`] = 11; // sand
    return v;
  },
  Tree: () => {
    const v: VoxelMap = {};
    for (let y = 0; y < 7; y++) v[`8,${y},8`] = 14; // trunk
    for (let y = 5; y <= 8; y++)
      for (let x = 6; x <= 10; x++)
        for (let z = 6; z <= 10; z++) {
          const r = Math.abs(x - 8) + Math.abs(z - 8) + Math.abs(y - 7);
          if (r <= 3 && !(x === 8 && z === 8 && y < 7)) v[`${x},${y},${z}`] = 13; // leaves
        }
    return v;
  },
};
