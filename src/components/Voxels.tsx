"use client";

import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { GRID } from "@/lib/types";
import { nonAirVoxels } from "@/lib/voxel";
import { blockColorRGB } from "@/lib/blocks";

const BOX = new THREE.BoxGeometry(1, 1, 1);
const MAT = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0.0 });
const dummy = new THREE.Object3D();
const color = new THREE.Color();

/**
 * Instanced **flat-colour** render of all non-air voxels in a 32^3 grid, centred
 * at the origin. This is the lightweight editor path (compact block ids → hashed
 * palette colours, no textures), used by VoxelBuilder. For the textured, real-block
 * preview use {@link VoxelViewer}/PrismarineViewer instead.
 */
export function Voxels({ grid }: { grid: Uint8Array }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const voxels = useMemo(() => nonAirVoxels(grid), [grid]);
  const offset = GRID / 2 - 0.5;

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    voxels.forEach((v, i) => {
      dummy.position.set(v.x - offset, v.y - offset, v.z - offset);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      const [r, g, b] = blockColorRGB(v.b);
      mesh.setColorAt(i, color.setRGB(r, g, b));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [voxels, offset]);

  if (voxels.length === 0) return null;
  return (
    <instancedMesh
      ref={ref}
      args={[BOX, MAT, voxels.length]}
      castShadow
      receiveShadow
    />
  );
}
