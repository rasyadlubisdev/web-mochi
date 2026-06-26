"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { fetchVoxelGrid } from "@/lib/blockAtlas";
import { buildGridMesh, type GridMesh } from "@/lib/cubaneRenderer";

/**
 * Live, textured 3D preview of a build, rendered by the real schematic-renderer
 * engine (Cubane) from its raw Minecraft block-state grid + the vanilla resource
 * pack. Every block — including small ones (levers, torches, buttons, rails,
 * stairs, fences) — renders with its true model shape, position and per-face
 * textures. The camera is framed on the build's true bounding box, so a flat map
 * stays flat and a tall tower stays tall. Model only: no bounding box, no ground.
 */
export function PrismarineViewer({
  id,
  grid,
  className,
  fallback,
}: {
  /** render a gallery build by id (fetches its raw grid) … */
  id?: string;
  /** … or render a raw grid directly (e.g. an uploaded schematic). */
  grid?: Uint16Array;
  className?: string;
  fallback?: ReactNode;
}) {
  const [data, setData] = useState<GridMesh | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    setData(null);
    setFailed(false);
    (async () => {
      try {
        const g = grid ?? (id ? await fetchVoxelGrid(id) : null);
        if (!g) throw new Error("no voxel source");
        const mesh = await buildGridMesh(g);
        if (!alive) {
          disposeGroupGeometries(mesh.group);
          return;
        }
        setData(mesh);
      } catch {
        if (alive) setFailed(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, [id, grid]);

  // Dispose merged geometries when the rendered mesh changes/unmounts. Materials
  // and the atlas texture are shared across builds (MaterialRegistry), so we never
  // dispose those here.
  useEffect(() => {
    return () => {
      if (data) disposeGroupGeometries(data.group);
    };
  }, [data]);

  if (failed) return <div className={className}>{fallback ?? null}</div>;
  if (!data) return <div className={`${className ?? ""} skeleton`} />;

  return (
    <div className={className}>
      <Canvas dpr={[1, 2]} gl={{ antialias: true }} camera={{ fov: 45, near: 0.1, far: 2000 }}>
        <color attach="background" args={["#0d0d16"]} />
        <ambientLight intensity={0.75} />
        <directionalLight position={[40, 70, 30]} intensity={1.25} />
        <directionalLight position={[-30, 25, -35]} intensity={0.35} color="#8b5cf6" />
        <Build mesh={data} />
      </Canvas>
    </div>
  );
}

function disposeGroupGeometries(group: THREE.Group) {
  group.traverse((o) => {
    if (o instanceof THREE.Mesh) o.geometry?.dispose();
  });
}

function Build({ mesh }: { mesh: GridMesh }) {
  const { camera, size } = useThree();
  const controls = useRef<React.ComponentRef<typeof OrbitControls>>(null);

  // Frame the camera on the build's bounding sphere (true proportions, no squish).
  useEffect(() => {
    const radius = 0.5 * mesh.size.length() || 8;
    const fov = ((camera as THREE.PerspectiveCamera).fov * Math.PI) / 180;
    const aspect = Math.max(0.1, size.width / size.height);
    const fitV = radius / Math.tan(fov / 2);
    const fitH = radius / (Math.tan(fov / 2) * aspect);
    const dist = 1.3 * Math.max(fitV, fitH);

    const az = Math.PI / 4;
    const el = Math.PI / 6;
    camera.position.set(
      dist * Math.cos(el) * Math.cos(az),
      dist * Math.sin(el),
      dist * Math.cos(el) * Math.sin(az),
    );
    camera.lookAt(0, 0, 0);
    if (controls.current) {
      controls.current.target.set(0, 0, 0);
      controls.current.update();
    }
  }, [mesh, camera, size.width, size.height]);

  return (
    <>
      <primitive object={mesh.group} />
      <OrbitControls
        ref={controls}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.8}
        minDistance={2}
        maxDistance={400}
      />
    </>
  );
}
