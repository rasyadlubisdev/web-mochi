"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { buildVoxelMesh, fetchVoxelGrid, loadAtlas, loadAtlasTexture, type MeshResult } from "@/lib/blockAtlas";

/**
 * Live, textured 3D preview of a build, meshed from its raw Minecraft block-state
 * grid (real block IDs from data.parquet) with the prebuilt texture atlas. The
 * camera is framed on the build's true bounding box, so a flat map stays flat and
 * a tall tower stays tall — never squished to fit the container.
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
  const [data, setData] = useState<{ tex: THREE.Texture; mesh: MeshResult } | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    setData(null);
    setFailed(false);
    (async () => {
      try {
        const [atlas, tex] = await Promise.all([loadAtlas(), loadAtlasTexture()]);
        const g = grid ?? (id ? await fetchVoxelGrid(id) : null);
        if (!g) throw new Error("no voxel source");
        if (!alive) return;
        setData({ tex, mesh: buildVoxelMesh(g, atlas) });
      } catch {
        if (alive) setFailed(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, [id, grid]);

  if (failed) return <div className={className}>{fallback ?? null}</div>;
  if (!data) return <div className={`${className ?? ""} skeleton`} />;

  return (
    <div className={className}>
      <Canvas dpr={[1, 2]} gl={{ antialias: true }} camera={{ fov: 45, near: 0.1, far: 2000 }}>
        <color attach="background" args={["#0d0d16"]} />
        <ambientLight intensity={0.75} />
        <directionalLight position={[40, 70, 30]} intensity={1.25} />
        <directionalLight position={[-30, 25, -35]} intensity={0.35} color="#8b5cf6" />
        <Build mesh={data.mesh} tex={data.tex} />
      </Canvas>
    </div>
  );
}

function Build({ mesh, tex }: { mesh: MeshResult; tex: THREE.Texture }) {
  const { camera, size } = useThree();
  const controls = useRef<React.ComponentRef<typeof OrbitControls>>(null);

  const texturedMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        map: tex,
        vertexColors: true,
        roughness: 0.95,
        metalness: 0,
        alphaTest: 0.5,
      }),
    [tex],
  );
  const translucentMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        map: tex,
        vertexColors: true,
        roughness: 0.25,
        metalness: 0,
        transparent: true,
        alphaTest: 0.05,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    [tex],
  );
  const flatMat = useMemo(
    () => new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0 }),
    [],
  );

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
      {mesh.textured && <mesh geometry={mesh.textured} material={texturedMat} />}
      {mesh.flat && <mesh geometry={mesh.flat} material={flatMat} />}
      {mesh.translucent && <mesh geometry={mesh.translucent} material={translucentMat} renderOrder={1} />}
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
