import * as THREE from "three";
import { AssetLoader } from "./AssetLoader";

/**
 * Stubbed SignRenderer for the web preview.
 *
 * The full schematic-renderer bakes vanilla-accurate sign geometry + glyph text
 * into the entity sign texture. The gallery preview doesn't render sign text, so
 * signs fall through to their (near-empty) block model. This keeps the engine
 * dependency-free of the entity sign atlas while every other block renders fully.
 */
export class SignRenderer {
	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	constructor(_assetLoader: AssetLoader) {}

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	public isSign(_blockId: string): boolean {
		return false;
	}

	public async buildSignMesh(
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		_blockId: string,
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		_properties: Record<string, string>,
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		_nbtData?: Record<string, unknown>,
	): Promise<THREE.Object3D | null> {
		return null;
	}
}
