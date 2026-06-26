import * as THREE from "three";

/**
 * Stubbed EntityRenderer for the web preview.
 *
 * The full schematic-renderer ships a 10 MB `models.json` of base64 GLTF entity
 * models (chests, shulker boxes, …) loaded via GLTFLoader. For the gallery
 * preview we don't bundle that — block-entities (chests/shulkers) simply fall
 * through to their regular block-model rendering, or render nothing if they are
 * pure entities. This keeps the client bundle small while every normal block
 * still renders with full Cubane geometry + textures.
 */
export class EntityRenderer {
	private debug = false;

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	public async createEntityMesh(_entityName: string): Promise<THREE.Object3D | null> {
		return null;
	}

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	public async preloadModels(_entityNames: string[]): Promise<void> {
		/* no-op: entity models are not bundled in the web preview */
	}

	public setDebug(debug: boolean): void {
		this.debug = debug;
	}
}
