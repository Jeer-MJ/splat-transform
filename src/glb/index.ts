import { lstat, mkdir, readFile as fsReadFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import process, { exit } from 'node:process';
import { parseArgs } from 'node:util';
import { Vec3 } from 'playcanvas';

import { createDevice } from '../cli/node-device';
import { NodeFileSystem } from '../cli/node-file-system';
import {
    alignGridBounds,
    fillExterior,
    fillFloor,
    carve,
    type NavSeed
} from '../lib/voxel';
import { SparseVoxelGrid } from '../lib/voxel/sparse-voxel-grid';
import {
    cropToOccupied,
    cropToNavigable,
    writeOctreeFiles
} from '../lib/writers/write-voxel';
import { buildCollisionMesh } from '../lib/writers/collision-glb';
import { buildSparseOctree } from '../lib/writers/sparse-octree';
import { logWrittenFile } from '../lib/writers/utils';
import { GpuDilation } from '../lib/gpu';
import { logger, TextRenderer, fmtCount, fmtBytes, fmtTime } from '../lib/utils';
import type { CollisionMeshShape } from '../lib/types';

// ============================================================================
// Types & Interfaces
// ============================================================================

interface GLTFJson {
    scene?: number;
    scenes?: { nodes?: number[] }[];
    nodes?: {
        mesh?: number;
        translation?: number[];
        rotation?: number[];
        scale?: number[];
        matrix?: number[];
        children?: number[];
    }[];
    meshes?: {
        primitives: {
            attributes: { POSITION?: number };
            indices?: number;
            mode?: number;
        }[];
    }[];
    accessors?: {
        bufferView?: number;
        byteOffset?: number;
        componentType: number;
        count: number;
        type: string;
    }[];
    bufferViews?: {
        buffer: number;
        byteOffset?: number;
        byteLength: number;
        byteStride?: number;
    }[];
}

type Triangle = {
    a: Vec3;
    b: Vec3;
    c: Vec3;
};

// ============================================================================
// Matrix & Vector Math Helpers (Zero-dependency & High performance)
// ============================================================================

function transformPoint(m: Float32Array, x: number, y: number, z: number, out: Vec3): void {
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    const rw = w !== 0 ? 1 / w : 1;
    out.x = (m[0] * x + m[4] * y + m[8] * z + m[12]) * rw;
    out.y = (m[1] * x + m[5] * y + m[9] * z + m[13]) * rw;
    out.z = (m[2] * x + m[6] * y + m[10] * z + m[14]) * rw;
}

function getLocalMatrix(node: any): Float32Array {
    const m = new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
    if (node.matrix) {
        for (let i = 0; i < 16; i++) {
            m[i] = node.matrix[i];
        }
        return m;
    }

    const t = node.translation || [0, 0, 0];
    const r = node.rotation || [0, 0, 0, 1]; // quaternion [x, y, z, w]
    const s = node.scale || [1, 1, 1];

    const qx = r[0], qy = r[1], qz = r[2], qw = r[3];
    const sx = s[0], sy = s[1], sz = s[2];

    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;

    m[0] = (1 - 2 * (yy + zz)) * sx;
    m[1] = (2 * (xy + wz)) * sx;
    m[2] = (2 * (xz - wy)) * sx;
    m[3] = 0;

    m[4] = (2 * (xy - wz)) * sy;
    m[5] = (1 - 2 * (xx + zz)) * sy;
    m[6] = (2 * (yz + wx)) * sy;
    m[7] = 0;

    m[8] = (2 * (xz + wy)) * sz;
    m[9] = (2 * (yz - wx)) * sz;
    m[10] = (1 - 2 * (xx + yy)) * sz;
    m[11] = 0;

    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m[15] = 1;

    return m;
}

function multiplyMatrices(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let i = 0; i < 4; i++) {
                sum += a[row + i * 4] * b[i + col * 4];
            }
            out[row + col * 4] = sum;
        }
    }
    return out;
}

// ============================================================================
// GLB Binary Parsing Functions
// ============================================================================

function parseGlb(buffer: ArrayBuffer): { gltf: GLTFJson; binData: Uint8Array } {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    if (magic !== 0x46546C67) {
        throw new Error('Invalid GLB file: magic signature mismatch');
    }
    const version = view.getUint32(4, true);
    if (version !== 2) {
        throw new Error(`Unsupported GLB version: ${version}. Only version 2 is supported.`);
    }
    const length = view.getUint32(8, true);
    if (length !== buffer.byteLength) {
        throw new Error(`Invalid GLB file length: header states ${length} bytes, got ${buffer.byteLength}`);
    }

    let gltf: GLTFJson | null = null;
    let binData: Uint8Array | null = null;

    let offset = 12;
    while (offset < length) {
        if (offset + 8 > length) {
            break;
        }
        const chunkLength = view.getUint32(offset, true);
        const chunkType = view.getUint32(offset + 4, true);
        offset += 8;

        if (offset + chunkLength > length) {
            throw new Error(`GLB chunk extends beyond file boundary at offset ${offset}`);
        }

        const chunkData = new Uint8Array(buffer, offset, chunkLength);
        offset += chunkLength;

        if (chunkType === 0x4E4F534A) { // JSON
            const jsonText = new TextDecoder().decode(chunkData);
            gltf = JSON.parse(jsonText) as GLTFJson;
        } else if (chunkType === 0x004E4942) { // BIN
            binData = chunkData;
        }
    }

    if (!gltf) {
        throw new Error('GLB JSON chunk not found');
    }
    if (!binData) {
        throw new Error('GLB BIN chunk not found');
    }

    return { gltf, binData };
}

function readVec3Accessor(accessorIdx: number, gltf: GLTFJson, binData: Uint8Array): Float32Array {
    if (!gltf.accessors) throw new Error('GLB has no accessors');
    const accessor = gltf.accessors[accessorIdx];
    const count = accessor.count;
    const viewIdx = accessor.bufferView;
    const accessorByteOffset = accessor.byteOffset || 0;

    const out = new Float32Array(count * 3);

    if (viewIdx === undefined) {
        return out;
    }

    if (!gltf.bufferViews) throw new Error('GLB has no bufferViews');
    const view = gltf.bufferViews[viewIdx];
    const viewByteOffset = view.byteOffset || 0;
    const byteStride = view.byteStride || 12;

    const startOffset = viewByteOffset + accessorByteOffset;
    const dataView = new DataView(binData.buffer, binData.byteOffset + startOffset);

    for (let i = 0; i < count; i++) {
        const offset = i * byteStride;
        out[i * 3 + 0] = dataView.getFloat32(offset, true);
        out[i * 3 + 1] = dataView.getFloat32(offset + 4, true);
        out[i * 3 + 2] = dataView.getFloat32(offset + 8, true);
    }

    return out;
}

function readScalarAccessor(accessorIdx: number, gltf: GLTFJson, binData: Uint8Array): Uint32Array {
    if (!gltf.accessors) throw new Error('GLB has no accessors');
    const accessor = gltf.accessors[accessorIdx];
    const count = accessor.count;
    const viewIdx = accessor.bufferView;
    const accessorByteOffset = accessor.byteOffset || 0;

    const out = new Uint32Array(count);

    if (viewIdx === undefined) {
        return out;
    }

    if (!gltf.bufferViews) throw new Error('GLB has no bufferViews');
    const view = gltf.bufferViews[viewIdx];
    const viewByteOffset = view.byteOffset || 0;
    const compType = accessor.componentType;
    let byteStride = view.byteStride;

    let itemSize = 2; // default short
    if (compType === 5121) itemSize = 1;
    else if (compType === 5125) itemSize = 4;

    if (byteStride === undefined) {
        byteStride = itemSize;
    }

    const startOffset = viewByteOffset + accessorByteOffset;
    const dataView = new DataView(binData.buffer, binData.byteOffset + startOffset);

    for (let i = 0; i < count; i++) {
        const offset = i * byteStride;
        if (compType === 5121) {
            out[i] = dataView.getUint8(offset);
        } else if (compType === 5123) {
            out[i] = dataView.getUint16(offset, true);
        } else if (compType === 5125) {
            out[i] = dataView.getUint32(offset, true);
        }
    }

    return out;
}

// ============================================================================
// CLI Argument Parsing & Setup
// ============================================================================

const cliOptionsConfig = {
    overwrite: { type: 'boolean', short: 'w', default: false },
    help: { type: 'boolean', short: 'h', default: false },
    'voxel-params': { type: 'string', default: '0.05' },
    'voxel-external-fill': { type: 'string' },
    'voxel-floor-fill': { type: 'string' },
    'voxel-carve': { type: 'string' },
    'seed-pos': { type: 'string', default: '0,0,0' },
    'collision-mesh': { type: 'string', short: 'K' }
} as const;

const usage = `
Usage: splat-transform-glb [options] <input.glb> <output.voxel.json>

Options:
  -w, --overwrite                     Overwrite output files if they exist.
  -h, --help                          Show this help message.
  -p, --voxel-params <resolution>     Voxel resolution in world units. Default: 0.05
  --voxel-external-fill [radius]      Dilation radius for exterior fill (meters). Default: 1.6
  --voxel-floor-fill [radius]         Floor fill dilation radius (meters). Default: 1.6
  --voxel-carve <height,radius>        Carve a navigable height/radius capsule. Default: 1.6,0.2
  --seed-pos <x,y,z>                  Seed position in world space. Default: 0,0,0
  -K, --collision-mesh [smooth|faces] Collision mesh geometry generation mode. Default: smooth
`;

const fileExists = async (filename: string) => {
    try {
        await lstat(filename);
        return true;
    } catch (e: any) {
        if (e?.code === 'ENOENT') {
            return false;
        }
        throw e;
    }
};

const main = async () => {
    const startTime = performance.now();

    const peakMemoryBytes = (): number => {
        const raw = process.resourceUsage().maxRSS;
        return process.platform === 'win32' ? raw : raw * 1024;
    };

    const liveMemoryBytes = (): number => {
        const u = process.memoryUsage();
        return u.heapUsed + u.external;
    };

    const reportDone = (failed = false) => {
        const elapsedMs = performance.now() - startTime;
        const verb = failed ? 'failed in' : 'done in';
        const line = `${verb} ${fmtTime(elapsedMs)}  [peak ${fmtBytes(peakMemoryBytes())}]`;
        if (failed) {
            logger.error(line);
        } else {
            logger.info(line);
        }
    };

    const renderer = new TextRenderer({
        write: (chunk) => process.stderr.write(chunk),
        output: (chunk) => process.stdout.write(chunk),
        getPeakMemory: peakMemoryBytes,
        getLiveMemory: liveMemoryBytes
    });
    logger.setRenderer(renderer);

    const { values: v, positionals } = parseArgs({
        args: process.argv.slice(2),
        strict: true,
        allowPositionals: true,
        options: cliOptionsConfig
    });

    if (v.help || positionals.length < 2) {
        logger.output(usage.trim());
        exit(0);
    }

    const inputFilename = resolve(positionals[0]);
    const outputFilename = resolve(positionals[1]);

    if (!outputFilename.endsWith('.voxel.json')) {
        logger.error('Output filename must end with .voxel.json');
        exit(1);
    }

    if (!v.overwrite && await fileExists(outputFilename)) {
        logger.error(`File '${outputFilename}' already exists. Use -w option to overwrite.`);
        exit(1);
    }

    logger.info(`splat-transform GLB Voxelizer`);

    try {
        const voxelResolution = Number(v['voxel-params']);
        if (isNaN(voxelResolution) || voxelResolution <= 0) {
            throw new Error(`Invalid voxel resolution: ${v['voxel-params']}`);
        }

        const parseVec = (value: string, count: number): number[] => {
            const parts = value.split(',').map(p => Number(p));
            if (parts.length !== count || parts.some(isNaN)) {
                throw new Error(`Expected ${count} comma-separated numbers, got: ${value}`);
            }
            return parts;
        };

        let navExteriorRadius: number | undefined;
        if (v['voxel-external-fill'] !== undefined) {
            navExteriorRadius = v['voxel-external-fill'] ? Number(v['voxel-external-fill']) : 1.6;
        }

        let floorFill = false;
        let floorFillDilation = 0;
        if (v['voxel-floor-fill'] !== undefined) {
            floorFill = true;
            floorFillDilation = v['voxel-floor-fill'] ? Number(v['voxel-floor-fill']) : 1.6;
        }

        let navCapsule: { height: number; radius: number } | undefined;
        if (v['voxel-carve'] !== undefined) {
            if (v['voxel-carve']) {
                const [height, radius] = parseVec(v['voxel-carve'], 2);
                navCapsule = { height, radius };
            } else {
                navCapsule = { height: 1.6, radius: 0.2 };
            }
        }

        const [sx, sy, sz] = parseVec(v['seed-pos'], 3);
        const navSeed: NavSeed = { x: sx, y: sy, z: sz };

        const collisionMesh = v['collision-mesh'] !== undefined ?
            ((v['collision-mesh'] === '' ? 'smooth' : v['collision-mesh']) as CollisionMeshShape) :
            false;

        const hasNav = !!(navCapsule && navCapsule.height > 0);
        const hasFillExterior = !!(navExteriorRadius !== undefined);
        const hasFloorFill = floorFill;

        // 1. Read input GLB
        logger.info(`Loading GLB: ${basename(inputFilename)}`);
        const fileData = await fsReadFile(inputFilename);
        const { gltf, binData } = parseGlb(fileData.buffer);

        // 2. Traversal and transform matrix computation
        const nodesWorldMatrices = new Map<number, Float32Array>();
        const traverse = (nodeIdx: number, parentWorldMatrix: Float32Array) => {
            if (!gltf.nodes) return;
            const node = gltf.nodes[nodeIdx];
            const localMatrix = getLocalMatrix(node);
            const worldMatrix = multiplyMatrices(parentWorldMatrix, localMatrix);
            nodesWorldMatrices.set(nodeIdx, worldMatrix);

            if (node.children) {
                for (const childIdx of node.children) {
                    traverse(childIdx, worldMatrix);
                }
            }
        };

        const sceneIdx = gltf.scene !== undefined ? gltf.scene : 0;
        if (gltf.scenes && gltf.scenes[sceneIdx] && gltf.scenes[sceneIdx].nodes) {
            const nodes = gltf.scenes[sceneIdx].nodes!;
            const identity = new Float32Array([
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            ]);
            for (const rootNodeIdx of nodes) {
                traverse(rootNodeIdx, identity);
            }
        }

        // 3. Extract triangles in world space
        const triangles: Triangle[] = [];
        const minBounds = new Vec3(Infinity, Infinity, Infinity);
        const maxBounds = new Vec3(-Infinity, -Infinity, -Infinity);

        if (gltf.nodes && gltf.meshes) {
            for (let nodeIdx = 0; nodeIdx < gltf.nodes.length; nodeIdx++) {
                const node = gltf.nodes[nodeIdx];
                if (node.mesh === undefined) continue;

                const worldMatrix = nodesWorldMatrices.get(nodeIdx);
                if (!worldMatrix) continue; // Skip nodes not in active scene

                const mesh = gltf.meshes[node.mesh];
                for (const primitive of mesh.primitives) {
                    const mode = primitive.mode !== undefined ? primitive.mode : 4;
                    if (mode !== 4) continue; // Support TRIANGLES only

                    const posAccessorIdx = primitive.attributes.POSITION;
                    if (posAccessorIdx === undefined) continue;

                    const positions = readVec3Accessor(posAccessorIdx, gltf, binData);
                    const vertexCount = positions.length / 3;

                    let indices: Uint32Array;
                    if (primitive.indices === undefined) {
                        indices = new Uint32Array(vertexCount);
                        for (let i = 0; i < vertexCount; i++) {
                            indices[i] = i;
                        }
                    } else {
                        indices = readScalarAccessor(primitive.indices, gltf, binData);
                    }

                    for (let i = 0; i < indices.length; i += 3) {
                        const idx0 = indices[i];
                        const idx1 = indices[i + 1];
                        const idx2 = indices[i + 2];

                        const a = new Vec3();
                        const b = new Vec3();
                        const c = new Vec3();

                        transformPoint(worldMatrix, positions[idx0 * 3], positions[idx0 * 3 + 1], positions[idx0 * 3 + 2], a);
                        transformPoint(worldMatrix, positions[idx1 * 3], positions[idx1 * 3 + 1], positions[idx1 * 3 + 2], b);
                        transformPoint(worldMatrix, positions[idx2 * 3], positions[idx2 * 3 + 1], positions[idx2 * 3 + 2], c);

                        for (const p of [a, b, c]) {
                            if (p.x < minBounds.x) minBounds.x = p.x;
                            if (p.y < minBounds.y) minBounds.y = p.y;
                            if (p.z < minBounds.z) minBounds.z = p.z;
                            if (p.x > maxBounds.x) maxBounds.x = p.x;
                            if (p.y > maxBounds.y) maxBounds.y = p.y;
                            if (p.z > maxBounds.z) maxBounds.z = p.z;
                        }

                        triangles.push({ a, b, c });
                    }
                }
            }
        }

        logger.info(`Extracted ${fmtCount(triangles.length)} triangles from GLB`);

        if (triangles.length === 0) {
            throw new Error('No triangles found in GLB file');
        }

        // 4. Align grid bounds
        const exteriorPad = hasFillExterior ?
            (Math.ceil(navExteriorRadius! / voxelResolution) + 1) * voxelResolution :
            0;
        const floorPad = hasFloorFill ?
            (Math.ceil(floorFillDilation / voxelResolution) + 1) * voxelResolution :
            0;
        const padXZ = Math.max(exteriorPad, floorPad);
        const padY = exteriorPad;

        let gridBounds = alignGridBounds(
            minBounds.x - padXZ, minBounds.y - padY, minBounds.z - padXZ,
            maxBounds.x + padXZ, maxBounds.y + padY, maxBounds.z + padXZ,
            voxelResolution
        );

        const blockSize = 4 * voxelResolution;
        const nbx = Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize);
        const nby = Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize);
        const nbz = Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize);
        const nx = nbx << 2;
        const ny = nby << 2;
        const nz = nbz << 2;

        logger.info(`Voxel Grid size: ${nx}x${ny}x${nz} (${nbx}x${nby}x${nbz} blocks), voxelResolution: ${voxelResolution}`);

        // 5. Point sampling of triangles directly into SparseVoxelGrid
        const grid = new SparseVoxelGrid(nx, ny, nz);
        const S = voxelResolution / 2.5;

        const samplingBar = logger.bar('Voxelizing', triangles.length);
        for (let tIdx = 0; tIdx < triangles.length; tIdx++) {
            const { a, b, c } = triangles[tIdx];

            const abX = b.x - a.x, abY = b.y - a.y, abZ = b.z - a.z;
            const acX = c.x - a.x, acY = c.y - a.y, acZ = c.z - a.z;

            const L1 = Math.sqrt(abX * abX + abY * abY + abZ * abZ);
            const L2 = Math.sqrt(acX * acX + acY * acY + acZ * acZ);

            const n_u = Math.max(1, Math.ceil(L1 / S));
            for (let i = 0; i <= n_u; i++) {
                const u = i / n_u;
                const vMax = 1 - u;
                const n_v = Math.max(1, Math.ceil((vMax * L2) / S));

                const ux = u * abX;
                const uy = u * abY;
                const uz = u * abZ;

                for (let j = 0; j <= n_v; j++) {
                    const v = (j / n_v) * vMax;
                    const px = a.x + ux + v * acX;
                    const py = a.y + uy + v * acY;
                    const pz = a.z + uz + v * acZ;

                    const ix = Math.floor((px - gridBounds.min.x) / voxelResolution);
                    const iy = Math.floor((py - gridBounds.min.y) / voxelResolution);
                    const iz = Math.floor((pz - gridBounds.min.z) / voxelResolution);

                    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
                        grid.setVoxel(ix, iy, iz);
                    }
                }
            }
            samplingBar.tick();
        }
        samplingBar.end();

        // 6. WebGPU post-processing passes (dilation/flood fill/carving)
        let processedGrid = grid;
        let gpuDilation: GpuDilation | null = null;

        const needsGpuDilation = hasFillExterior || hasNav || (hasFloorFill && floorFillDilation > 0);
        if (needsGpuDilation) {
            logger.info('Initializing WebGPU device for voxel post-processing dilation...');
            const device = await createDevice();
            gpuDilation = new GpuDilation(device);
        }

        try {
            if (hasFillExterior) {
                const sub = logger.group('Fill exterior');
                const fillResult = await fillExterior(
                    processedGrid, gridBounds, voxelResolution,
                    navExteriorRadius!, navSeed,
                    gpuDilation!
                );
                processedGrid = fillResult.grid;
                gridBounds = fillResult.gridBounds;
                sub.end();
            }

            if (hasFloorFill) {
                const sub = logger.group('Fill floor');
                const floorResult = await fillFloor(
                    processedGrid, gridBounds, voxelResolution, floorFillDilation, gpuDilation
                );
                processedGrid = floorResult.grid;
                gridBounds = floorResult.gridBounds;
                sub.end();
            }

            if (hasNav) {
                const sub = logger.group('Carve');
                const navResult = await carve(
                    processedGrid, gridBounds, voxelResolution,
                    navCapsule!.height, navCapsule!.radius,
                    navSeed,
                    gpuDilation!
                );
                processedGrid = navResult.grid;
                gridBounds = navResult.gridBounds;
                sub.end();
            }
        } finally {
            if (gpuDilation) {
                gpuDilation.destroy();
            }
        }

        const cropSub = logger.group('Cropping');
        const finalCrop = hasFillExterior || hasFloorFill ?
            cropToNavigable(processedGrid, gridBounds, voxelResolution) :
            cropToOccupied(processedGrid, gridBounds, voxelResolution);
        processedGrid = finalCrop.grid;
        gridBounds = finalCrop.gridBounds;
        cropSub.end();

        // 7. Write assets
        const glbBytes = collisionMesh ?
            buildCollisionMesh(processedGrid, gridBounds, voxelResolution, collisionMesh) :
            null;

        const octree = buildSparseOctree(
            processedGrid,
            gridBounds,
            { min: minBounds, max: maxBounds },
            voxelResolution,
            { consumeGrid: true }
        );

        logger.info(`octree depth: ${octree.treeDepth}`);
        logger.info(`interior nodes: ${fmtCount(octree.numInteriorNodes)}`);
        logger.info(`mixed leaves: ${fmtCount(octree.numMixedLeaves)}`);

        const writingSub = logger.group('Writing');
        const fs = new NodeFileSystem();
        await writeOctreeFiles(fs, outputFilename, octree);

        if (glbBytes) {
            const glbFilename = outputFilename.replace('.voxel.json', '.collision.glb');
            await fs.createWriter(glbFilename).then(async (writer) => {
                await writer.write(glbBytes);
                await writer.close();
            });
            logWrittenFile(basename(glbFilename), glbBytes.length);
        }
        writingSub.end();

    } catch (err) {
        logger.error(err);
        reportDone(true);
        exit(1);
    }

    reportDone();
    exit(0);
};

export { main };
