import { Float16Array } from '@petamoriken/float16';
import reactionDiffusionWGSL from './shader/reaction-diffusion.wgsl?raw';
import * as wgh from 'webgpu-utils';

let _device, pipeline, bindGroupParams, bindGroup0, bindGroup1, paramsUniforms, blurTextures = [null, null];
let bindGroupParamsLayout, bindGroup0Layout;
let inTexture;

// shader constants
const kernelSize = 3;
const workgroupSize = [8, 8];
// each thread handles a tile of pixels
const tileSize = [3, 3];
// holds all the pixels needed for one workgroup
const cacheSize = [
    tileSize[0] * workgroupSize[0],
    tileSize[1] * workgroupSize[1]
];
// the cache has to include the boundary pixels needed for a 
// valid evaluation of the kernel within the dispatch area
const dispatchSize = [
    cacheSize[0] - Math.max(0, (kernelSize - 1)),
    cacheSize[1] - Math.max(0, (kernelSize - 1)),
];

export function initReactionDiffusion(device, tex) {
    _device = device;
    const module = _device.createShaderModule({ code: reactionDiffusionWGSL });
    const defs = wgh.makeShaderDataDefinitions(reactionDiffusionWGSL);
    const pipelineDesc = {
        compute: {
            module,
            entryPoint: 'compute_main',
        }
    };
    const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, pipelineDesc);
    descriptors[1].entries.push({
        binding: 2,
        storageTexture: { access: 'write-only', format: 'rgba16float' },
        visibility: GPUShaderStage.COMPUTE
    });
    bindGroupParamsLayout = _device.createBindGroupLayout(descriptors[0]);
    bindGroup0Layout = _device.createBindGroupLayout(descriptors[1]);
    const layout = _device.createPipelineLayout({
        bindGroupLayouts: [bindGroupParamsLayout, bindGroup0Layout]
    });
    pipeline = _device.createComputePipeline({
        layout,
        ...pipelineDesc
    });

    paramsUniforms = wgh.makeStructuredView(defs.uniforms.params);
    paramsUniforms.set({
        kernelSize,
        dispatchSize
    });
    paramsUniforms.gpuBuffer = _device.createBuffer({
        size: paramsUniforms.arrayBuffer.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _device.queue.writeBuffer(paramsUniforms.gpuBuffer, 0, paramsUniforms.arrayBuffer);
    bindGroupParams = _device.createBindGroup({
        layout: bindGroupParamsLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsUniforms.gpuBuffer } }
        ]
    });

    createBindGroups([1, 1], tex);
}

function createBindGroups(viewportSize, tex) { 
    inTexture = tex;
    const w = Math.round(inTexture.width / 10);
    const h = Math.round(inTexture.height / 10);

    blurTextures = blurTextures.map((v, ndx) => {
        const texture = _device.createTexture({
            size: { width: w, height: h },
            format: 'rgba16float',
            usage: 
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        let dataf16;
        if (ndx === 0) {
            const rgba = new Array(w * h * 4).fill(0);
            const s = 20;
            const bx = [w / 2 - s, w / 2 + s];
            const by = [h / 2 - s, h / 2 + s];
            for(let x=0; x<w; x++) {
                for(let y=0; y<h; y++) {
                    const v = x > bx[0] && x < bx[1] && y > by[0] && y < by[1];
                    rgba[(x + y * w) * 4 + 0] = v ? 0 : 1;
                    rgba[(x + y * w) * 4 + 1] = v ? 1 : 0;
                    rgba[(x + y * w) * 4 + 2] = 1;
                    rgba[(x + y * w) * 4 + 3] = 1;
                }
            }
            dataf16 = new Float16Array(rgba);
        } else {
            dataf16 = new Float16Array(new Array(w * h * 4).fill(0));
        }
        
        _device.queue.writeTexture({ texture }, dataf16.buffer, { bytesPerRow: w * 8 }, { width: w, height: h });

        return texture;
    });

    bindGroup0 = _device.createBindGroup({
        layout: bindGroup0Layout,
        entries: [
            { binding: 0, resource: inTexture.createView() },
            { binding: 1, resource: blurTextures[0].createView() },
            { binding: 2, resource: blurTextures[1].createView() },
        ]
    });
    bindGroup1 = _device.createBindGroup({
        layout: bindGroup0Layout,
        entries: [
            { binding: 0, resource: inTexture.createView() },
            { binding: 1, resource: blurTextures[1].createView() },
            { binding: 2, resource: blurTextures[0].createView() },
        ]
    });
}

export function getReactionDiffusionResultTexture() {
    return blurTextures[1];
}

export function resizeReactionDiffusion(viewportSize, tex) {
    createBindGroups(viewportSize, tex);
}

export function addReactionDiffusionCommands(cmdEncoder) {
    const passEncoder = cmdEncoder.beginComputePass();

    const dispatches = [
        Math.ceil(inTexture.width / dispatchSize[0]),
        Math.ceil(inTexture.height / dispatchSize[1])
    ];

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroupParams);

    for(let i = 0; i < 10; i++) {
        passEncoder.setBindGroup(1, bindGroup0);
        passEncoder.dispatchWorkgroups(...dispatches);
    
        passEncoder.setBindGroup(1, bindGroup1);
        passEncoder.dispatchWorkgroups(...dispatches);
    }
  
    passEncoder.end();
}