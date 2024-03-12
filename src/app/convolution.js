import convolutionWGSL from './shader/convolution.wgsl?raw';
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

export function initConvolution(device, tex) {
    _device = device;
    inTexture = tex;
    const module = _device.createShaderModule({ code: convolutionWGSL });
    const defs = wgh.makeShaderDataDefinitions(convolutionWGSL);
    const pipelineDesc = {
        compute: {
            module,
            entryPoint: 'compute_main',
        }
    };
    const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, pipelineDesc);
    descriptors[1].entries.push({
        binding: 2,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' },
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

    createBindGroups([1, 1]);
}

function createBindGroups(viewportSize) { 
    const w = inTexture.width;
    const h = inTexture.height;

    blurTextures = blurTextures.map(() => {
        const texture = _device.createTexture({
            size: { width: w, height: h },
            format: 'rgba8unorm',
            usage: 
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        _device.queue.writeTexture({ texture }, new Uint8Array(new Array(w * h * 4).fill(255)), { bytesPerRow: w * 4 }, { width: w, height: h });

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

export function getConvolutionResultTexture() {
    return blurTextures[1];
}

export function resizeConvolution(viewportSize) {
    createBindGroups(viewportSize);
}

export function addConvolutionCommands(cmdEncoder) {
    const pass = cmdEncoder.beginComputePass();

    const dispatches = [
        Math.ceil(inTexture.width / dispatchSize[0]),
        Math.ceil(inTexture.height / dispatchSize[1])
    ];

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroupParams);

    pass.setBindGroup(1, bindGroup0);
    pass.dispatchWorkgroups(...dispatches);

    pass.setBindGroup(1, bindGroup1);
    pass.dispatchWorkgroups(...dispatches);
  
    pass.end();
}