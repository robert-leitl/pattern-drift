import convolutionWGSL from './shader/convolution.wgsl?raw';
import * as wgh from 'webgpu-utils';

let _device, pipeline, bindGroupParams, bindGroupHInit, bindGroupH, bindGroupV, paramsUniforms, flipUniform, size, blurTextures = [null, null];
let bindGroupParamsLayout, bindGroupHLayout;
let inTexture;

// shader constants
const kernelSize = 3;
const workgroupSize = [8, 8];
// each thread handles a tile of pixels
const tileSize = [4, 4];
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
        binding: 1,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' },
        visibility: GPUShaderStage.COMPUTE
    });
    bindGroupParamsLayout = _device.createBindGroupLayout(descriptors[0]);
    bindGroupHLayout = _device.createBindGroupLayout(descriptors[1]);
    const layout = _device.createPipelineLayout({
        bindGroupLayouts: [bindGroupParamsLayout, bindGroupHLayout]
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
    blurTextures = blurTextures.map(() => 
    _device.createTexture({
            size: { width: inTexture.width, height: inTexture.height },
            format: 'rgba8unorm',
            usage: 
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        })
    );
    bindGroupHInit = _device.createBindGroup({
        layout: bindGroupHLayout,
        entries: [
            { binding: 0, resource: inTexture.createView() },
            { binding: 1, resource: blurTextures[1].createView() },
        ]
    });
    bindGroupH = _device.createBindGroup({
        layout: bindGroupHLayout,
        entries: [
            { binding: 0, resource: blurTextures[0].createView() },
            { binding: 1, resource: blurTextures[1].createView() },
        ]
    });
    bindGroupV = _device.createBindGroup({
        layout: bindGroupHLayout,
        entries: [
            { binding: 0, resource: blurTextures[1].createView() },
            { binding: 1, resource: blurTextures[0].createView() },
        ]
    });
}

export function getConvolutionResultTexture() {
    return blurTextures[1];
}

export function resizeConvolution(viewportSize) {
    size = [...viewportSize];
    createBindGroups(viewportSize);
}

export function addConvolutionCommands(cmdEncoder) {
    const pass = cmdEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroupParams);

    pass.setBindGroup(1, bindGroupHInit);
    pass.dispatchWorkgroups(
        Math.ceil(inTexture.width / dispatchSize[0]),
        Math.ceil(inTexture.height / dispatchSize[1])
    );
  
    pass.end();
}