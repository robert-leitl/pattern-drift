import blurWGSL from './shader/blur.wgsl?raw';
import * as wgh from 'webgpu-utils';

let _device, pipeline, bindGroupParams, bindGroupHInit, bindGroupH, bindGroupV, paramsUniforms, flipUniform, size, blurTextures = [null, null];
let bindGroupParamsLayout, bindGroupHLayout;
let inTexture;

// shader constants
const tileDim = 128;
const batch = [4, 4];

const settings = {
    filterSize: 3,
    iterations: 1,
};

let blockDim = tileDim - Math.max(0, (settings.filterSize - 1));

export function initBlur(device, tex) {
    _device = device;
    inTexture = tex;
    const module = _device.createShaderModule({ code: blurWGSL });
    const defs = wgh.makeShaderDataDefinitions(blurWGSL);
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
    paramsUniforms.gpuBuffer = _device.createBuffer({
        size: paramsUniforms.arrayBuffer.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _device.queue.writeBuffer(paramsUniforms.gpuBuffer, 0, new Uint32Array([settings.filterSize, blockDim]))
    flipUniform = wgh.makeStructuredView(defs.uniforms.flip);
    flipUniform.gpuBufferH = _device.createBuffer({
        size: flipUniform.arrayBuffer.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _device.queue.writeBuffer(flipUniform.gpuBufferH, 0, new Uint32Array([0]));
    flipUniform.gpuBufferV = _device.createBuffer({
        size: flipUniform.arrayBuffer.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _device.queue.writeBuffer(flipUniform.gpuBufferV, 0, new Uint32Array([1]));
    const sampler = _device.createSampler({
        minFilter: 'nearest',
        magFilter: 'nearest'
    });
    bindGroupParams = _device.createBindGroup({
        layout: bindGroupParamsLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsUniforms.gpuBuffer } },
            { binding: 1, resource: sampler },
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
            { binding: 2, resource: { buffer: flipUniform.gpuBufferH } },
        ]
    });
    bindGroupH = _device.createBindGroup({
        layout: bindGroupHLayout,
        entries: [
            { binding: 0, resource: blurTextures[0].createView() },
            { binding: 1, resource: blurTextures[1].createView() },
            { binding: 2, resource: { buffer: flipUniform.gpuBufferH } },
        ]
    });
    bindGroupV = _device.createBindGroup({
        layout: bindGroupHLayout,
        entries: [
            { binding: 0, resource: blurTextures[1].createView() },
            { binding: 1, resource: blurTextures[0].createView() },
            { binding: 2, resource: { buffer: flipUniform.gpuBufferV } },
        ]
    });
}

export function getBlurResultTexture() {
    return blurTextures[0];
}

export function resizeBlur(viewportSize, tex) {
    size = [...viewportSize];
    createBindGroups(viewportSize, tex);
}

export function addBlurCommands(cmdEncoder) {
    const pass = cmdEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroupParams);

    pass.setBindGroup(1, bindGroupHInit);
    pass.dispatchWorkgroups(
        Math.ceil(inTexture.width / blockDim),
        Math.ceil(inTexture.height / batch[1])
    );

    pass.setBindGroup(1, bindGroupV);
    pass.dispatchWorkgroups(
        Math.ceil(inTexture.height / blockDim),
        Math.ceil(inTexture.width / batch[1])
    );
  
    pass.end();
}