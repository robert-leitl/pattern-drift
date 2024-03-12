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

    createBindGroups([1, 1], tex);
}

function createBindGroups(viewportSize, tex) { 
    inTexture = tex;
    const w = inTexture.width / 2;
    const h = inTexture.height / 2;

    blurTextures = blurTextures.map((v, ndx) => {
        const texture = _device.createTexture({
            size: { width: w, height: h },
            format: 'rgba8unorm',
            usage: 
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        let data;
        if (ndx === 0) {
            const rgb = new Array(w * h * 4).fill(0);
            const bx = [w / 2 - 5, w / 2 + 5];
            const by = [h / 2 - 5, h / 2 + 5];
            for(let x=0; x<w; x++) {
                for(let y=0; y<h; y++) {
                    const v = x > bx[0] && x < bx[1] && y > by[0] && y < by[1];
                    rgb[(x + y * w) * 4 + 0] = v ? 0 : 255;
                    rgb[(x + y * w) * 4 + 1] = v ? 255 : 0;
                    rgb[(x + y * w) * 4 + 2] = 0;
                }
            }
            data = new Uint8Array(rgb);
        } else {
            data = new Uint8Array(new Array(w * h * 4).fill(255));
        }

        _device.queue.writeTexture({ texture }, data, { bytesPerRow: w * 4 }, { width: w, height: h });

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

export function resizeConvolution(viewportSize, tex) {
    createBindGroups(viewportSize, tex);
}

export function addConvolutionCommands(cmdEncoder) {
    const pass = cmdEncoder.beginComputePass();

    const dispatches = [
        Math.ceil(inTexture.width / dispatchSize[0]),
        Math.ceil(inTexture.height / dispatchSize[1])
    ];

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroupParams);

    for(let i = 0; i < 25; i++) {
        pass.setBindGroup(1, bindGroup0);
        pass.dispatchWorkgroups(...dispatches);
    
        pass.setBindGroup(1, bindGroup1);
        pass.dispatchWorkgroups(...dispatches);
    }
  
    pass.end();
}