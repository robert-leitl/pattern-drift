import compositeWGSL from './shader/composite.wgsl?raw';
import * as wgh from 'webgpu-utils';

let _device, pipeline, bindGroupParams, bindGroup1Layout, bindGroup1, uniforms, sampler;

export function initComposite(device, presentationFormat, tex) {
    _device = device;
    const module = _device.createShaderModule({ code: compositeWGSL });
    const defs = wgh.makeShaderDataDefinitions(compositeWGSL);
    const pipelineDesc = {
        vertex: {
            module: module,
            entryPoint: 'vertex_main',
        },
        fragment: {
            module,
            entryPoint:'frag_main',
            targets: [
                { format: presentationFormat }
            ]
        },
        primitive: {
            topology: 'triangle-list',
        },
    };
    const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, pipelineDesc);
    const bindGroupParamsLayout = _device.createBindGroupLayout(descriptors[0]);
    bindGroup1Layout = _device.createBindGroupLayout(descriptors[1]);
    const layout = _device.createPipelineLayout({
        bindGroupLayouts: [bindGroupParamsLayout, bindGroup1Layout]
    });
    pipeline = _device.createRenderPipeline({
        layout,
        ...pipelineDesc
    });
    
    uniforms = wgh.makeStructuredView(defs.uniforms.compUniforms);
    uniforms.gpuBuffer = _device.createBuffer({
        size: uniforms.arrayBuffer.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, // a uniform buffer as a destination to copy to
    });
    sampler = _device.createSampler({
        minFilter: 'linear',
        magFilter: 'nearest'
    });
    bindGroupParams = _device.createBindGroup({
        layout: bindGroupParamsLayout,
        entries: [
            { binding: 0, resource: { buffer: uniforms.gpuBuffer } },
        ]
    });
    bindGroup1 = _device.createBindGroup({
        layout: bindGroup1Layout,
        entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: tex.createView() },
        ]
    });
}

export function resizeComposite(viewportSize, tex) {
    uniforms.set({ viewportSize });
    _device.queue.writeBuffer(uniforms.gpuBuffer, 0, uniforms.arrayBuffer);

    bindGroup1 = _device.createBindGroup({
        layout: bindGroup1Layout,
        entries: [ 
            { binding: 0, resource: sampler },
            { binding: 1, resource: tex.createView() },
        ]
    });
}

export function addCompositeCommands(cmdEncoder, view) {
    const pass = cmdEncoder.beginRenderPass({
        colorAttachments: [
            {
                view,
                clearValue: { r: 1, g: 1, b: 1, a: 1},
                loadOp: 'clear',
                storeOp: 'store'
            }
        ]
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroupParams);
    pass.setBindGroup(1, bindGroup1);
    pass.draw(3);
    pass.end();
}