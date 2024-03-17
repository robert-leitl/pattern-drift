import {PostProcessingPass} from './post-processing-pass.js';
import {CompositeShader} from '../shader/composite.js';
import * as wgh from 'webgpu-utils';

export class CompositePass extends PostProcessingPass {

    constructor(renderer, reactionDiffusion) {
        super(renderer);

        this.reactionDiffusion = reactionDiffusion;

        // create bind group layouts
        const defs = wgh.makeShaderDataDefinitions(CompositeShader);
        const pipelineLayoutPartial = {
            fragment: {
                entryPoint:'frag_main'
            },
        }
        const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, pipelineLayoutPartial);
        const bindGroupLayout = renderer.device.createBindGroupLayout(descriptors[0]);

        this.sampler = renderer.device.createSampler({
            minFilter: 'linear',
            magFilter: 'nearest'
        });
        this.bindGroup = renderer.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: this.sampler },
                { binding: 1, resource: this.reactionDiffusion.resultStorageTexture.createView() },
            ]
        });

        this.renderPassDescriptorTemplate = { label: 'composite pass' };

        this.bindGroupLayouts = [bindGroupLayout];
    }

    async init() {
        await super.init(
            'composite pipeline',
            CompositeShader,
            'frag_main',
            this.bindGroupLayouts,
            this.renderer.presentationFormat
        );
    }

    get renderPassDescriptor() {
        return {
            ...this.renderPassDescriptorTemplate,
            colorAttachments: [ this.renderer.colorAttachment ],
        }
    }

    resize() {
        super.resize();
    }

    render(renderPassEncoder) {

        // set bind groups
        renderPassEncoder.setBindGroup(0, this.bindGroup);

        super.render(renderPassEncoder);
    }

}
