import * as wgh from 'webgpu-utils';
import {ReactionDiffusionShader, ReactionDiffusionShaderDispatchSize} from '../shader/reaction-diffusion.js';
import {Float16Array} from '@petamoriken/float16';

export class ReactionDiffusion {

    ITERATIONS = 10;

    constructor(renderer) {
        this.renderer = renderer;

        // create pipeline and bind group layouts
        const module = this.renderer.device.createShaderModule({ code: ReactionDiffusionShader });
        const defs = wgh.makeShaderDataDefinitions(ReactionDiffusionShader);
        this.pipelineDescriptor = {
            compute: {
                module,
                entryPoint: 'compute_main',
            }
        };
        const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, this.pipelineDescriptor);
        descriptors[0].entries.push({
            binding: 2,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
            visibility: GPUShaderStage.COMPUTE
        });
        this.bindGroupLayout = this.renderer.device.createBindGroupLayout(descriptors[0]);
    }

    async init() {
        const pipelineLayout = this.renderer.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });
        this.computePipeline = this.renderer.device.createComputePipeline({
            label: 'reaction diffusion compute pipeline',
            layout: pipelineLayout,
            ...this.pipelineDescriptor
        });

        const w = 200;
        const h = 200;

        this.blurTextures = new Array(2).fill(null).map((v, ndx) => {
            const texture = this.renderer.device.createTexture({
                size: { width: w, height: h },
                format: 'rgba16float',
                usage:
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });

            let data;
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
                data = new Float16Array(rgba);
            } else {
                data = new Float16Array(new Array(w * h * 4).fill(0));
            }

            this.renderer.device.queue.writeTexture({ texture }, data.buffer, { bytesPerRow: w * 8 }, { width: w, height: h });

            return texture;
        });

        this.emptyTexture = this.renderer.device.createTexture({
            label: 'empty texture',
            size: [1, 1],
            dimension: '2d',
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        this.swapBindGroups = [
            this.renderer.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.emptyTexture.createView() },
                    { binding: 1, resource: this.blurTextures[0].createView() },
                    { binding: 2, resource: this.blurTextures[1].createView() },
                ]
            }),
            this.renderer.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.emptyTexture.createView() },
                    { binding: 1, resource: this.blurTextures[1].createView() },
                    { binding: 2, resource: this.blurTextures[0].createView() },
                ]
            })
        ];
    }

    get resultStorageTexture() {
        return this.blurTextures[0];
    }

    compute(computePassEncoder) {
        const dispatches = [
            Math.ceil(200 / ReactionDiffusionShaderDispatchSize[0]),
            Math.ceil(200 / ReactionDiffusionShaderDispatchSize[1])
        ];

        computePassEncoder.setPipeline(this.computePipeline);

        for(let i = 0; i < this.ITERATIONS; i++) {
            computePassEncoder.setBindGroup(0, this.swapBindGroups[0]);
            computePassEncoder.dispatchWorkgroups(dispatches[0], dispatches[1]);

            computePassEncoder.setBindGroup(0, this.swapBindGroups[1]);
            computePassEncoder.dispatchWorkgroups(dispatches[0], dispatches[1]);
        }
    }
}
