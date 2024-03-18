import * as wgh from 'webgpu-utils';
import {PaintDispatchSize, PaintShader} from '../shader/paint.js';

export class Paint {

    constructor(renderer) {
        this.renderer = renderer;

        // create pipeline and bind group layouts
        const module = this.renderer.device.createShaderModule({ code: PaintShader });
        const defs = wgh.makeShaderDataDefinitions(PaintShader);
        this.pipelineDescriptor = {
            compute: {
                module,
                entryPoint: 'compute_main',
            }
        };
        const descriptors = wgh.makeBindGroupLayoutDescriptors(defs, this.pipelineDescriptor);
        descriptors[0].entries.push({
            binding: 1,
            storageTexture: { access: 'write-only', format: 'rgba16float' },
            visibility: GPUShaderStage.COMPUTE
        });
        this.bindGroupLayout = this.renderer.device.createBindGroupLayout(descriptors[0]);

        const pipelineLayout = this.renderer.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });
        this.computePipeline = this.renderer.device.createComputePipeline({
            label: 'paint compute pipeline',
            layout: pipelineLayout,
            ...this.pipelineDescriptor
        });

        this.init(100, 100);
    }

    init(width, height) {
        // bind groups are swapped each frame
        this.currentSwapIndex = 0;

        this.createTextures(width, height);
        this.createBindGroups();
    }

    get resultStorageTexture() {
        return this.swapTextures[0];
    }

    createTextures(width, height) {
        if (this.swapTextures) {
            this.swapTextures.forEach(texture => texture.destroy());
        }

        this.swapTextures = new Array(2).fill(null).map((v, ndx) => {
            return this.renderer.device.createTexture({
                size: {width, height},
                format: 'rgba16float',
                usage:
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });
        });

        this.dispatches = [
            Math.ceil(width / PaintDispatchSize[0]),
            Math.ceil(height / PaintDispatchSize[1])
        ];
    }

    createBindGroups() {
        this.swapBindGroups = [
            this.renderer.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[0].createView() },
                    { binding: 1, resource: this.swapTextures[1].createView() },
                ]
            }),
            this.renderer.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: this.swapTextures[1].createView() },
                    { binding: 1, resource: this.swapTextures[0].createView() },
                ]
            })
        ];
    }

    compute(computePassEncoder) {
        computePassEncoder.setPipeline(this.computePipeline);
        computePassEncoder.setBindGroup(0, this.swapBindGroups[this.currentSwapIndex]);
        computePassEncoder.dispatchWorkgroups(this.dispatches[0], this.dispatches[1]);

        this.currentSwapIndex = (this.currentSwapIndex + 1) % 2;
    }

}
