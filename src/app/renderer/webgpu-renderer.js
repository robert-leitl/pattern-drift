export class WebGPURenderer {

    constructor(canvas) {
        this.canvas = canvas;
    }

    async init(adapter) {
        this.device = await adapter.requestDevice();

        this.context = this.canvas.getContext('webgpu');
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });

        this.colorAttachmentTemplate = {
            view: null,
            clearValue: { r: 1, g: 1, b: 1, a: 1},
            loadOp: 'clear',
            storeOp: 'store'
        };
    }

    get colorAttachment() {
        return {
            ...this.colorAttachmentTemplate,
            view: this.context.getCurrentTexture().createView()
        };
    }

    setSize(width, height) {
        if (!this.device) return;

        this.canvas.width = Math.max(1, Math.min(width, this.device.limits.maxTextureDimension2D));
        this.canvas.height = Math.max(1, Math.min(height, this.device.limits.maxTextureDimension2D));
    }

    getSize() {
        return [this.canvas.width, this.canvas.height];
    }
}
