import * as wgh from 'webgpu-utils';
import { initComposite, addCompositeCommands, resizeComposite } from './composite';
import { initBlur, addBlurCommands, resizeBlur, getBlurResultTexture } from './blur';
import { addConvolutionCommands, getConvolutionResultTexture, initConvolution, resizeConvolution } from './convolution';

let adapter, device, context, presentationFormat;
let canvas, pixelRatio, viewportSize;

async function main() {
  pixelRatio = window.devicePixelRatio;
  adapter = await navigator.gpu?.requestAdapter();
  device = await adapter?.requestDevice();
  if (!device) {
    fail('need a browser that supports WebGPU');
    return;
  }
  
  canvas = document.querySelector('canvas');

  context = canvas.getContext('webgpu');
  presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  const imgTex = await wgh.createTextureFromImage(device, new URL('../assets/img.jpg', import.meta.url), {
    mips: false,
    flipY: true,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
  });
  const testTexture = createTestTexture([100, 100]);
  
  initConvolution(device, testTexture);
  initBlur(device, testTexture);
  initComposite(device, presentationFormat, imgTex);

  const observer = new ResizeObserver(entries => {
    for (const entry of entries) {
      const width = entry.contentBoxSize[0].inlineSize;
      const height = entry.contentBoxSize[0].blockSize;
      resize(width, height);
    }
  });
  observer.observe(canvas);

  resize(1, 1);
  run();
}

function createTestTexture(size) {
  const w = size[0];
  const h = size[1];
  const rgb = new Array(w * h * 4).fill(255);
  for(let x=0; x<w; x++) {
    for(let y=0; y<h; y++) {
      const v = x > w / 3 && x < (2 * w) / 3 && y > h / 3 && y < (2 * h) / 3 ? 0 : 255;
      rgb[(x + y * w) * 4 + 0] = v;
      rgb[(x + y * w) * 4 + 1] = v;
      rgb[(x + y * w) * 4 + 2] = v;
    }
  }
  const data = new Uint8Array(rgb);
  const texture = device.createTexture({
    size: { width: w, height: w },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture }, data, { bytesPerRow: w * 4 }, { width: w, height: h });
  return texture;
}

function resize(width, height) {
  canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
  canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
  viewportSize = [canvas.width, canvas.height].map(v => v * pixelRatio);

  resizeConvolution(viewportSize);
  resizeBlur(viewportSize);
  resizeComposite(viewportSize, getConvolutionResultTexture());
}

function run(t = 0) {
  render();
  requestAnimationFrame(t => run(t));
}

function render() {
  if (canvas.width < 1 || canvas.height < 1) return;

  const cmdEncoder = device.createCommandEncoder();

  addConvolutionCommands(cmdEncoder);
  addBlurCommands(cmdEncoder);
  addCompositeCommands(cmdEncoder, context.getCurrentTexture().createView());

  device.queue.submit([cmdEncoder.finish()]);
}

function fail(msg) {
  const elem = document.createElement('p');
  elem.textContent = msg;
  elem.style.color = 'red';
  document.body.appendChild(elem);
}

main();