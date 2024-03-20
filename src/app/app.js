import {WebGPURenderer} from './renderer/webgpu-renderer.js';
import {CompositePass} from './post-processing/composite-pass.js';
import {ReactionDiffusion} from './compute/reaction-diffusion.js';
import {Paint} from './compute/paint.js';
import {isMobileDevice} from './utils/is-mobile.js';

let devicePixelRatio, renderer, paint, reactionDiffusion, compositePass, gpuTiming;

const REACTION_DIFFUSION_RESOLUTION_FACTOR = isMobileDevice ? .2 : 0.25;

const PAINT_RESOLUTION_FACTOR = 0.75;

const timing = {
  // the target duration of one frame in milliseconds
  TARGET_FRAME_DURATION_MS: 16,

  // total time in milliseconds
  timeMS: 0,

  // duration between the previous and the current animation frame in milliseconds
  deltaTimeMS: 0,

  // total frame count according to the target frame duration
  frames: 0,

  // relative frames according to the target frame duration (1 = 60 fps)
  // gets smaller with higher frame rates --> use to adapt animation timing
  deltaFrames: 0,
};

const pointerInfo = {
  isDown: false,

  // normalized pointer position (0..1, flip-y)
  position: [0, 0],

  // normalized pointer position from the previous frame
  previousPosition: [0, 0],
};

async function init() {
  const adapter = await navigator.gpu?.requestAdapter();

  if (!adapter) return;

  const canvas = document.querySelector('canvas');
  devicePixelRatio = Math.min(2, window.devicePixelRatio);

  renderer = new WebGPURenderer(canvas, devicePixelRatio);
  await renderer.init(adapter);

  paint = new Paint(renderer);

  reactionDiffusion = new ReactionDiffusion(renderer, paint);

  compositePass = new CompositePass(renderer, paint, reactionDiffusion);
  await compositePass.init();

  initGPUTiming();
  initPointerInteraction(canvas);
  initResizeObserver(canvas);

  run(0);
}

function initGPUTiming() {
  if (!renderer.canTimestamp) return;

  const querySet = renderer.device.createQuerySet({
    type: 'timestamp',
    count: 2,
  });
  const resolveBuffer = renderer.device.createBuffer({
    size: querySet.count * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const resultBuffer = renderer.device.createBuffer({
    size: resolveBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  gpuTiming = {
    querySet, resolveBuffer, resultBuffer, averageValue: 0, maxValue: 0
  };
}

function initPointerInteraction(canvas) {
  canvas.addEventListener('pointerdown', e => onPointerDown(e));
  canvas.addEventListener('pointerup', e => onPointerUp(e));
  canvas.addEventListener('pointerleave', e => onPointerUp(e));
  canvas.addEventListener('pointermove', e => onPointerMove(e));
}

function getNormalizedPointerPosition(e) {
  const size = renderer.getSize().map(value => value / window.devicePixelRatio);
  return [
    e.clientX / size[0],
    1 - (e.clientY / size[1])
  ];
}

function onPointerDown(e) {
  pointerInfo.isDown = true;

  pointerInfo.position = getNormalizedPointerPosition(e);
  pointerInfo.previousPosition = [...pointerInfo.position];
}

function onPointerMove(e) {
  if (!pointerInfo.isDown) return;

  pointerInfo.position = getNormalizedPointerPosition(e);
}

function onPointerUp(e) {
  pointerInfo.isDown = false;
}

function initResizeObserver(canvas) {
  // handle resizing of the canvas
  const observer = new ResizeObserver(entries => {
    const entry = entries[0];
    const contentBox = entry.contentBoxSize;
    const dpContentBox = entry.devicePixelContentBoxSize;
    const width = dpContentBox?.[0].inlineSize || contentBox[0].inlineSize * devicePixelRatio;
    const height = dpContentBox?.[0].blockSize || contentBox[0].blockSize * devicePixelRatio;
    resize(width, height);
  });
  observer.observe(canvas);
}

function run(t = 0) {
  updateTiming(t);

  const commandEncoder = renderer.device.createCommandEncoder();

  animate(commandEncoder);
  render(commandEncoder);

  renderer.device.queue.submit([commandEncoder.finish()]);

  if (renderer.canTimestamp && gpuTiming.resultBuffer.mapState === 'unmapped') {
    gpuTiming.resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const times = new BigInt64Array(gpuTiming.resultBuffer.getMappedRange());
      const gpuTime = Number(times[1] - times[0]);
      gpuTiming.maxValue = Math.max(gpuTime, gpuTiming.maxValue);
      gpuTiming.averageValue += gpuTime;
      gpuTiming.averageValue /= 2;
      //console.log(Math.round(gpuTiming.maxValue / 100000) / 10);
      gpuTiming.resultBuffer.unmap();
    });
  }

  requestAnimationFrame(t => run(t));
}

function resize(width, height) {
  if (width <= 1 || height <=1 ) return;

  renderer.setSize(width, height);
  const viewportSize = renderer.getSize();

  paint.init(
      Math.round(viewportSize[0] * PAINT_RESOLUTION_FACTOR),
      Math.round(viewportSize[1] * PAINT_RESOLUTION_FACTOR)
  );

  reactionDiffusion.init(
      Math.round(viewportSize[0] * REACTION_DIFFUSION_RESOLUTION_FACTOR),
      Math.round(viewportSize[1] * REACTION_DIFFUSION_RESOLUTION_FACTOR)
  );

  compositePass.setSize(viewportSize[0], viewportSize[1]);
}

function animate(commandEncoder) {
  const computePassEncoder = commandEncoder.beginComputePass({
      ...(renderer.canTimestamp && {
        timestampWrites: {
          querySet: gpuTiming.querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        }
      })
  });
  paint.compute(computePassEncoder, timing, pointerInfo);
  reactionDiffusion.compute(computePassEncoder);
  computePassEncoder.end();

  if (renderer.canTimestamp) {
    commandEncoder.resolveQuerySet(gpuTiming.querySet, 0, 2, gpuTiming.resolveBuffer, 0);
    if (gpuTiming.resultBuffer.mapState === 'unmapped') {
      commandEncoder.copyBufferToBuffer(gpuTiming.resolveBuffer, 0, gpuTiming.resultBuffer, 0, gpuTiming.resultBuffer.size);
    }
  }

  pointerInfo.previousPosition = [...pointerInfo.position];
}

function render(commandEncoder) {
  const compositePassEncoder = commandEncoder.beginRenderPass({...compositePass.renderPassDescriptor});
  compositePass.render(compositePassEncoder);
  compositePassEncoder.end();
}

function updateTiming(t) {
  timing.deltaTimeMS = Math.max(1, Math.min(timing.TARGET_FRAME_DURATION_MS * 2, t - timing.timeMS));
  timing.timeMS = t;
  timing.deltaFrames = timing.deltaTimeMS / timing.TARGET_FRAME_DURATION_MS;
  timing.frames += timing.deltaFrames;
}

export const App = {
  init
};
