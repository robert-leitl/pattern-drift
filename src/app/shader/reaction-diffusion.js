
// shader constants
import {WGSLBilinearSample, WGSLMapFunction} from './chunks.js';

const kernelSize = 3;
const workgroupSize = [8, 8];
// each thread handles a tile of pixels
const tileSize = [2, 2];
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

export const ReactionDiffusionShaderDispatchSize = dispatchSize;

// language=C
export const ReactionDiffusionShader = `

const kernelSize = ${kernelSize};
const dispatchSize = vec2u(${dispatchSize[0]},${dispatchSize[1]});
const tileSize = vec2u(${tileSize[0]},${tileSize[1]});

@group(0) @binding(0) var seedTex: texture_2d<f32>;
@group(0) @binding(1) var inputTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

// the cache for the texture lookups (tileSize * workgroupSize)
var<workgroup> cache: array<array<vec4f, ${cacheSize[0]}>, ${cacheSize[1]}>;

${WGSLMapFunction}

${WGSLBilinearSample}

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, 1)
fn compute_main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationID : vec3<u32>,
  @builtin(global_invocation_id) globalInvocationID : vec3<u32>
) {
  // each thread adds a tile of pixels to the workgroups shared memory

  let kernelArea: u32 = kernelSize * kernelSize;

  // the kernel offset (number of pixels next to the center of the kernel) defines
  // the border area next to the dispatched (=work) area that has to be included
  // within the pixel cache
  let kernelOffset: vec2u = vec2((kernelSize - 1) / 2);

  // the local pixel offset of this threads tile
  let tileOffset: vec2u = localInvocationID.xy * tileSize;

  // the global pixel offset of the workgroup
  let dispatchOffset: vec2u = workGroupID.xy * dispatchSize;

  // get texture dimensions
  let dims: vec2u = vec2<u32>(textureDimensions(inputTex, 0));
  let seedDim: vec2u = vec2<u32>(textureDimensions(seedTex, 0));

  // add this threads tiles pixels to the cache
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;

      // subtract the kernel offset to include the border pixels needed
      // for the convolution of the kernel within the dispatch (work) area
      var sample: vec2u = dispatchOffset + local - kernelOffset;
      var sampleUv: vec2f = vec2f(sample);
      
      // get the sample within the seed texture dimension
      let seedSample: vec2u = vec2u((vec2f(sample) / vec2f(dims)) * vec2f(seedDim));
      let seed: vec4f = textureLoad(seedTex, seedSample, 0);
      
      // offset the input sampling by the paint velocity
      sampleUv -= normalize(seed.xy + .001) * max(0., min(.08, length(seed.xy)));
      
      // perform manual bilinear sampling of the input texture
      let input: vec4f = texture2D_bilinear(inputTex, sampleUv, dims);
      var value: vec4f = vec4f(input.rg, seed.xy);
      
      // fill in chemical b from the seed texture
      let paint = clamp(seed.b, 0., 1.);
      value.g = mix(input.g, (input.g + paint) / 2, seed.a);
      value.r = mix(input.r, (input.r + (1. - paint)) / 2., seed.a);
      value.g = clamp(value.g, 0., 1.);
      value.r = clamp(value.r, 0., 1.);
      
      cache[local.y][local.x] = value;
    }
  }

  workgroupBarrier();

  // global pixel bounds within an application of the kernel is valid
  let bounds: vec4u = vec4u(
    dispatchOffset,
    min(dims, dispatchOffset + dispatchSize)
  );

  let laplacian: array<f32, 9> = array(
    0.05, 0.20, 0.05,
    0.20, -1.0, 0.20,
    0.05, 0.20, 0.05,
  );

  // run through the whole cache area
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;
      let sample: vec2u = dispatchOffset + local - kernelOffset;

      // only apply the kernel to pixels for which we have all
      // necessary pixels in the cache
      if (all(sample >= bounds.xy) && all(sample < bounds.zw)) {
      
        let uv: vec2f = vec2f(sample) / vec2f(dims);

        // convolution with laplacian kernel
        var lap = vec2f(0);
        let ks: i32 = i32(kernelSize);
        for (var x = 0; x < ks; x++) {
          for (var y = 0; y < ks; y++) {
            let i = vec2i(local) + vec2(x, y) - vec2i(kernelOffset);
            lap += cache[i.y][i.x].xy * laplacian[y * ks + x];
          }
        }

        // reaction diffusion calculation
        let cacheValue: vec4f = cache[local.y][local.x];
        let seedVel: vec2f = cacheValue.zw;
        let seedVelLen: f32 = length(seedVel);
        let rd0 = cacheValue.xy;
        // update params from seed values
        let rdScale = .9;
        let dA = rdScale;
        let dB = .3 * rdScale - min(0.1, max(0., seedVelLen * .5));
        let feedInt = min(1., max(0., 1. - pow(seedVelLen - 1., 4.) - .1));
        let killInt = min(1., max(0., pow(seedVelLen, 4.) + .4));
        let feed = map(feedInt, 0., 1., 0.005, .095);
        let kill = map(killInt, 0., 1., 0.055, .07);
        // calculate result
        let A = rd0.x;
        let B = rd0.y;
        let reaction = A * B * B;
        let rd = vec2f(
          A + (dA * lap.x - reaction + feed * (1. - A)),
          B + (dB * lap.y + reaction - (kill + feed) * B),
        );

        textureStore(outputTex, sample, vec4(rd, 0., 1.0));

        // debug code
        //textureStore(outputTex, sample, vec4(cache[local.y][local.x], 1.0));
        //textureStore(outputTex, sample, vec4(uv, 0., 1.0));
      }
    }
  }
}

`;

