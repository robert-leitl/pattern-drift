struct Params {
    kernelSize: u32,
    dispatchSize: vec2u
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var seedTex: texture_2d<f32>;
@group(1) @binding(1) var inputTex: texture_2d<f32>;
@group(1) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

// the cache for the texture lookups (tileSize * workgroupSize)
var<workgroup> cache: array<array<vec3f, 24>, 24>;

@compute @workgroup_size(8, 8, 1)
fn compute_main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationID : vec3<u32>,
  @builtin(global_invocation_id) globalInvocationID : vec3<u32>
) {
  // each thread adds a tile of pixels to the workgroups shared memory
  let tileSize: vec2u = vec2u(3, 3);

  let kernelArea: u32 = params.kernelSize * params.kernelSize;

  // the kernel offset (number of pixels next to the center of the kernel) defines
  // the border area next to the dispatched (=work) area that has to be included
  // within the pixel cache
  let kernelOffset: vec2u = vec2((params.kernelSize - 1) / 2);

  // the local pixel offset of this threads tile
  let tileOffset: vec2u = localInvocationID.xy * tileSize;

  // the global pixel offset of the workgroup
  let dispatchOffset: vec2u = workGroupID.xy * params.dispatchSize;

  let dims: vec2u = vec2<u32>(textureDimensions(inputTex, 0));

  // add this threads tiles pixels to the cache
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;

      // subtract the kernel offset to include the border pixels needed
      // for the convolution of the kernel within the dispatch (work) area
      var sample: vec2u = dispatchOffset + local - kernelOffset;

      // clamp the sample to the edges of the input texture
      sample = clamp(sample, vec2u(0, 0), dims);

      let seed: vec4f = textureLoad(seedTex, sample, 0);
      let input: vec4f = textureLoad(inputTex, sample, 0);
      cache[local.y][local.x] = input.rgb;
    }
  }

  workgroupBarrier();

  // global pixel bounds within an application of the kernel is valid
  let bounds: vec4u = vec4u(
    dispatchOffset,
    min(dims, dispatchOffset + params.dispatchSize)
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

        // convolution with laplacian kernel
        var lap = vec2f(0);
        let ks: i32 = i32(params.kernelSize);
        for (var x = 0; x < ks; x++) {
          for (var y = 0; y < ks; y++) {
            let i = vec2i(local) + vec2(x, y) - vec2i(kernelOffset);
            lap += cache[i.y][i.x].xy * laplacian[y * ks + x];
          }
        }

        let dA = 1.;
        let dB = .4;
        let feed = .062;
        let kill = .062;

        let rd0 = cache[local.y][local.x].xy;
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
        //textureStore(outputTex, sample, vec4(f32(localInvocationID.x) / 10., f32(localInvocationID.y) / 10., 0., 1.0));
      }
    }
  }
}