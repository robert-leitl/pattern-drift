
export const convolutionWGSL = `

struct Params {
    kernelSize: i32,
    blockSize: vec2u,
    pixelCacheTileSize: vec2u
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var inputTex: texture_2d<f32>;
@group(1) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

var<workgroup> pixelCache: array<array<vec3f, 32>, 32>;

@compute @workgroup_size(8, 8, 1)
fn compute_main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationID : vec3<u32>,
  @builtin(global_invocation_id) globalInvocationID : vec3<u32>
) {
  let tile: vec2i = vec2i(params.pixelCacheTileSize);

  // fill the pixel cache: each thread loads blockSize of pixels to the memory shared by the workgroup





  let tileSize: vec2u = vec2u(8);
  let pixelCacheSize: vec2u = vec2u(32);
  //let center: vec2u = globalInvocationID.xy;
  //let uv: vec2u = workGroupID.xy * params.blockSize + localInvocationID.xy * params.pixelCacheTileSize;

  

  var ndx: vec2u = workGroupID.xy * params.blockSize + localInvocationID.xy * params.pixelCacheTileSize;

  for (var r=0; r<tile.x; r++) {
    for (var c=0; c<tile.y; c++) {
      let uv: vec2u = ndx + vec2u(u32(c), u32(r));

      var acc = vec3f(0);
      var count: u32 = 0;

      let fs = (params.kernelSize - 1) / 2;
      for (var x = -fs; x <= fs; x++) {
        for (var y = -fs; y <= fs; y++) {
          let sample = vec2i(uv) + vec2(x, y);
          acc += textureLoad(inputTex, sample, 0).rgb;
          count++;
        }
      }

      acc /= f32(count);
      textureStore(outputTex, uv, vec4(acc, 1.0));

    }
  }
  
  
}`;