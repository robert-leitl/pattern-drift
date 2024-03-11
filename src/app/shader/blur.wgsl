struct Params {
    filterSize: i32,
    blockSize: u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var texSampler: sampler;
@group(1) @binding(0) var inputTex: texture_2d<f32>;
@group(1) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(2) var<uniform> flip: u32;


// shared memory to 'cache' texture lookups in a 128x4 pixel stripe
var<workgroup> stripe : array<array<vec3f, 128>, 4>; // 128 = stripeDim

@compute @workgroup_size(64, 1, 1)
fn compute_main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationID : vec3<u32>
) {
  _ = texSampler; // unused binding 
  let workgroupSizeX: i32 = 64;
  let stripeDim: i32 = 128;
  let workGroupsPerStripe: u32 = u32(stripeDim / workgroupSizeX);
  let filterOffset: i32 = max(0, (params.filterSize - 1)) / 2;
  let dims: vec2i = vec2<i32>(textureDimensions(inputTex, 0));
  let baseIndex: vec2i = vec2<i32>(workGroupID.xy * vec2(params.blockSize, 4) + localInvocationID.xy * vec2(workGroupsPerStripe, 1)) - vec2(filterOffset, 0);

  // each thread samples 16 pixels to the shared memory of the workgroup
  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var loadIndex = baseIndex + vec2(c, r);

      if (flip != 0u) {
        loadIndex = loadIndex.yx;
      }

      /*stripe[r][workGroupsPerStripe * localInvocationID.x + u32(c)] = textureSampleLevel(
        inputTex,
        texSampler,
        (vec2<f32>(loadIndex) + vec2<f32>(0.5, 0.5)) / vec2<f32>(dims),
        0.0
      ).rgb;*/

      stripe[r][workGroupsPerStripe * localInvocationID.x + u32(c)] = textureLoad(
        inputTex,
        loadIndex,
        0
      ).rgb;
    }
  }

  workgroupBarrier();

  // each thread writes 16 blurred pixels to the result texture
  for (var r = 0; r < 4; r++) {
    for (var c = 0; c < 4; c++) {
      var writeIndex = baseIndex + vec2(c, r);
      if (flip != 0) {
        writeIndex = writeIndex.yx;
      }

      let center = i32(workGroupsPerStripe * localInvocationID.x) + c;
      if (center >= filterOffset && center < 128 - filterOffset && all(writeIndex < dims)) {
        var acc = vec3(0.0, 0.0, 0.0);
        for (var f = 0; f < params.filterSize; f++) {
          var i = center + f - filterOffset;
          acc = acc + (1.0 / f32(params.filterSize)) * stripe[r][i];
        }
        textureStore(outputTex, writeIndex, vec4(acc, 1.0));
      }
    }
  }
}