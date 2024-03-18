
// shader constants
const workgroupSize = [8, 8];
// each thread handles a tile of pixels
const tileSize = [3, 3];
// the required number of workgroup dispatches is given by
// dividing the total area by the dispatch size
const dispatchSize = [
    tileSize[0] * workgroupSize[0],
    tileSize[1] * workgroupSize[1]
];

export const PaintDispatchSize = dispatchSize;

// language=C
export const PaintShader = `

struct RenderInfo {
    viewportSize: vec2f,
    deltaTimeMS: f32,
    timeMS: f32
};

struct PointerInfo {
    position: vec2f,
    previousPosition: vec2f,
    velocity: vec2f
};

const dispatchSize = vec2u(${dispatchSize[0]},${dispatchSize[1]});
const tileSize = vec2u(${tileSize[0]},${tileSize[1]});

@group(0) @binding(0) var<uniform> renderInfo: RenderInfo;
@group(0) @binding(1) var<uniform> pointerInfo: PointerInfo;
@group(1) @binding(0) var inputTex: texture_2d<f32>;
@group(1) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

fn sdSegment( p: vec2f, a: vec2f, b: vec2f ) -> f32 {
    let pa = p-a;
    let ba = b-a;
    let h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

fn blendScreen(base: f32, blend: f32, opacity: f32) -> f32 {
    let r = 1.0-((1.0-base)*(1.0-blend));
    return r * opacity + base * (1.0 - opacity);
}

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, 1)
fn compute_main(
  @builtin(workgroup_id) workGroupID : vec3<u32>,
  @builtin(local_invocation_id) localInvocationID : vec3<u32>,
  @builtin(global_invocation_id) globalInvocationID : vec3<u32>
) {
  // the local pixel offset of this threads tile
  let tileOffset: vec2u = localInvocationID.xy * tileSize;

  // the global pixel offset of the workgroup
  let dispatchOffset: vec2u = workGroupID.xy * dispatchSize;

  // run through the whole tile
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;
      let sample: vec2u = dispatchOffset + local;
      let uv: vec2f = vec2f(sample) / renderInfo.viewportSize;
      let inputValue = textureLoad(inputTex, sample, 0);
      
      let offset = pointerInfo.velocity * renderInfo.deltaTimeMS;
      let strength = length(offset) * 2.;
      let d = max(0., sdSegment(uv, pointerInfo.position, pointerInfo.previousPosition));
      let segment = 1. - smoothstep(0., 0.5 * strength, d);
      
      var value = inputValue.r + segment;
      
      // dissipate the paint over time
      value *= 0.95;
      
      let alpha = clamp(1. - value, 0., 1.);

      //textureStore(outputTex, sample, vec4(f32(localInvocationID.x) / 10., f32(localInvocationID.y) / 10., 0., 1.0));
      //textureStore(outputTex, sample, vec4(uv, 0., 1.0));
      
      textureStore(outputTex, sample, vec4(value, 0., 0., alpha));
    }
  }
}

`;
