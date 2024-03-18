
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
    velocity: vec2f,
    previousVelocity: vec2f,
};

const dispatchSize = vec2u(${dispatchSize[0]},${dispatchSize[1]});
const tileSize = vec2u(${tileSize[0]},${tileSize[1]});

@group(0) @binding(0) var<uniform> renderInfo: RenderInfo;
@group(0) @binding(1) var<uniform> pointerInfo: PointerInfo;
@group(1) @binding(0) var inputTex: texture_2d<f32>;
@group(1) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

fn sdSegment( p: vec2f, a: vec2f, b: vec2f ) -> vec2f {
    let pa = p-a;
    let ba = b-a;
    let h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return vec2f(length( pa - ba*h ), h);
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
  
  let dims: vec2u = vec2<u32>(textureDimensions(inputTex, 0));
  
  let aspectFactor: vec2f = vec2f(dims) / f32(max(dims.x, dims.y));

  // run through the whole tile
  for (var c=0u; c<tileSize.x; c++) {
    for (var r=0u; r<tileSize.y; r++) {
      let local: vec2u = vec2u(c, r) + tileOffset;
      let sample: vec2u = dispatchOffset + local;
      let inputValue = textureLoad(inputTex, sample, 0);
      
      // get the uv coords from the sample position to calculate the signed distance field value
      var uv: vec2f = vec2f(sample) / vec2f(dims);
      
      // aspect correction
      let st = uv * aspectFactor;
      let pointerPos = pointerInfo.position * aspectFactor;
      let prevPointerPos = pointerInfo.previousPosition * aspectFactor;
      
      // get the distance to the segment to draw
      let sdf = sdSegment(st, pointerPos, prevPointerPos);
      let dist = max(0., sdf.x);
      
      // calculate the radius for the new and previous point
      let radiusScale = 1.5;
      let offset = pointerInfo.velocity * renderInfo.deltaTimeMS;
      let strength = length(offset);
      let newRadius = strength * radiusScale;
      let prevOffset = pointerInfo.previousVelocity * renderInfo.deltaTimeMS;
      let prevRadius = length(prevOffset) * radiusScale;
      
      // interpolate between previous and new radius over the segment length
      var radius = newRadius * (1. - sdf.y) + prevRadius * sdf.y;
      radius = clamp(radius, 0.0, .1);
      
      // get a smooth paint from the distance to the segment
      let smoothness = .01;
      var paint = 1. - smoothstep(radius, radius + smoothness * .5, dist + smoothness * .4);
      
      // the strength according to the velocity
      paint = min(1., paint * strength * 200.);
      
      
      
      
      var vel: vec2f = pointerInfo.velocity * 10. * paint;
      vel = vec2(0.);
      vel = (inputValue.xy + vel) / 2.;
      var flowVel = st * 2. - 1.;
      flowVel = normalize(flowVel) * max(0., (2. - length(flowVel)));
      
      
      let velOffset: vec2u = vec2u((uv - (vel + flowVel) * .01) * vec2f(dims));
      let offsetInputValue = textureLoad(inputTex, velOffset, 0);
      //paint += offsetInputValue.b * .25 - .1;
      vel += offsetInputValue.xy * 0.1;
      
      // combine with the previous paint
      var value = clamp(inputValue.b + paint, 0., 1.);
      
      var result: vec4f = vec4(vec4(vel, value, value));
      
      // dissipate the paint over time
      result *= 0.95;


      //textureStore(outputTex, sample, vec4(f32(localInvocationID.x) / 10., f32(localInvocationID.y) / 10., 0., 1.0));
      //textureStore(outputTex, sample, vec4(uv, 0., 1.0));
      
      textureStore(outputTex, sample, result);
    }
  }
}

`;
