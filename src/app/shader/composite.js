// language=C
export const CompositeShader = `

@group(0) @binding(0) var colorTexSampler: sampler;
@group(0) @binding(1) var colorTex: texture_2d<f32>;
@group(0) @binding(2) var paintTex: texture_2d<f32>;

// Based on: https://www.shadertoy.com/view/ldBfRV
fn distort(r: vec2<f32>, alpha: f32) -> vec2<f32> {
    return r + r * -alpha * (1. - dot(r, r) * 1.25);
}

@fragment
fn frag_main(@location(0) uv : vec2f) -> @location(0) vec4f {
    let colorTexSize : vec2f = vec2f(textureDimensions(colorTex));
    let colorTexelSize = 1. / colorTexSize;
    
    // bulge distortion
    var st: vec2f = uv * 2. - 1.;
    st *= 0.5;
    st = distort(st, -1.);
    st = st * .5 + .5;

    var color : vec4f = textureSample(colorTex, colorTexSampler, st);
    var paint : vec4f = textureSample(paintTex, colorTexSampler, st);
    let vel: vec2f = paint.xy;
    let velLen: f32 = dot(vel, vel);
    
    // emboss effect
    let embossScale = 1.;
    let tl: vec4f = textureSample(colorTex, colorTexSampler, st + vec2(-colorTexelSize.x,  colorTexelSize.y) * embossScale);
    let br: vec4f = textureSample(colorTex, colorTexSampler, st + colorTexelSize * embossScale);
    var emboss: vec3f = (2.0 * br.rgb - color.rgb - tl.rgb);
    let luminance: f32 = clamp(0.299 * emboss.r + 0.587 * emboss.g + 0.114 * emboss.b, 0.0, 1.0);
    emboss = vec3f(0.) + vec3(luminance);
    
    let rgb = color.rgb;// + 0.5 * vec3f((normalize(paint.xy + 0.00001) * min(1., length(paint.xy))) * .5 + .5, 1.);
    
    // main color
    let contourHardness = min(.4, velLen * 20.);
    var contour: vec3f = 1. - vec3f(smoothstep(0. + contourHardness * .3, .9 - contourHardness, rgb.g));

    //return vec4(rgb + sum, 1.);
    return vec4(contour, 1.);
}

`;
