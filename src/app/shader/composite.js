// language=C
export const CompositeShader = `

@group(0) @binding(0) var colorTexSampler: sampler;
@group(0) @binding(1) var colorTex: texture_2d<f32>;
@group(0) @binding(2) var paintTex: texture_2d<f32>;

// Credit: https://www.shadertoy.com/view/ldBfRV
fn distort(r: vec2<f32>, alpha: f32) -> vec2<f32> {
    return r + r * -alpha * (1. - dot(r, r));
}

@fragment
fn frag_main(@location(0) uv : vec2f) -> @location(0) vec4f {
    let colorTexSize : vec2f = vec2f(textureDimensions(colorTex));
    
    var st: vec2f = uv * 2. - 1.;
    st *= 0.5;
    st = distort(st, -1.);
    st = st * .5 + .5;

    var color : vec4f = textureSample(colorTex, colorTexSampler, st);
    var paint : vec4f = textureSample(paintTex, colorTexSampler, st);
    
    let rgb = color.rgb + 0.5 * vec3f((normalize(paint.xy + 0.00001) * min(1., length(paint.xy))) * .5 + .5, 1.);

    return vec4(rgb, 1.);
}

`;
