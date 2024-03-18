export const CompositeShader = `

@group(0) @binding(0) var colorTexSampler: sampler;
@group(0) @binding(1) var colorTex: texture_2d<f32>;

@fragment
fn frag_main(@location(0) uv : vec2f) -> @location(0) vec4f {
    let colorTexSize : vec2f = vec2f(textureDimensions(colorTex));

    var color : vec4f = textureSample(colorTex, colorTexSampler, uv);
    color.a = 1.;

    return color;
}

`;
