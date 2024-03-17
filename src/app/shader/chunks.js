export const PostProcessingVertexShader = `
	struct Inputs {
		@location(0) position: vec2<f32>,
		@location(1) uv: vec2<f32>,
	}

	struct Output {
		@builtin(position) position: vec4<f32>,
        @location(0) uv: vec2f
	}

	@vertex
	fn vertex_main(input: Inputs) -> Output {
		var output: Output;
		output.position = vec4(input.position, 0.0, 1.0);
		output.uv = input.uv;

		return output;
	}
`
