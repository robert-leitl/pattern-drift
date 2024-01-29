import computeShader from './shader/compute.wgsl?raw'

async function start() {
    if (!navigator.gpu) {
        throw new Error('this browser does not support WebGPU');
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('this browser supports webgpu but it appears disabled');
        return;
    }

    const device = await adapter?.requestDevice();
    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.message}`);

        // 'reason' will be 'destroyed' if we intentionally destroy the device.
        if (info.reason !== 'destroyed') {
            // try again
            start();
        }
    });
    
    console.log(adapter, device);

    await main(device);
}

async function main(device) {
    const canvas = document.querySelector('canvas');
    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format
    });

    const dispatchCount = [2, 2, 2];
    const workgroupSize = [4, 4, 4];
    const arrayProd = a => a.reduce((p, v) => p * v);
    const numThreadsPerWorkgroup = arrayProd(workgroupSize);
    const numWorkgroups = arrayProd(dispatchCount);
    const numResults = numWorkgroups * numThreadsPerWorkgroup;
    console.log(numThreadsPerWorkgroup, numWorkgroups, numResults);
    const indexShader = `
        @group(0) @binding(0) var<storage, read_write> workgroupResult: array<vec3u>;
        @group(0) @binding(1) var<storage, read_write> localResult: array<vec3u>;
        @group(0) @binding(2) var<storage, read_write> globalResult: array<vec3u>;

        @compute @workgroup_size(${workgroupSize}) fn indexCompute(
            @builtin(workgroup_id) workgroup_id: vec3<u32>,
            @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
            @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
            @builtin(local_invocation_index) local_invocation_index: u32,
            @builtin(num_workgroups) num_workgroups: vec3<u32>
        ) {
            let workgroup_index = workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.x * num_workgroups.y;
            let global_invocation_index = workgroup_index * ${numThreadsPerWorkgroup} + local_invocation_index;

            workgroupResult[global_invocation_index] = workgroup_id;
            localResult[global_invocation_index] = local_invocation_id;
            globalResult[global_invocation_index] = global_invocation_id;
        }
    `;

    const module = device.createShaderModule({
        label: 'compute shader module',
        code: indexShader
    });
    const computePipeline = device.createComputePipeline({
        label: 'compute pipeline',
        layout: 'auto',
        compute: {
            module,
            entryPoint: 'indexCompute'
        }
    });
    const size = numResults * 4 * 4;
    let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const workgroupBuffer = device.createBuffer({size, usage});
    const localBuffer = device.createBuffer({size, usage});
    const globalBuffer = device.createBuffer({size, usage});
    usage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    const workgroupReadBuffer = device.createBuffer({size, usage});
    const localReadBuffer = device.createBuffer({size, usage});
    const globalReadBuffer = device.createBuffer({size, usage});
    const bindGroup = device.createBindGroup({
        label: 'compute work buffer bind group',
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: workgroupBuffer }},
            { binding: 1, resource: { buffer: localBuffer }},
            { binding: 2, resource: { buffer: globalBuffer }}
        ]
    });

    const compute = async () => {
        const encoder = device.createCommandEncoder({ label: 'compute command encoder '});

        const pass = encoder.beginComputePass({ label: 'compute pass' });
        pass.setPipeline(computePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(...dispatchCount);
        pass.end();

        encoder.copyBufferToBuffer(workgroupBuffer, 0, workgroupReadBuffer, 0, size);
        encoder.copyBufferToBuffer(localBuffer, 0, localReadBuffer, 0, size);
        encoder.copyBufferToBuffer(globalBuffer, 0, globalReadBuffer, 0, size);

        device.queue.submit([encoder.finish()]);

        await Promise.all([
            workgroupReadBuffer.mapAsync(GPUMapMode.READ),
            localReadBuffer.mapAsync(GPUMapMode.READ),
            globalReadBuffer.mapAsync(GPUMapMode.READ)
        ]);
        const workgroup = new Uint32Array(workgroupReadBuffer.getMappedRange());
        const local = new Uint32Array(localReadBuffer.getMappedRange());
        const global = new Uint32Array(globalReadBuffer.getMappedRange());

        const get3 = (arr, i) => {
            const off = i * 4;
            return `${arr[off]}, ${arr[off + 1]}, ${arr[off + 2]}`;
        };
        
        for (let i = 0; i < numResults; ++i) {
            if (i % numThreadsPerWorkgroup === 0) {
                log(`\
---------------------------------------
global                 local     global   dispatch: ${i / numThreadsPerWorkgroup}
invoc.    workgroup    invoc.    invoc.
index     id           id        id
---------------------------------------`);
            }
            log(` ${i.toString().padStart(3)}:      ${get3(workgroup, i)}      ${get3(local, i)}   ${get3(global, i)}`)
        }
        
        function log(...args) {
          console.log(args)
        }

        workgroupReadBuffer.unmap();
        localReadBuffer.unmap();
        globalReadBuffer.unmap();
    }


    const render = (t) => {
        const aspect = canvas.width / canvas.height;
    }

    const observer = new ResizeObserver(entries => {
        entries.forEach(entry => {
            const canvas = entry.target;
            const width = entry.contentBoxSize[0].inlineSize;
            const height = entry.contentBoxSize[0].blockSize;
            canvas.width = Math.min(width, device.limits.maxTextureDimension2D);
            canvas.height = Math.min(height, device.limits.maxTextureDimension2D);
            // re-render
            render(0);
        });
    });
    observer.observe(canvas);

    const animate = (t) => {
        render(t);
        requestAnimationFrame(t => animate(t));
    };
    animate();

    document.body.style.overflow =  'auto ';
    canvas.style.display = 'none';

    await compute();
}


start();