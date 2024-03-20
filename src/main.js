// view elements
import {App} from './app/app.js';

const noWebGPUMessage = document.querySelector('#no-webgpu')

async function init() {
    // check compatibility
    const hasWebGPU = !!navigator.gpu;
    if (!hasWebGPU) {
        noWebGPUMessage.style.display = '';
        return;
    }

    // init tweakpane

    await App.init();
}

init();
