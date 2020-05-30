// ES2015

import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

let webcam;
let model;

const square = 8; // 8*8 = 64

// Initializing the application
async function init() {
    try {

        createCanvases();
        webcam = await tfd.webcam(document.getElementById('webcam'));
        await loadModel();
        const prediction = await getPrediction();
        await displayFilters(prediction);

    } catch (e) {

        console.log(e);

    }
}

// Display convolution filters in HTML5 canvases
// prediction [None, 126, 126, 64]
async function displayFilters(prediction) {

    let index = 0;
    for(let i = 0; i < square; i++) {
        for(let j = 0; j < square; j++) {

            // [None, 126, 126, 64]
            // ↓
            const filter = tf.squeeze(
                tf.split(prediction, 64, 3)[index],
                0
            );
            // ↓
            // [126, 126, index] [-1.0, +1.0]
            // ↓
            const filterMapped = tf.add(tf.mul(filter, 0.5), 0.5);
            // ↓
            // [126, 126, index] [0.0, 1.0]

            const canvas = document.getElementById(`output-${index}`);
            await tf.browser.toPixels(filterMapped, canvas);
            
            index++;

        }
        document.body.appendChild(
            document.createElement('br'),
        );
    }

}

// Creating HTML5 canvas elements to display convolution filters
function createCanvases() {

    let index = 0;
    for(let i = 0; i < square; i++) {
        for(let j = 0; j < square; j++) {
            const canvas = document.createElement('canvas');
            canvas.width = 126;
            canvas.height = 126;
            canvas.id = `output-${index}`;
            document.body.appendChild(canvas);
            index++;
        }
        document.body.appendChild(
            document.createElement('br'),
        );
    }

}

// Loaded the TensorFlow.js model
// (which was converted from a Keras model.h5)
async function loadModel() {

    model = await tf.loadLayersModel('http://127.0.0.1:8080/model.json');
    model.summary();

}

// Capture webcam's current frame
async function getImage() {
    const img = await webcam.capture();
    const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(255));
    img.dispose();
    return processedImg;
}

// Predict filters with current webcam frame
async function getPrediction() {

    // Capture current webcam's frame
    const img = await getImage();

    // Predict
    const prediction = model.predict(img);
    console.log(`prediction.shape: ${prediction.shape}`);

    return prediction;
}

init();