var video;
var tfliteModel;
var videoWidth = 320;
var videoHeight = 0;
var labels = ["With Mask", "Without Mask", "Wearing improperly"]

const createDetectionResultBox = (left, top, width, height, name, score) => {
    score = score * 100;

    let container = document.createElement("div");
    container.classList.add("box-container");

    let box = document.createElement("div");

    let label = document.createElement("div");
    label.classList.add("label");
    label.textContent = `${name} (${score.toFixed(2)}%)`;

    if (name === "With Mask") {
        box.classList.add("box-with-mask");
        label.classList.add("label-with-mask");
    } else if (name === "Without Mask") {
        box.classList.add("box-without-mask");
        label.classList.add("label-without-mask");
    } else {
        box.classList.add("box-wear-imporper");
        label.classList.add("label-wear-improper");
    }

    container.appendChild(label);
    container.appendChild(box);

    container.style.right = `${left - 1}px`;
    container.style.top = `${top - 1}px`;
    box.style.width = `${width + 1}px`;
    box.style.height = `${height + 1}px`;

    return container;
}

const renderDetectionResult = (boxes, classes, scores, n) => {
    let boxesContainer = document.querySelector(".boxes-container");
    boxesContainer.innerHTML = "";
    boxesContainer.style.top = `${video.getBoundingClientRect().y}px`;
    boxesContainer.style.left = `${video.getBoundingClientRect().x}px`;
    boxesContainer.style.width = `${videoWidth}px`;
    boxesContainer.style.height = `${videoHeight}px`;

    for (let i = 0; i < n; i++) {
        let boundingBox = boxes.slice(i*4, (i+1)*4);
        let name = labels[classes[i]];
        let score = scores[i];
        let y_min = Math.floor(boundingBox[0] * videoHeight);
        let y_max = Math.floor(boundingBox[2] * videoHeight);
        let x_min = Math.floor(boundingBox[1] * videoWidth);
        let x_max = Math.floor(boundingBox[3] * videoWidth);

        if (score > 0.5) {
            let boxContainer = createDetectionResultBox(
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min,
                name,
                score
            );
            boxesContainer.appendChild(boxContainer);
        } 
    }
}

async function predict(input) {
    let prediction = await tfliteModel.predict(input);
    let result = {};

    result["n"] = Array.from(await prediction["StatefulPartitionedCall:0"].data())
    result["scores"] = Array.from(await prediction["StatefulPartitionedCall:1"].data())
    result["classes"] = Array.from(await prediction["StatefulPartitionedCall:2"].data())
    result["boxes"] = Array.from(await prediction["StatefulPartitionedCall:3"].data());

    renderDetectionResult(result["boxes"], result["classes"], result["scores"], result["n"])
}

async function captureStream(){
    let input = tf.expandDims(tf.browser.fromPixels(video), 0);
    input = tf.image.resizeBilinear(input, [320, 320]);
    input = input.cast("int32");
    await predict(input);

    window.requestAnimationFrame(captureStream);
}

async function start(){
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
        video.srcObject = stream;
        video.play();
    })
    .catch((err) => {
        console.log("An error occurred: " + err);
    });
    
    video.addEventListener('canplay', () => {
        videoHeight = video.videoHeight;
        videoWidth = video.videoWidth;
        window.requestAnimationFrame(captureStream);
    }, false);
}

async function loadModel(){
    tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/');
    tfliteModel = await tflite.loadTFLiteModel('/static/facemask_detection.tflite');
    console.log("Model loaded: ", tfliteModel)

    start();
}

document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    loadModel();
 }, false);