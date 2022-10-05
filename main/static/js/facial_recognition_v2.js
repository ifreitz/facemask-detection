// https://itnext.io/face-api-js-javascript-api-for-face-recognition-in-the-browser-with-tensorflow-js-bcc2a6c4cf07

window.addEventListener('DOMContentLoaded', run, false);
var labeledFaceDescriptors;
var faceMatcher;

async function run() {
    // load the models
    const MODEL_URL = '/static/models/'

    await faceapi.loadSsdMobilenetv1Model(MODEL_URL)
    await faceapi.loadFaceLandmarkModel(MODEL_URL)
    await faceapi.loadFaceRecognitionModel(MODEL_URL)
    console.log("Model loaded...")

    // load_known_faces()

    // try to access users webcam and stream the images
    // to the video element
    const videoEl = document.getElementById('inputVideo')
    navigator.getUserMedia(
        { video: {} },
        stream => videoEl.srcObject = stream,
        err => console.error(err)
    )

    videoEl.onplay = (event) => {
        onPlay(videoEl)
    };
}

async function load_known_faces() {
    const known_labels = [
        "IFD",
        "IFD",
        "IFD",
        "IFD",
        "JEGS",
        "JEGS",
        "RDC",
        "RDC",
    ]
    const known_images = [
        "/static/img/face_data/ifd1.jpg",
        "/static/img/face_data/ifd2.jpg",
        "/static/img/face_data/ifd3.jpg",
        "/static/img/face_data/ifd4.jpg",
        "/static/img/face_data/jegs1.jpg",
        "/static/img/face_data/jegs2.jpg",
        "/static/img/face_data/rdc1.jpg",
        "/static/img/face_data/rdc2.jpg",
    ]

    labeledFaceDescriptors = await Promise.all(
        known_images.map(async (imgUrl, index) => {
            // fetch image data from urls and convert blob to HTMLImage element
            const img = await faceapi.fetchImage(imgUrl)
            const label = known_labels[index]

            // detect the face with the highest score in the image and compute it's landmarks and face descriptor
            const fullFaceDescription = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()

            if (!fullFaceDescription) {
                throw new Error(`no faces detected for ${label}`)
            }

            const faceDescriptors = [fullFaceDescription.descriptor]
            return new faceapi.LabeledFaceDescriptors(label, faceDescriptors)
        })
    )

    const maxDescriptorDistance = 0.6
    faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance)
}

async function detect(videoEl) {
    console.log("Detecting...")
    const input = document.getElementById('inputVideo')
    const canvas = document.getElementById('overlay')
    const dims = faceapi.matchDimensions(canvas, videoEl, true)
    console.log(dims)
    
    let fullFaceDescriptions = await faceapi.detectAllFaces(input).withFaceLandmarks().withFaceDescriptors()
    console.log("2.")
    fullFaceDescriptions = faceapi.resizeResults(fullFaceDescriptions, dims)
    console.log("3.")
    faceapi.draw.drawDetections(canvas, fullFaceDescriptions)
    console.log("4.")
    
    // const results = fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor))
    // results.forEach((bestMatch, i) => {
    //     const box = fullFaceDescriptions[i].detection.box
    //     const text = bestMatch.toString()
    //     const drawBox = new faceapi.draw.DrawBox(box, { label: text })
    //     drawBox.draw(canvas)
    // })
}

async function onPlay(videoEl) {
    await detect(videoEl)
    setTimeout(() => onPlay(videoEl))
}