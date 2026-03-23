console.log("Realtime JS Loaded");

let mediaRecorder;
let audioChunks = [];
let timer;
let seconds = 0;

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const resultDiv = document.getElementById("result");

recordBtn.addEventListener("click", async () => {

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    seconds = 0;

    mediaRecorder.start();

    recordBtn.disabled = true;
    stopBtn.disabled = false;

    resultDiv.style.display = "block";
    resultDiv.innerHTML = "Recording... 0s";

    timer = setInterval(() => {
        seconds++;
        resultDiv.innerHTML = "Recording... " + seconds + "s";
    }, 1000);

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };
});

stopBtn.addEventListener("click", () => {

    mediaRecorder.stop();
    clearInterval(timer);

    recordBtn.disabled = false;
    stopBtn.disabled = true;

    mediaRecorder.onstop = async () => {

        resultDiv.innerHTML = "Processing...";

        const blob = new Blob(audioChunks, { type: "audio/webm" });

        const formData = new FormData();
        formData.append("audio", blob, "recording.webm");

        const response = await fetch("/predict/", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = "Error: " + data.error;
        } else {
            resultDiv.innerHTML =
                "<b>Prediction:</b> " + data.prediction + "<br>" +
                "<b>Confidence:</b> " + (data.confidence * 100).toFixed(2) + "%";
        }
    };
});
