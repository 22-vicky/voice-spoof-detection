document.addEventListener("DOMContentLoaded", function () {
    // Get elements safely
    const button = document.getElementById("checkBtn");
    const fileInput = document.getElementById("audioInput");
    const resultDiv = document.getElementById("result");

    if (!button || !fileInput || !resultDiv) {
        console.error("One or more required elements are missing!");
        return;
    }

    button.addEventListener("click", async function () {
        const file = fileInput.files?.[0]; // optional chaining prevents crash
        if (!file) {
            resultDiv.style.display = "block";
            resultDiv.className = "result error";
            resultDiv.innerHTML = "Please select an audio file.";
            return;
        }

        const formData = new FormData();
        formData.append("audio", file);

        resultDiv.style.display = "block";
        resultDiv.className = "result";
        resultDiv.innerHTML = "Processing...";

        try {
            const response = await fetch("/predict/", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                resultDiv.className = "result error";
                resultDiv.innerHTML = "Error: " + data.error;
            } else {
                const cls = data.prediction.toLowerCase() === "genuine" ? "genuine" : "spoofed";
                resultDiv.className = "result " + cls;
                resultDiv.innerHTML =
                    `<b>Prediction:</b> ${data.prediction}<br>` +
                    `<b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%`;
            }

        } catch (error) {
            console.error(error);
            resultDiv.className = "result error";
            resultDiv.innerHTML = "Server connection failed!";
        }
    });
});

