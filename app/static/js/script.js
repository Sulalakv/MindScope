
const videoInput = document.getElementById("video");
const uploadZone = document.getElementById("upload-zone");
const uploadContent = document.getElementById("upload-content");
const processingContent = document.getElementById("processing-content");
const videoName = document.getElementById("video-name");
const uploadForm = document.getElementById("upload-form");
const submitButton = document.getElementById("submit-button");
const buttonLabel = document.getElementById("button-label");

const MAX_FILE_SIZE = 100 * 1024 * 1024;
const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov"];

function getExtension(filename) {
    const dotIndex = filename.lastIndexOf(".");
    return dotIndex >= 0 ? filename.slice(dotIndex).toLowerCase() : "";
}

function showFile(file) {
    if (!file || !videoName || !uploadZone) return;

    const extension = getExtension(file.name);

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
        videoInput.value = "";
        videoName.textContent = "Unsupported format. Choose MP4, AVI or MOV.";
        uploadZone.classList.remove("has-file");
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        videoInput.value = "";
        videoName.textContent = "File exceeds the 100 MB limit.";
        uploadZone.classList.remove("has-file");
        return;
    }

    videoName.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    uploadZone.classList.add("has-file");
}

if (videoInput) {
    videoInput.addEventListener("change", () => {
        showFile(videoInput.files?.[0]);
    });
}

if (uploadZone && videoInput) {
    uploadZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadZone.classList.add("is-dragging");
    });

    uploadZone.addEventListener("dragleave", (event) => {
        if (!uploadZone.contains(event.relatedTarget)) {
            uploadZone.classList.remove("is-dragging");
        }
    });

    uploadZone.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadZone.classList.remove("is-dragging");

        const file = event.dataTransfer?.files?.[0];

        if (!file) return;

        const extension = getExtension(file.name);

        if (!ALLOWED_EXTENSIONS.includes(extension)) {
            showFile(file);
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            showFile(file);
            return;
        }

        try {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            videoInput.files = transfer.files;
            showFile(file);
        } catch (error) {
            console.error("Unable to attach dropped file:", error);
            videoName.textContent = "Please use the Choose a video button.";
        }
    });
}

if (uploadForm && videoInput && submitButton && buttonLabel) {
    uploadForm.addEventListener("submit", (event) => {
        const file = videoInput.files?.[0];

        if (!file) {
            event.preventDefault();
            videoInput.reportValidity();
            return;
        }

        const extension = getExtension(file.name);

        if (!ALLOWED_EXTENSIONS.includes(extension)) {
            event.preventDefault();
            showFile(file);
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            event.preventDefault();
            showFile(file);
            return;
        }

        submitButton.disabled = true;
        buttonLabel.textContent = "Processing video…";

        if (uploadContent && processingContent) {
            uploadContent.hidden = true;
            processingContent.hidden = false;
        }

        if (uploadZone) {
            uploadZone.classList.remove("is-dragging", "has-file");
        }
    });
}
