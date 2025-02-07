document.addEventListener("DOMContentLoaded", function () {
    // Event listener for file input
    document.getElementById("fileinput").addEventListener("change", handleFileUpload);

    // Handle image upload via drag-and-drop
    var dropZone = document.getElementById("drop-zone");
    dropZone.addEventListener("dragover", function(e) {
        e.preventDefault();
        dropZone.classList.add("dragging");
    });

    dropZone.addEventListener("dragleave", function(e) {
        e.preventDefault();
        dropZone.classList.remove("dragging");
    });

    dropZone.addEventListener("drop", function(e) {
        e.preventDefault();
        dropZone.classList.remove("dragging");

        // Handle file upload
        var file = e.dataTransfer.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // Handle file upload (either drag-drop or file input)
    function handleFileUpload(file) {
        var reader = new FileReader();
        reader.onload = function(e) {
            var image = new Image();
            image.onload = function() {
                // Show the uploaded image in the drop zone
                var imagePreviewContainer = document.getElementById("imagePreviewContainer");
                imagePreviewContainer.style.display = "block"; // Show preview container
                document.getElementById("imagePreview").src = e.target.result;
            };
            image.src = e.target.result; // Set the source to the file reader's result
        };
        reader.readAsDataURL(file); // Convert file to base64
    }

    // Handle the "Detect Disease" button click
    document.getElementById("send").addEventListener("click", function () {
        var image = document.getElementById("imagePreview").src;

        // Ensure the image is available
        if (image === "" || image === "data:,") {
            alert("Please upload an image first!");
            return;
        }

        // Convert the image to base64
        var base64Image = image.split(",")[1]; // The base64 string

        // Prepare the data to send to the server
        var data = {
            "image": base64Image
        };

        // Make an AJAX POST request to the /predict route
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            document.querySelector(".jsonRes").textContent = `Prediction Result: ${JSON.stringify(result)}`;
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred while processing the request.");
        });
    });
});
