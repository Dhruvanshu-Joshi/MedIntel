// // Display uploaded image
// document.getElementById('imageUpload').addEventListener('change', function(event) {
//     const file = event.target.files[0];
//     if (file) {
//         const reader = new FileReader();
//         reader.onload = function(e) {
//             document.getElementById('imagePreview').innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
//         };
//         reader.readAsDataURL(file);
//     }
// });

// // Extract text and show the processed image
// function extractText() {
//     const formData = new FormData(document.getElementById('uploadForm'));

//     // Call the backend for image processing
//     fetch('/upload', {
//         method: 'POST',
//         body: formData,
//     })
//     .then(response => response.blob())
//     .then(blob => {
//         const imageUrl = URL.createObjectURL(blob);
//         document.getElementById('processedImage').src = imageUrl;
//         document.getElementById('processedImage').style.display = 'block';
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// }
document.addEventListener("DOMContentLoaded", () => {
    const handwrittenLink = document.getElementById("handwritten");
    const contentArea = document.getElementById("content");

    handwrittenLink.addEventListener("click", (event) => {
        event.preventDefault(); // Prevent default anchor behavior
        loadContent('/handwritten'); // Load the handwritten.html content
    });

    function loadContent(url) {
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok ' + response.statusText);
                }
                return response.text();
            })
            .then(html => {
                contentArea.innerHTML = html; // Inject the fetched HTML into the content area
            })
            .catch(error => {
                console.error('There was a problem with the fetch operation:', error);
            });
    }
});
