async function processImage() {
  const fileInput = document.getElementById("imageInput");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image first!");
    return;
  }

  // Show loading indicator
  document.getElementById("loading").style.display = "block";

  // Prepare form data
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict/", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();

    // Display results
    const inputImage = document.getElementById("inputImage");
    const outputImage = document.getElementById("outputImage");

    inputImage.src = data.input_image;
    outputImage.src = data.output_image;

    inputImage.style.display = "block";
    outputImage.style.display = "block";
  } catch (error) {
    console.error("Error:", error);
    alert("Error processing image");
  } finally {
    // Hide loading indicator
    document.getElementById("loading").style.display = "none";
  }
}
