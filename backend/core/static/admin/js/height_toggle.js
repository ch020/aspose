document.addEventListener("DOMContentLoaded", function () {
  const heightInput = document.getElementById("id_height");
  const toggle = document.getElementById("toggle-imperial");
  const imperialFields = document.getElementById("imperial-fields");
  const feetInput = document.getElementById("feet-input");
  const inchesInput = document.getElementById("inches-input");

  if (!heightInput || !toggle || !imperialFields || !feetInput || !inchesInput) return;

  // Convert feet+inches to cm
  function updateHeightFromImperial() {
    const feet = parseFloat(feetInput.value) || 0;
    const inches = parseFloat(inchesInput.value) || 0;
    const totalInches = (feet * 12) + inches;
    const cm = totalInches * 2.54;
    heightInput.value = cm.toFixed(1);
  }

  toggle.addEventListener("change", () => {
    if (toggle.checked) {
      imperialFields.style.display = "block";
      feetInput.value = "";
      inchesInput.value = "";
      heightInput.readOnly = true;
    } else {
      imperialFields.style.display = "none";
      heightInput.readOnly = false;
    }
  });

  feetInput.addEventListener("input", updateHeightFromImperial);
  inchesInput.addEventListener("input", updateHeightFromImperial);
});