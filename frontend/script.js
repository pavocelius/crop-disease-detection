// Point this at your deployed FastAPI backend
const API_URL = "http://127.0.0.1:8000/predict"; // replace with your Render URL after deployment

const fileInput = document.getElementById('fileInput');
const fileDrop = document.getElementById('fileDrop');
const fileDropLabel = document.getElementById('fileDropLabel');
const preview = document.getElementById('preview');
const uploadStatus = document.getElementById('uploadStatus');
const heroUploadBtn = document.getElementById('heroUploadBtn');

const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent = document.getElementById('resultContent');
const diseaseName = document.getElementById('diseaseName');
const confidenceText = document.getElementById('confidenceText');
const severityText = document.getElementById('severityText');
const progressFill = document.getElementById('progressFill');
const recommendationList = document.getElementById('recommendationList');

// Recommendations by keyword — mirrors the treatment table in the project report
function recommendationsFor(name) {
  const n = name.toLowerCase();
  if (n.includes('healthy')) {
    return ["No disease detected — keep up your current care routine.", "Continue checking leaves weekly for early signs of trouble."];
  }
  if (n.includes('virus') || n.includes('mosaic') || n.includes('curl')) {
    return ["Remove and destroy infected plants to stop the spread.", "Control whiteflies and aphids, which commonly carry plant viruses.", "Use resistant seed varieties for future planting."];
  }
  if (n.includes('bacterial') || n.includes('canker')) {
    return ["Prune and remove infected leaves or branches.", "Avoid overhead watering — water the soil, not the leaves.", "Apply a copper-based bactericide spray."];
  }
  return ["Remove infected leaves and dispose of them away from the field.", "Improve air circulation by spacing plants further apart.", "Apply a fungicide suited to this crop.", "Avoid overhead watering, which helps fungus spread."];
}

function severityFor(confidencePct) {
  if (confidencePct >= 90) return "High";
  if (confidencePct >= 70) return "Moderate";
  return "Low — consider a clearer photo to confirm";
}

function prettyName(raw) {
  return raw.replace(/_+/g, ' ').replace(/\s+/g, ' ').trim()
    .replace(/\b\w/g, c => c.toUpperCase());
}

heroUploadBtn.addEventListener('click', () => {
  document.getElementById('detector').scrollIntoView({ behavior: 'smooth' });
  fileInput.click();
});

fileDrop.addEventListener('click', () => fileInput.click());
fileDrop.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

['dragover', 'dragleave', 'drop'].forEach(evt => fileDrop.addEventListener(evt, e => e.preventDefault()));
fileDrop.addEventListener('dragover', () => fileDrop.classList.add('dragging'));
fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('dragging'));
fileDrop.addEventListener('drop', e => {
  fileDrop.classList.remove('dragging');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.match(/image\/(jpeg|png)/)) {
    uploadStatus.textContent = "Please upload a JPG or PNG image.";
    return;
  }
  uploadStatus.textContent = '';

  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    fileDropLabel.style.display = 'none';
  };
  reader.readAsDataURL(file);

  submitForDiagnosis(file);
}

async function submitForDiagnosis(file) {
  uploadStatus.textContent = "Analyzing image…";
  resultPlaceholder.style.display = 'block';
  resultPlaceholder.textContent = "Analyzing…";
  resultContent.style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(API_URL, { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Server error ' + res.status);
    const data = await res.json();
    showResult(data);
    uploadStatus.textContent = '';
  } catch (err) {
    uploadStatus.textContent = "Couldn't reach the detection server. Check your connection and try again.";
    resultPlaceholder.textContent = "Upload a leaf image to see the diagnosis here.";
  }
}

function showResult(data) {
  const pct = Math.round(data.confidence * 100);
  // Prefer the backend's own status field (Healthy/Diseased) if present
  const isHealthy = data.status ? data.status === "Healthy" : /healthy/i.test(data.disease);

  resultPlaceholder.style.display = 'none';
  resultContent.style.display = 'block';

  diseaseName.textContent = isHealthy ? "Healthy Leaf" : prettyName(data.disease);
  confidenceText.textContent = `Confidence: ${pct}%`;
  severityText.textContent = isHealthy ? "Severity: None" : `Severity: ${severityFor(pct)}`;
  progressFill.style.width = pct + '%';

  const tips = recommendationsFor(data.disease);
  recommendationList.innerHTML = tips.map(t => `<li>${t}</li>`).join('');
}

function displayResult(data){

    document.getElementById("diseaseName").innerText =
        data.disease;

    document.getElementById("confidence").innerText =
        "Confidence: " + data.confidence + "%";

    document.getElementById("severity").innerText =
        "Severity: " + data.severity;

    const treatmentList =
        document.getElementById("treatmentList");

    treatmentList.innerHTML = "";

    data.treatment.forEach(item => {

        const li = document.createElement("li");

        li.textContent = item;

        treatmentList.appendChild(li);

    });

    const improvementList =
        document.getElementById("improvementList");

    improvementList.innerHTML = "";

    data.improvements.forEach(item => {

        const li = document.createElement("li");

        li.textContent = item;

        improvementList.appendChild(li);

    });

}