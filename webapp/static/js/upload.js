const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileListEl = document.getElementById('file-list');
const emptyNote = document.getElementById('empty-note');
const paperCount = document.getElementById('paper-count');
const analyzeBtn = document.getElementById('analyze-btn');
const analyzeStatus = document.getElementById('analyze-status');
const resultsSummary = document.getElementById('results-summary');

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function refreshPapers() {
  const res = await fetch('/api/papers');
  const data = await res.json();
  renderPapers(data.papers);
  updateAnalyzeLabel(data.has_results);
}

function renderPapers(papers) {
  fileListEl.innerHTML = '';
  paperCount.textContent = `${papers.length} uploaded`;
  emptyNote.style.display = papers.length ? 'none' : 'block';
  papers.forEach(p => {
    const row = document.createElement('div');
    row.className = 'file-row';
    row.innerHTML = `
      <span class="file-icon">PDF</span>
      <span class="file-name">${p.filename}</span>
      <span class="file-size">${fmtSize(p.size_bytes)}</span>
      <button class="file-remove" title="Remove" data-id="${p.paper_id}">&times;</button>
    `;
    fileListEl.appendChild(row);
  });
  fileListEl.querySelectorAll('.file-remove').forEach(btn => {
    btn.addEventListener('click', async () => {
      await fetch(`/api/papers/${btn.dataset.id}`, { method: 'DELETE' });
      refreshPapers();
    });
  });
}

async function uploadFiles(fileArr) {
  const pdfs = fileArr.filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
  if (!pdfs.length) return;
  const fd = new FormData();
  pdfs.forEach(f => fd.append('files', f));
  await fetch('/api/papers/upload', { method: 'POST', body: fd });
  refreshPapers();
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => uploadFiles(Array.from(fileInput.files)));

['dragenter', 'dragover'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('dragover'); })
);
dropzone.addEventListener('drop', e => uploadFiles(Array.from(e.dataTransfer.files)));

// ---------- tag inputs ----------
function setupTagInput(boxId, inputId, tagClass) {
  const box = document.getElementById(boxId);
  const input = document.getElementById(inputId);
  let tags = [];

  function render() {
    box.querySelectorAll('.tag').forEach(t => t.remove());
    tags.forEach((tag, i) => {
      const el = document.createElement('span');
      el.className = 'tag' + (tagClass ? ' ' + tagClass : '');
      el.innerHTML = `${tag}<button data-i="${i}">&times;</button>`;
      box.insertBefore(el, input);
    });
    box.querySelectorAll('.tag button').forEach(btn => {
      btn.addEventListener('click', () => {
        tags.splice(Number(btn.dataset.i), 1);
        render();
        saveTargets();
      });
    });
  }

  input.addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ',') && input.value.trim()) {
      e.preventDefault();
      const val = input.value.trim().replace(/,$/, '');
      if (val && !tags.includes(val)) tags.push(val);
      input.value = '';
      render();
      saveTargets();
    } else if (e.key === 'Backspace' && !input.value && tags.length) {
      tags.pop();
      render();
      saveTargets();
    }
  });

  return { get: () => tags, set: (v) => { tags = v || []; render(); } };
}

const elasticityTags = setupTagInput('elasticity-box', 'elasticity-input', '');
const productTags = setupTagInput('product-box', 'product-input', 'product-tag');

let saveTimer = null;
function saveTargets() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    fetch('/api/targets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ elasticities: elasticityTags.get(), products: productTags.get() }),
    });
  }, 300);
}

async function loadTargets() {
  const res = await fetch('/api/targets');
  const data = await res.json();
  elasticityTags.set(data.elasticities || []);
  productTags.set(data.products || []);
}

// ---------- analyze ----------
function updateAnalyzeLabel(hasResults) {
  analyzeBtn.textContent = hasResults ? 'Update analysis' : 'Analyze';
}

analyzeBtn.addEventListener('click', async () => {
  const wasUpdate = analyzeBtn.textContent.trim() === 'Update analysis';
  analyzeBtn.disabled = true;
  analyzeStatus.textContent = 'Parsing and filtering corpus…';
  analyzeStatus.classList.add('busy');
  resultsSummary.classList.remove('show');
  try {
    const res = await fetch('/api/analyze', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      analyzeStatus.textContent = data.error || 'Analysis failed.';
      analyzeStatus.classList.remove('busy');
      analyzeBtn.disabled = false;
      return;
    }
    document.getElementById('s-papers').textContent = data.papers;
    document.getElementById('s-failed').textContent = data.papers_failed;
    document.getElementById('s-chunks').textContent = data.chunks_total;
    document.getElementById('s-kept').textContent =
      `${data.chunks_kept} (${data.chunks_total ? Math.round(100 * data.chunks_kept / data.chunks_total) : 0}%)`;
    resultsSummary.classList.add('show');
    analyzeStatus.textContent = wasUpdate
      ? 'Done. Downstream Cheap AI / Detection AI results were cleared — re-run them from their pages.'
      : 'Done.';
    analyzeStatus.classList.remove('busy');
    updateAnalyzeLabel(true);
  } catch (e) {
    analyzeStatus.textContent = 'Analysis failed — check the server log.';
    analyzeStatus.classList.remove('busy');
  }
  analyzeBtn.disabled = false;
});

refreshPapers();
loadTargets();
