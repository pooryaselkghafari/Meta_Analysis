const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileListEl = document.getElementById('file-list');
const emptyNote = document.getElementById('empty-note');
const paperCount = document.getElementById('paper-count');
const analyzeBtn = document.getElementById('analyze-btn');
const pauseBtn = document.getElementById('pause-btn');
const stopBtn = document.getElementById('stop-btn');
const analyzeStatus = document.getElementById('analyze-status');
const analyzeProgress = document.getElementById('analyze-progress');
const analyzeProgressFill = document.getElementById('analyze-progress-fill');
const analyzeProgressLabel = document.getElementById('analyze-progress-label');
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

// ---------- analyze (durable job + pause/stop) ----------
function updateAnalyzeLabel(hasResults) {
  analyzeBtn.textContent = hasResults ? 'Update analysis' : 'Analyze';
}

function showProgressBar() {
  analyzeProgress.hidden = false;
  analyzeProgressFill.style.width = '0%';
  analyzeProgressLabel.textContent = 'Starting…';
}

function hideProgressBar() {
  analyzeProgress.hidden = true;
}

function setPauseButton(paused) {
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  pauseBtn.classList.toggle('is-paused', paused);
  pauseBtn.disabled = false;
}

function shortName(id) {
  if (!id) return '';
  return id.length > 42 ? id.slice(0, 40) + '…' : id;
}

function applyAnalyzeProgress(p) {
  const done = p.completed ?? 0;
  const total = p.total || 0;
  const pct = total ? Math.min(100, Math.round(100 * done / total)) : 0;
  analyzeProgressFill.style.width = `${pct}%`;
  setPauseButton(!!p.paused);

  const nPapers = p.papers_total || Math.ceil(total / 2) || 0;
  const idx = p.paper_index || 0;
  const name = shortName(p.paper_id);
  const paperBit = name
    ? (idx ? `paper ${idx}/${nPapers}: "${name}"` : `"${name}"`)
    : (nPapers ? `${nPapers} papers` : '');

  let phaseLabel = 'Working';
  if (p.phase === 'parsing') phaseLabel = 'Parsing';
  else if (p.phase === 'chunking') phaseLabel = 'Building chunks';
  else if (p.phase === 'filtering') phaseLabel = 'Filtering';
  else if (p.phase === 'writing') phaseLabel = 'Writing results';
  else if (p.phase === 'done') phaseLabel = 'Finishing';

  let label = `${phaseLabel}${paperBit ? ` — ${paperBit}` : ''} — ${pct}%`;
  if (p.paused) label = `Paused — ${label}`;
  analyzeProgressLabel.textContent = label;

  // Keep the status line for final/error messages only — don't duplicate
  // the progress label while a run is in flight.
  analyzeStatus.textContent = '';
  analyzeStatus.classList.toggle('busy', !p.paused && !!p.running);
}

function showSummaryFromProgress(p) {
  document.getElementById('s-papers').textContent = p.papers_ok != null ? p.papers_ok : '—';
  document.getElementById('s-failed').textContent = p.papers_failed != null ? p.papers_failed : '—';
  document.getElementById('s-chunks').textContent = p.chunks_total != null ? p.chunks_total : '—';
  const kept = p.chunks_kept;
  const total = p.chunks_total;
  document.getElementById('s-kept').textContent =
    kept != null
      ? `${kept}${total ? ` (${Math.round(100 * kept / total)}%)` : ''}`
      : '—';
  resultsSummary.classList.add('show');
}

function finishAnalyzeUi(finalProgress, wasUpdate) {
  hideProgressBar();
  pauseBtn.hidden = true;
  stopBtn.hidden = true;
  stopBtn.disabled = false;
  if (finalProgress.stopped) {
    analyzeStatus.textContent = finalProgress.error
      || `Stopped — ${finalProgress.completed || 0}/${finalProgress.total || 0} papers done. Previous corpus left unchanged.`;
  } else if (finalProgress.error) {
    analyzeStatus.textContent = `${finalProgress.error} (${finalProgress.completed || 0}/${finalProgress.total || 0} papers before the error).`;
  } else {
    showSummaryFromProgress(finalProgress);
    const warnings = (finalProgress.chunk_warnings || []).filter(Boolean);
    analyzeStatus.textContent = wasUpdate
      ? 'Done. Downstream Cheap AI / Detection AI results were cleared — re-run them from their pages.'
      : 'Done.';
    if (warnings.length) {
      analyzeStatus.textContent += ' ' + warnings.join(' ');
    }
    updateAnalyzeLabel(true);
  }
  analyzeStatus.classList.remove('busy');
  analyzeBtn.disabled = false;
}

async function pollUntilAnalyzeDone() {
  for (;;) {
    let p;
    try {
      const res = await fetch('/api/analyze/progress');
      p = await res.json();
    } catch (e) {
      await new Promise(r => setTimeout(r, 1500));
      continue;
    }
    applyAnalyzeProgress(p);
    if (!p.running) return p;
    await new Promise(r => setTimeout(r, 1500));
  }
}

stopBtn.addEventListener('click', async () => {
  if (!confirm('Stop this run? The previous corpus (if any) is kept; this run will not finish writing new results.')) return;
  stopBtn.disabled = true;
  pauseBtn.disabled = true;
  try {
    const res = await fetch('/api/analyze/stop', { method: 'POST' });
    const p = await res.json();
    if (!res.ok) {
      alert(p.error || 'Could not stop the job.');
      stopBtn.disabled = false;
      pauseBtn.disabled = false;
    }
  } catch (e) {
    stopBtn.disabled = false;
    pauseBtn.disabled = false;
  }
});

pauseBtn.addEventListener('click', async () => {
  pauseBtn.disabled = true;
  const isPaused = pauseBtn.classList.contains('is-paused');
  const endpoint = isPaused ? '/api/analyze/resume' : '/api/analyze/pause';
  try {
    const res = await fetch(endpoint, { method: 'POST' });
    const p = await res.json();
    if (res.ok) applyAnalyzeProgress(p);
    else pauseBtn.disabled = false;
  } catch (e) {
    pauseBtn.disabled = false;
  }
});

analyzeBtn.addEventListener('click', async () => {
  const wasUpdate = analyzeBtn.textContent.trim() === 'Update analysis';
  analyzeBtn.disabled = true;
  analyzeStatus.textContent = 'Starting corpus analysis…';
  analyzeStatus.classList.add('busy');
  resultsSummary.classList.remove('show');
  showProgressBar();
  pauseBtn.hidden = false;
  stopBtn.hidden = false;
  setPauseButton(false);
  try {
    const res = await fetch('/api/analyze', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      if (res.status === 409) {
        const finalProgress = await pollUntilAnalyzeDone();
        finishAnalyzeUi(finalProgress, wasUpdate);
        await refreshPapers();
        return;
      }
      hideProgressBar();
      pauseBtn.hidden = true;
      stopBtn.hidden = true;
      analyzeStatus.textContent = data.error || 'Analysis failed.';
      analyzeStatus.classList.remove('busy');
      analyzeBtn.disabled = false;
      return;
    }
    const finalProgress = await pollUntilAnalyzeDone();
    finishAnalyzeUi(finalProgress, wasUpdate);
    await refreshPapers();
  } catch (e) {
    hideProgressBar();
    pauseBtn.hidden = true;
    stopBtn.hidden = true;
    analyzeStatus.textContent = 'Analysis failed — check the server log.';
    analyzeStatus.classList.remove('busy');
    analyzeBtn.disabled = false;
  }
});

async function resumeAnalyzeIfRunning() {
  try {
    const res = await fetch('/api/analyze/progress');
    const p = await res.json();
    if (!p.running) return;
    const wasUpdate = true;
    analyzeBtn.disabled = true;
    showProgressBar();
    pauseBtn.hidden = false;
    stopBtn.hidden = false;
    applyAnalyzeProgress(p);
    const finalProgress = await pollUntilAnalyzeDone();
    finishAnalyzeUi(finalProgress, wasUpdate);
    await refreshPapers();
  } catch (e) {
    // ignore
  }
}

refreshPapers();
loadTargets();
resumeAnalyzeIfRunning();
