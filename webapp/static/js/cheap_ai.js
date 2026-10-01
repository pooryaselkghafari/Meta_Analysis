const thumbList = document.getElementById('thumb-list');
const manuscript = document.getElementById('manuscript');
const classifyBtn = document.getElementById('classify-btn');
const classifyStatus = document.getElementById('classify-status');
const classifyProgress = document.getElementById('classify-progress');
const classifyProgressFill = document.getElementById('classify-progress-fill');
const classifyProgressLabel = document.getElementById('classify-progress-label');
const pauseBtn = document.getElementById('pause-btn');
const stopBtn = document.getElementById('stop-btn');
const nextStageBtn = document.getElementById('next-stage-btn');

let papers = [];
let activeId = null;

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const TYPE_LABELS = {
  regression_table: 'regression table',
  result_text: 'result text',
  methodology: 'methodology',
  data_description: 'data description',
  literature_review: 'literature review',
  other: 'other',
};

async function loadPapers() {
  const res = await fetch('/api/cheap-ai/papers');
  const data = await res.json();
  papers = data.papers;
  renderThumbs();
  if (papers.length) {
    selectPaper((papers.find(p => p.paper_id === activeId) || papers[0]).paper_id);
  } else {
    manuscript.innerHTML = `
      <div class="manuscript-empty">
        <div class="big">No kept chunks yet</div>
        <div>Run Analyze on the Corpus page first — this page only classifies chunks that already passed the heuristic filter.</div>
      </div>`;
  }
  await refreshStageState();
}

async function refreshStageState() {
  try {
    const res = await fetch('/api/pipeline/state');
    const state = await res.json();
    classifyBtn.textContent = state.cheap_ai_done ? 'Update classification' : 'Classify all kept chunks';
    nextStageBtn.disabled = !state.cheap_ai_done;
    nextStageBtn.title = state.cheap_ai_done ? '' : 'Classify all kept chunks first';
  } catch (e) {
    // leave button state as-is if the state endpoint is unreachable
  }
}

nextStageBtn.addEventListener('click', () => {
  if (!nextStageBtn.disabled) window.location.href = '/detection-ai';
});

function renderThumbs() {
  thumbList.innerHTML = '';
  papers.forEach(p => {
    const card = document.createElement('button');
    card.className = 'thumb-card' + (p.paper_id === activeId ? ' active' : '');
    card.dataset.id = p.paper_id;
    const pct = p.total_chunks ? Math.round(100 * p.classified_chunks / p.total_chunks) : 0;
    card.innerHTML = `
      <div class="thumb-img-wrap">
        <img src="/api/papers/${p.paper_id}/thumbnail" alt=""
             onerror="this.remove(); this.parentElement.innerHTML='<span class=&quot;fallback&quot;>&#128196;</span>'">
      </div>
      <div class="thumb-name">${escapeHtml(p.paper_id)}</div>
      <div class="thumb-stat">${p.classified_chunks}/${p.total_chunks} classified &middot; ${pct}%</div>
    `;
    card.addEventListener('click', () => selectPaper(p.paper_id));
    thumbList.appendChild(card);
  });
}

async function selectPaper(paperId) {
  activeId = paperId;
  renderThumbs();
  const res = await fetch(`/api/cheap-ai/papers/${paperId}/chunks`);
  const data = await res.json();
  renderManuscript(paperId, data.chunks);
}

function renderManuscript(paperId, chunks) {
  const meta = papers.find(p => p.paper_id === paperId) || { total_chunks: chunks.length, classified_chunks: 0 };
  const pct = meta.total_chunks ? Math.round(100 * meta.classified_chunks / meta.total_chunks) : 0;

  const chunkHtml = chunks.map(c => {
    const labelList = (c.labels && c.labels.length) ? c.labels : (c.chunk_type ? [c.chunk_type] : []);
    // A chunk the model answered for but that didn't parse is distinct from
    // one that just hasn't been reached yet — same underlying "no
    // chunk_type" state, but the cause (and the fix: usually just re-running
    // Update classification) is different, so it gets its own visible tag
    // rather than looking identical to "not classified yet".
    const tagsHtml = labelList.length
      ? labelList.map(l => `<span class="chunk-tag type-${l}">${TYPE_LABELS[l] || l}</span>`).join('')
      : (c.classification_parse_failed
          ? `<span class="chunk-tag parse-failed" title="${escapeHtml(c.classification_parse_error_raw || '')}">model response didn’t parse — retry (hover for raw output)</span>`
          : '<span class="chunk-tag type-none">not classified yet</span>');
    const confHtml = c.classification_confidence
      ? `<span class="chunk-tag confidence">conf: ${c.classification_confidence}</span>`
      : '';
    const excludedHtml = (c.keep_classification === false)
      ? '<span class="chunk-tag excluded">excluded by classifier</span>'
      : '';
    const excludedClass = (c.keep_classification === false) ? ' dropped' : '';
    const labelSelect = `
      <select class="chunk-label-select" data-id="${escapeHtml(c.chunk_id)}">
        <option value="" ${!c.chunk_type ? 'selected' : ''} disabled>set label…</option>
        ${Object.keys(TYPE_LABELS).map(t =>
          `<option value="${t}" ${c.chunk_type === t ? 'selected' : ''}>${TYPE_LABELS[t]}</option>`
        ).join('')}
      </select>`;
    return `
      <div class="chunk${excludedClass}">
        <div class="chunk-meta">
          ${tagsHtml}
          ${confHtml}
          ${excludedHtml}
          ${c.contains_table ? '<span class="chunk-tag table">table</span>' : ''}
          ${c.is_abstract ? '<span class="chunk-tag abstract">abstract</span>' : ''}
          ${labelSelect}
        </div>
        <div class="chunk-text">${escapeHtml(c.text)}</div>
      </div>`;
  }).join('');

  manuscript.innerHTML = `
    <div class="manuscript-head">
      <div class="manuscript-title">${escapeHtml(paperId)}</div>
      <div class="manuscript-stats">
        <span class="stat-chip"><span class="dot" style="background:var(--focus)"></span>Classified <span class="v">${meta.classified_chunks}/${meta.total_chunks}</span></span>
        <span class="stat-chip">Coverage <span class="v">${pct}%</span></span>
      </div>
    </div>
    <div class="manuscript-body">${chunkHtml || '<div class="empty-note">No kept chunks for this paper.</div>'}</div>
  `;

  manuscript.querySelectorAll('.chunk-label-select').forEach(sel => {
    sel.addEventListener('click', e => e.stopPropagation());
    sel.addEventListener('change', () => overrideLabel(paperId, sel.dataset.id, sel.value));
  });
}

async function overrideLabel(paperId, chunkId, chunkType) {
  if (!chunkType) return;
  try {
    await fetch(`/api/cheap-ai/papers/${paperId}/chunks/${encodeURIComponent(chunkId)}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunk_type: chunkType }),
    });
  } catch (e) {
    // fall through to reload regardless
  }
  await loadPapers();
}

function showProgressBar() {
  classifyProgress.hidden = false;
  classifyProgressFill.style.width = '0%';
  classifyProgressLabel.textContent = 'Starting…';
}

function hideProgressBar() {
  classifyProgress.hidden = true;
}

function setPauseButton(paused) {
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  pauseBtn.classList.toggle('is-paused', paused);
  pauseBtn.disabled = false;
}

function applyClassifyProgress(p) {
  // Progress bar tracks completed work (source of truth), not "chunk entered".
  const done = p.completed != null ? p.completed : p.classified;
  const pct = p.total ? Math.round(100 * done / p.total) : 0;
  classifyProgressFill.style.width = `${pct}%`;
  setPauseButton(!!p.paused);
  if (p.paused) {
    classifyProgressLabel.textContent = `Paused — ${done}/${p.total} completed (${p.classified} classified) — ${pct}%`;
    classifyStatus.textContent = `Paused. ${p.classified}/${p.total} done so far.`;
    classifyStatus.classList.remove('busy');
  } else {
    classifyProgressLabel.textContent = p.paper_id
      ? `Working on "${p.paper_id}" — ${done}/${p.total} completed (${p.classified} classified) — ${pct}%`
      : `${done}/${p.total} completed (${p.classified} classified) — ${pct}%`;
    classifyStatus.textContent = `Classifying kept chunks… ${done}/${p.total} done`;
    classifyStatus.classList.add('busy');
  }
}

async function refreshClassifyViews(paperId) {
  const papersRes = await fetch('/api/cheap-ai/papers');
  const papersData = await papersRes.json();
  papers = papersData.papers;
  renderThumbs();
  if (activeId && activeId === paperId) {
    const chunksRes = await fetch(`/api/cheap-ai/papers/${activeId}/chunks`);
    const chunksData = await chunksRes.json();
    renderManuscript(activeId, chunksData.chunks);
  }
}

// Job runs in a detached subprocess; POST only starts it. Progress file is
// the source of truth — poll until running:false so proxy timeouts can't
// kill the UI when models get slower.
async function pollUntilClassifyDone() {
  for (;;) {
    let p;
    try {
      const res = await fetch('/api/cheap-ai/classify/progress');
      p = await res.json();
    } catch (e) {
      await new Promise(r => setTimeout(r, 1500));
      continue;
    }
    applyClassifyProgress(p);
    try {
      await refreshClassifyViews(p.paper_id);
    } catch (e) {
      // view refresh is best-effort during a run
    }
    if (!p.running) return p;
    await new Promise(r => setTimeout(r, 1500));
  }
}


stopBtn.addEventListener('click', async () => {
  if (!confirm('Stop this run? Progress so far is kept, but the job will not continue.')) return;
  stopBtn.disabled = true;
  pauseBtn.disabled = true;
  try {
    const res = await fetch('/api/cheap-ai/classify/stop', { method: 'POST' });
    const p = await res.json();
    if (!res.ok) {
      alert(p.error || 'Could not stop the job.');
      stopBtn.disabled = false;
      pauseBtn.disabled = false;
      return;
    }
    // Polling loop will see running:false and finish the UI.
  } catch (e) {
    stopBtn.disabled = false;
    pauseBtn.disabled = false;
  }
});

pauseBtn.addEventListener('click', async () => {
  pauseBtn.disabled = true;
  const isPaused = pauseBtn.classList.contains('is-paused');
  const endpoint = isPaused ? '/api/cheap-ai/classify/resume' : '/api/cheap-ai/classify/pause';
  try {
    const res = await fetch(endpoint, { method: 'POST' });
    const p = await res.json();
    if (res.ok) applyClassifyProgress(p);
    else pauseBtn.disabled = false;
  } catch (e) {
    pauseBtn.disabled = false;
  }
});

classifyBtn.addEventListener('click', async () => {
  classifyBtn.disabled = true;
  classifyStatus.textContent = 'Classifying kept chunks…';
  classifyStatus.classList.add('busy');
  showProgressBar();
  pauseBtn.hidden = false;
  stopBtn.hidden = false;
  setPauseButton(false);
  try {
    const res = await fetch('/api/cheap-ai/classify', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      if (res.status === 409) {
        const finalProgress = await pollUntilClassifyDone();
        hideProgressBar();
        pauseBtn.hidden = true;
  stopBtn.hidden = true;
        if (finalProgress.stopped) {
      classifyStatus.textContent = finalProgress.error
        || `Stopped — ${finalProgress.completed || 0}/${finalProgress.total || 0} completed.`;
    } else if (finalProgress.error) {
          classifyStatus.textContent = `${finalProgress.error} (${finalProgress.completed || finalProgress.classified}/${finalProgress.total} completed before the error).`;
        } else {
          classifyStatus.textContent = finalProgress.parse_failed
            ? `Classified ${finalProgress.classified} of ${finalProgress.total} chunks — ${finalProgress.parse_failed} had a response that didn't parse (flagged below; try "Update classification" again).`
            : `Classified ${finalProgress.classified} of ${finalProgress.total} chunks.`;
        }
        classifyStatus.classList.remove('busy');
        classifyBtn.disabled = false;
        await loadPapers();
        return;
      }
      hideProgressBar();
      pauseBtn.hidden = true;
  stopBtn.hidden = true;
      classifyStatus.textContent = data.error || 'Classification failed.';
      classifyStatus.classList.remove('busy');
      classifyBtn.disabled = false;
      await loadPapers();
      return;
    }
    const finalProgress = await pollUntilClassifyDone();
    hideProgressBar();
    pauseBtn.hidden = true;
  stopBtn.hidden = true;
    if (finalProgress.stopped) {
      classifyStatus.textContent = finalProgress.error
        || `Stopped — ${finalProgress.completed || 0}/${finalProgress.total || 0} completed.`;
    } else if (finalProgress.error) {
      classifyStatus.textContent = `${finalProgress.error} (${finalProgress.completed || finalProgress.classified}/${finalProgress.total} completed before the error).`;
    } else {
      classifyStatus.textContent = finalProgress.parse_failed
        ? `Classified ${finalProgress.classified} of ${finalProgress.total} chunks — ${finalProgress.parse_failed} had a response that didn't parse (flagged below; try "Update classification" again).`
        : `Classified ${finalProgress.classified} of ${finalProgress.total} chunks.`;
    }
    classifyStatus.classList.remove('busy');
    await loadPapers();
  } catch (e) {
    hideProgressBar();
    pauseBtn.hidden = true;
  stopBtn.hidden = true;
    classifyStatus.textContent = 'Classification failed — check the server log.';
    classifyStatus.classList.remove('busy');
    await loadPapers();
  }
  classifyBtn.disabled = false;
});

async function resumeClassifyIfRunning() {
  try {
    const res = await fetch('/api/cheap-ai/classify/progress');
    const p = await res.json();
    if (!p.running) return;
    classifyBtn.disabled = true;
    showProgressBar();
    pauseBtn.hidden = false;
  stopBtn.hidden = false;
    applyClassifyProgress(p);
    const finalProgress = await pollUntilClassifyDone();
    hideProgressBar();
    pauseBtn.hidden = true;
  stopBtn.hidden = true;
    if (finalProgress.stopped) {
      classifyStatus.textContent = finalProgress.error
        || `Stopped — ${finalProgress.completed || 0}/${finalProgress.total || 0} completed.`;
    } else if (finalProgress.error) {
      classifyStatus.textContent = `${finalProgress.error} (${finalProgress.completed || finalProgress.classified}/${finalProgress.total} completed before the error).`;
    } else {
      classifyStatus.textContent = finalProgress.parse_failed
        ? `Classified ${finalProgress.classified} of ${finalProgress.total} chunks — ${finalProgress.parse_failed} had a response that didn't parse (flagged below; try "Update classification" again).`
        : `Classified ${finalProgress.classified} of ${finalProgress.total} chunks.`;
    }
    classifyStatus.classList.remove('busy');
    classifyBtn.disabled = false;
    await loadPapers();
  } catch (e) {
    // ignore
  }
}

loadPapers();
resumeClassifyIfRunning();
