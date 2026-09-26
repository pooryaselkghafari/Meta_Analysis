const thumbList = document.getElementById('thumb-list');
const manuscript = document.getElementById('manuscript');
const detectBtn = document.getElementById('detect-btn');
const detectStatus = document.getElementById('detect-status');
const detectProgress = document.getElementById('detect-progress');
const detectProgressFill = document.getElementById('detect-progress-fill');
const detectProgressLabel = document.getElementById('detect-progress-label');
const nextStageBtn = document.getElementById('next-stage-btn');

let papers = [];
let activeId = null;

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const KEEP_REASON_LABELS = {
  contains_target_estimate: 'contains target estimate',
  contains_target_variable_definition: 'defines target variable',
  contains_model_specification_for_target_estimate: 'model spec for target estimate',
  literature_relevant_context: 'literature — relevant context',
  literature_estimate_only: 'literature — other paper\'s estimate',
  off_target: 'off target',
  no_estimate_signal: 'no estimate signal',
};

async function loadPapers() {
  const res = await fetch('/api/detection-ai/papers');
  const data = await res.json();
  papers = data.papers;
  renderThumbs();
  if (papers.length) {
    selectPaper((papers.find(p => p.paper_id === activeId) || papers[0]).paper_id);
  } else {
    manuscript.innerHTML = `
      <div class="manuscript-empty">
        <div class="big">No kept chunks yet</div>
        <div>Run Analyze on the Corpus page first — this page only screens chunks that already passed the heuristic filter.</div>
      </div>`;
  }
  try {
    const stateRes = await fetch('/api/pipeline/state');
    const state = await stateRes.json();
    detectBtn.textContent = state.detection_ai_done ? 'Update detection' : 'Run detection over kept chunks';
    nextStageBtn.disabled = !state.detection_ai_done;
    nextStageBtn.title = state.detection_ai_done ? '' : 'Run detection over kept chunks first';
  } catch (e) {
    // leave button state as-is if the state endpoint is unreachable
  }
}

nextStageBtn.addEventListener('click', () => {
  if (!nextStageBtn.disabled) window.location.href = '/main-ai';
});

function renderThumbs() {
  thumbList.innerHTML = '';
  papers.forEach(p => {
    const card = document.createElement('button');
    card.className = 'thumb-card' + (p.paper_id === activeId ? ' active' : '');
    card.dataset.id = p.paper_id;
    const pct = p.total_chunks ? Math.round(100 * p.screened_chunks / p.total_chunks) : 0;
    card.innerHTML = `
      <div class="thumb-img-wrap">
        <img src="/api/papers/${p.paper_id}/thumbnail" alt=""
             onerror="this.remove(); this.parentElement.innerHTML='<span class=&quot;fallback&quot;>&#128196;</span>'">
      </div>
      <div class="thumb-name">${escapeHtml(p.paper_id)}</div>
      <div class="thumb-stat">${p.screened_chunks}/${p.total_chunks} screened &middot; ${pct}%</div>
    `;
    card.addEventListener('click', () => selectPaper(p.paper_id));
    thumbList.appendChild(card);
  });
}

async function selectPaper(paperId) {
  activeId = paperId;
  renderThumbs();
  const res = await fetch(`/api/detection-ai/papers/${paperId}/chunks`);
  const data = await res.json();
  renderManuscript(paperId, data.chunks);
}

function renderManuscript(paperId, chunks) {
  const meta = papers.find(p => p.paper_id === paperId) || { total_chunks: chunks.length, screened_chunks: 0 };
  const pct = meta.total_chunks ? Math.round(100 * meta.screened_chunks / meta.total_chunks) : 0;

  const chunkHtml = chunks.map(c => {
    const dropped = c.detected === false;
    let detectTag;
    if (c.detected === true) detectTag = '<span class="chunk-tag detect-yes">estimate likely</span>';
    else if (c.detected === false) detectTag = '<span class="chunk-tag detect-no">no estimate</span>';
    else detectTag = '<span class="chunk-tag detect-pending">not screened yet</span>';
    const confTag = c.detection_confidence
      ? `<span class="chunk-tag confidence">conf: ${c.detection_confidence}</span>`
      : '';
    const reasonTag = c.keep_reason
      ? `<span class="chunk-tag reason">${escapeHtml(c.keep_reason.replace(/_/g, ' '))}</span>`
      : '';
    const tm = c.target_match;
    const matchTags = [];
    if (tm && tm.elasticity_match) {
      const m = tm.elasticity_match;
      matchTags.push(`<span class="chunk-tag match">Elasticity ${m.match_type || '?'}${m.paper_term ? `: ${escapeHtml(m.paper_term)}` : ''}</span>`);
    }
    if (tm && tm.product_match) {
      const m = tm.product_match;
      matchTags.push(`<span class="chunk-tag match">Product ${m.match_type || '?'}${m.paper_term ? `: ${escapeHtml(m.paper_term)}` : ''}</span>`);
    }
    if (tm && tm.cross_price_product_match) {
      const m = tm.cross_price_product_match;
      matchTags.push(`<span class="chunk-tag match">Cross-price product ${m.match_type || '?'}${m.paper_term ? `: ${escapeHtml(m.paper_term)}` : ''}</span>`);
    }

    const reasonSelect = `
      <select class="chunk-label-select" data-id="${escapeHtml(c.chunk_id)}">
        <option value="" ${!c.keep_reason ? 'selected' : ''} disabled>set label…</option>
        ${Object.keys(KEEP_REASON_LABELS).map(k =>
          `<option value="${k}" ${c.keep_reason === k ? 'selected' : ''}>${KEEP_REASON_LABELS[k]}</option>`
        ).join('')}
      </select>`;
    return `
      <div class="chunk ${dropped ? 'dropped' : ''}">
        <div class="chunk-meta">
          ${detectTag}
          ${confTag}
          ${reasonTag}
          ${matchTags.join('')}
          ${c.contains_table ? '<span class="chunk-tag table">table</span>' : ''}
          ${c.is_abstract ? '<span class="chunk-tag abstract">abstract</span>' : ''}
          ${reasonSelect}
        </div>
        <div class="chunk-text">${escapeHtml(c.text)}</div>
      </div>`;
  }).join('');

  manuscript.innerHTML = `
    <div class="manuscript-head">
      <div class="manuscript-title">${escapeHtml(paperId)}</div>
      <div class="manuscript-stats">
        <span class="stat-chip"><span class="dot" style="background:var(--focus)"></span>Screened <span class="v">${meta.screened_chunks}/${meta.total_chunks}</span></span>
        <span class="stat-chip">Coverage <span class="v">${pct}%</span></span>
      </div>
    </div>
    <div class="manuscript-body">${chunkHtml || '<div class="empty-note">No kept chunks for this paper.</div>'}</div>
  `;

  manuscript.querySelectorAll('.chunk-label-select').forEach(sel => {
    sel.addEventListener('click', e => e.stopPropagation());
    sel.addEventListener('change', () => overrideKeepReason(paperId, sel.dataset.id, sel.value));
  });
}

async function overrideKeepReason(paperId, chunkId, keepReason) {
  if (!keepReason) return;
  try {
    await fetch(`/api/detection-ai/papers/${paperId}/chunks/${encodeURIComponent(chunkId)}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keep_reason: keepReason }),
    });
  } catch (e) {
    // fall through to reload regardless
  }
  await loadPapers();
}

function showProgressBar() {
  detectProgress.hidden = false;
  detectProgressFill.style.width = '0%';
  detectProgressLabel.textContent = 'Starting…';
}

function hideProgressBar() {
  detectProgress.hidden = true;
}

function applyDetectProgress(p) {
  const done = p.completed != null ? p.completed : p.screened;
  const pct = p.total ? Math.round(100 * done / p.total) : 0;
  detectProgressFill.style.width = `${pct}%`;
  detectProgressLabel.textContent = p.paper_id
    ? `Working on "${p.paper_id}" — ${done}/${p.total} completed (${p.screened} screened) — ${pct}%`
    : `${done}/${p.total} completed (${p.screened} screened) — ${pct}%`;
  detectStatus.textContent = `Screening kept chunks… ${done}/${p.total} done`;
}

async function refreshDetectViews(paperId) {
  const papersRes = await fetch('/api/detection-ai/papers');
  const papersData = await papersRes.json();
  papers = papersData.papers;
  renderThumbs();
  if (activeId && activeId === paperId) {
    const chunksRes = await fetch(`/api/detection-ai/papers/${activeId}/chunks`);
    const chunksData = await chunksRes.json();
    renderManuscript(activeId, chunksData.chunks);
  }
}

// Job runs in a detached subprocess; POST only starts it. Progress file is
// the source of truth — poll until running:false so proxy timeouts can't
// kill the UI when models get slower.
async function pollUntilDetectDone() {
  for (;;) {
    let p;
    try {
      const res = await fetch('/api/detection-ai/detect/progress');
      p = await res.json();
    } catch (e) {
      await new Promise(r => setTimeout(r, 1500));
      continue;
    }
    applyDetectProgress(p);
    try {
      await refreshDetectViews(p.paper_id);
    } catch (e) {
      // view refresh is best-effort during a run
    }
    if (!p.running) return p;
    await new Promise(r => setTimeout(r, 1500));
  }
}

detectBtn.addEventListener('click', async () => {
  detectBtn.disabled = true;
  detectStatus.textContent = 'Screening kept chunks…';
  detectStatus.classList.add('busy');
  showProgressBar();
  try {
    const res = await fetch('/api/detection-ai/detect', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      if (res.status === 409) {
        const finalProgress = await pollUntilDetectDone();
        hideProgressBar();
        if (finalProgress.error) {
          detectStatus.textContent = `${finalProgress.error} (${finalProgress.completed || finalProgress.screened}/${finalProgress.total} completed before the error).`;
        } else {
          detectStatus.textContent = `Screened ${finalProgress.screened} of ${finalProgress.total} chunks — dropped ${finalProgress.dropped || 0}.`;
        }
        detectStatus.classList.remove('busy');
        detectBtn.disabled = false;
        await loadPapers();
        return;
      }
      hideProgressBar();
      detectStatus.textContent = data.error || 'Detection failed.';
      detectStatus.classList.remove('busy');
      detectBtn.disabled = false;
      await loadPapers();
      return;
    }
    const finalProgress = await pollUntilDetectDone();
    hideProgressBar();
    if (finalProgress.error) {
      detectStatus.textContent = `${finalProgress.error} (${finalProgress.completed || finalProgress.screened}/${finalProgress.total} completed before the error).`;
    } else {
      detectStatus.textContent = `Screened ${finalProgress.screened} of ${finalProgress.total} chunks — dropped ${finalProgress.dropped || 0}.`;
    }
    detectStatus.classList.remove('busy');
    await loadPapers();
  } catch (e) {
    hideProgressBar();
    detectStatus.textContent = 'Detection failed — check the server log.';
    detectStatus.classList.remove('busy');
    await loadPapers();
  }
  detectBtn.disabled = false;
});

async function resumeDetectIfRunning() {
  try {
    const res = await fetch('/api/detection-ai/detect/progress');
    const p = await res.json();
    if (!p.running) return;
    detectBtn.disabled = true;
    showProgressBar();
    applyDetectProgress(p);
    const finalProgress = await pollUntilDetectDone();
    hideProgressBar();
    if (finalProgress.error) {
      detectStatus.textContent = `${finalProgress.error} (${finalProgress.completed || finalProgress.screened}/${finalProgress.total} completed before the error).`;
    } else {
      detectStatus.textContent = `Screened ${finalProgress.screened} of ${finalProgress.total} chunks — dropped ${finalProgress.dropped || 0}.`;
    }
    detectStatus.classList.remove('busy');
    detectBtn.disabled = false;
    await loadPapers();
  } catch (e) {
    // ignore
  }
}

loadPapers();
resumeDetectIfRunning();
