const thumbList = document.getElementById('thumb-list');
const manuscript = document.getElementById('manuscript');
const updateBtn = document.getElementById('update-btn');
const nextStageBtn = document.getElementById('next-stage-btn');
const stageStatus = document.getElementById('stage-status');

let papers = [];
let activeId = null;

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function loadPapers() {
  const res = await fetch('/api/results/papers');
  const data = await res.json();
  papers = data.papers;
  renderThumbs();
  if (papers.length) {
    const current = papers.find(p => p.paper_id === activeId);
    const target = current || papers.find(p => p.ok) || papers[0];
    selectPaper(target.paper_id);
  }
}

function renderThumbs() {
  thumbList.innerHTML = '';
  papers.forEach(p => {
    const card = document.createElement('button');
    card.className = 'thumb-card' + (p.paper_id === activeId ? ' active' : '');
    card.dataset.id = p.paper_id;

    const pct = p.total_chunks ? Math.round(100 * p.kept_chunks / p.total_chunks) : 0;
    const statLine = p.ok
      ? `${p.kept_chunks}/${p.total_chunks} kept &middot; ${pct}%`
      : 'parse failed';

    card.innerHTML = `
      <div class="thumb-img-wrap">
        <img src="/api/papers/${p.paper_id}/thumbnail" alt=""
             onerror="this.remove(); this.parentElement.innerHTML='<span class=&quot;fallback&quot;>&#128196;</span>'">
      </div>
      <div class="thumb-name">${escapeHtml(p.paper_id)}</div>
      <div class="thumb-stat ${p.ok ? '' : 'thumb-error'}">${statLine}</div>
    `;
    card.addEventListener('click', () => selectPaper(p.paper_id));
    thumbList.appendChild(card);
  });
}

async function selectPaper(paperId) {
  activeId = paperId;
  renderThumbs();

  const meta = papers.find(p => p.paper_id === paperId);
  if (!meta || !meta.ok) {
    manuscript.innerHTML = `
      <div class="manuscript-empty">
        <div class="big">Parse failed</div>
        <div>${escapeHtml((meta && meta.error) || 'Unknown error while parsing this paper.')}</div>
      </div>`;
    return;
  }

  const res = await fetch(`/api/results/papers/${paperId}/chunks`);
  const data = await res.json();
  renderManuscript(meta, data.chunks);
}

function renderManuscript(meta, chunks) {
  const pct = meta.total_chunks ? Math.round(100 * meta.kept_chunks / meta.total_chunks) : 0;

  const chunkHtml = chunks.map(c => {
    const dropped = !c.passes_filter;
    const tags = [];
    tags.push(dropped
      ? `<span class="chunk-tag dropped">dropped</span>`
      : `<span class="chunk-tag kept">kept</span>`);
    if (c.contains_table) tags.push(`<span class="chunk-tag table">table</span>`);
    if (dropped && c.filter_reason) {
      tags.push(`<span class="chunk-reason">${escapeHtml(c.filter_reason.replace(/_/g, ' '))}</span>`);
    }
    const overrideBtn = dropped
      ? `<button class="chunk-override-btn" data-id="${escapeHtml(c.chunk_id)}" data-action="return">Return to analysis</button>`
      : `<button class="chunk-override-btn drop" data-id="${escapeHtml(c.chunk_id)}" data-action="drop">Drop</button>`;
    return `
      <div class="chunk ${dropped ? 'dropped' : ''}">
        <div class="chunk-meta">${tags.join('')}${overrideBtn}</div>
        <div class="chunk-text">${escapeHtml(c.text)}</div>
      </div>`;
  }).join('');

  manuscript.innerHTML = `
    <div class="manuscript-head">
      <div class="manuscript-title">${escapeHtml(meta.paper_id)}</div>
      <div class="manuscript-stats">
        <span class="stat-chip kept"><span class="dot"></span>Kept <span class="v">${meta.kept_chunks}</span></span>
        <span class="stat-chip drop"><span class="dot"></span>Dropped <span class="v">${meta.total_chunks - meta.kept_chunks}</span></span>
        <span class="stat-chip">Reduction <span class="v">${100 - pct}%</span></span>
        <span class="legend">
          <span><span class="bar kept"></span>passed heuristic filter</span>
          <span><span class="bar drop"></span>dropped before any AI call</span>
        </span>
      </div>
    </div>
    <div class="manuscript-body">${chunkHtml || '<div class="empty-note">No chunks produced for this paper.</div>'}</div>
  `;

  manuscript.querySelectorAll('.chunk-override-btn').forEach(btn => {
    btn.addEventListener('click', () => overrideChunk(meta.paper_id, btn.dataset.id, btn.dataset.action === 'return'));
  });
}

async function overrideChunk(paperId, chunkId, passesFilter) {
  try {
    await fetch(`/api/results/papers/${paperId}/chunks/${encodeURIComponent(chunkId)}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ passes_filter: passesFilter }),
    });
  } catch (e) {
    // fall through to reload regardless — the UI will reflect whatever state actually landed
  }
  await loadPapers();
}

nextStageBtn.addEventListener('click', () => { window.location.href = '/cheap-ai'; });

updateBtn.addEventListener('click', async () => {
  updateBtn.disabled = true;
  stageStatus.textContent = 'Re-running Corpus parse & heuristic filter…';
  stageStatus.classList.add('busy');
  try {
    const res = await fetch('/api/analyze', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      stageStatus.textContent = data.error || 'Re-analysis failed.';
      stageStatus.classList.remove('busy');
      updateBtn.disabled = false;
      return;
    }
    stageStatus.textContent = 'Done. Downstream Cheap AI / Detection AI results were cleared — re-run them from their pages.';
    stageStatus.classList.remove('busy');
    await loadPapers();
  } catch (e) {
    stageStatus.textContent = 'Re-analysis failed — check the server log.';
    stageStatus.classList.remove('busy');
  }
  updateBtn.disabled = false;
});

loadPapers();
