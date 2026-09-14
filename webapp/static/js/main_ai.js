const paperFilter = document.getElementById('paper-filter');
const reviewOnlyFilter = document.getElementById('review-only-filter');
const recordsCount = document.getElementById('records-count');
const recordsTbody = document.getElementById('records-tbody');
const recordsEmpty = document.getElementById('records-empty');
const recordsTableWrap = document.getElementById('records-table-wrap');
const extractBtn = document.getElementById('extract-btn');
const extractStatus = document.getElementById('extract-status');
const extractProgress = document.getElementById('extract-progress');
const extractProgressFill = document.getElementById('extract-progress-fill');
const extractProgressLabel = document.getElementById('extract-progress-label');
const addRowBtn = document.getElementById('add-row-btn');
const dedupeBtn = document.getElementById('dedupe-btn');

let allRecords = [];
let expandedId = null;
let editingId = null;
let editDraft = null;

// Data-driven edit form — one entry per plain scalar field the user can
// correct directly. Composite fields (confidence_interval, time_period,
// test_statistic, row/column) are handled separately since they're each
// really 2+ inputs feeding one JSON value.
const EDIT_FIELDS = [
  { key: 'paper_id', label: 'Paper', type: 'text' },
  { key: 'target_elasticity_type', label: 'Elasticity type', type: 'text' },
  { key: 'target_product', label: 'Product', type: 'text' },
  { key: 'target_cross_price_product', label: 'Cross-price product', type: 'text' },
  // options filled in at load time from /api/food-groups (see loadFoodGroups) —
  // a fixed 8-value vocab, not something typed per-project like the fields above.
  { key: 'target_food_group', label: 'Food group', type: 'select', options: [] },
  { key: 'target_cross_price_food_group', label: 'Cross-price food group', type: 'select', options: [] },
  { key: 'paper_elasticity_wording_raw', label: 'Paper elasticity wording', type: 'text' },
  { key: 'paper_product_wording_raw', label: 'Paper product wording', type: 'text' },
  { key: 'paper_cross_price_product_wording_raw', label: 'Paper cross-price wording', type: 'text' },
  { key: 'match_type', label: 'Match type', type: 'select', options: ['exact', 'synonym', 'related_separate', 'no_match'] },
  { key: 'target_match_justification', label: 'Match justification', type: 'text' },
  { key: 'variable_role', label: 'Role', type: 'select', options: ['treatment', 'primary_regressor', 'control', 'unknown'] },
  { key: 'estimate_type', label: 'Estimate type', type: 'select', options: ['regression_coefficient', 'elasticity', 'semi_elasticity', 'multiplier', 'impulse_response', 'marginal_effect', 'hazard_ratio', 'other'] },
  { key: 'coefficient', label: 'Coefficient', type: 'number' },
  { key: 'standard_error', label: 'Standard error', type: 'number' },
  { key: 'p_value', label: 'P-value', type: 'number' },
  { key: 'significance_stars', label: 'Significance stars', type: 'text' },
  { key: 'elasticity_is_raw', label: 'Elasticity is raw (unchecked = derived)', type: 'bool' },
  { key: 'elasticity_transformation_type', label: 'Transformation type', type: 'select', options: ['none', 'log', 'log_difference', 'ratio', 'percent_change', 'first_difference', 'growth_rate', 'index', 'standardized', 'other', 'unknown'] },
  { key: 'elasticity_unit', label: 'Elasticity unit', type: 'text' },
  { key: 'unit_source', label: 'Unit source', type: 'select', options: ['table', 'data_section', 'inferred', 'unknown'] },
  { key: 'specification_status', label: 'Specification status', type: 'select', options: ['explicit_baseline', 'inferred_baseline', 'unknown'] },
  { key: 'baseline_evidence', label: 'Baseline evidence', type: 'text' },
  { key: 'model_type', label: 'Model type', type: 'text' },
  { key: 'countries_region', label: 'Country/region', type: 'text' },
  { key: 'frequency', label: 'Frequency', type: 'text' },
  { key: 'n_obs', label: 'N (obs)', type: 'number' },
  { key: 'n_units', label: 'N (units)', type: 'number' },
  { key: 'data_source', label: 'Data source', type: 'text' },
  { key: 'source_type', label: 'Source type', type: 'select', options: ['table', 'text', 'figure'] },
  { key: 'requires_review', label: 'Requires review', type: 'bool' },
  { key: 'review_reason', label: 'Review reasons (comma-separated)', type: 'text_list' },
];

function escapeHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fmtCoef(v) {
  if (v === null || v === undefined) return '–';
  const n = Number(v);
  return Number.isNaN(n) ? escapeHtml(v) : n.toFixed(3);
}

function fmtTimePeriod(tp) {
  if (!tp || (tp.start == null && tp.end == null)) return '–';
  if (tp.start != null && tp.end != null && tp.start !== tp.end) return `${tp.start}–${tp.end}`;
  return String(tp.start ?? tp.end);
}

async function loadPapers() {
  const res = await fetch('/api/main-ai/papers');
  const data = await res.json();
  const current = paperFilter.value;
  paperFilter.innerHTML = '<option value="">All papers</option>';
  (data.papers || []).forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.paper_id;
    opt.textContent = `${p.paper_id} (${p.extracted_chunks}/${p.total_chunks} chunks extracted)`;
    paperFilter.appendChild(opt);
  });
  if ([...paperFilter.options].some(o => o.value === current)) paperFilter.value = current;

  try {
    const stateRes = await fetch('/api/pipeline/state');
    const state = await stateRes.json();
    extractBtn.textContent = state.main_ai_done ? 'Update extraction' : 'Run extraction over detected chunks';
  } catch (e) {
    // leave button label as-is if the state endpoint is unreachable
  }
}

async function loadRecords() {
  const res = await fetch('/api/main-ai/records');
  const data = await res.json();
  allRecords = data.records || [];
  renderTable();
}

function matchingRecords() {
  return allRecords.filter(r => {
    if (paperFilter.value && r.paper_id !== paperFilter.value) return false;
    if (reviewOnlyFilter.checked && !r.requires_review) return false;
    return true;
  });
}

function toggleRow(estimateId) {
  if (editingId === estimateId) return; // don't collapse out from under an active edit
  expandedId = expandedId === estimateId ? null : estimateId;
  renderTable();
}

function startEdit(r) {
  editingId = r.estimate_id;
  expandedId = r.estimate_id;
  editDraft = {
    ...r,
    confidence_interval_low: r.confidence_interval ? r.confidence_interval[0] : '',
    confidence_interval_high: r.confidence_interval ? r.confidence_interval[1] : '',
    time_period_start: r.time_period ? r.time_period.start : '',
    time_period_end: r.time_period ? r.time_period.end : '',
    test_statistic_type: r.test_statistic ? r.test_statistic.type : '',
    test_statistic_value: r.test_statistic ? r.test_statistic.value : '',
    row_label: (r.source_location && r.source_location.row) || '',
    column_label: (r.source_location && r.source_location.column) || '',
    review_reason: (r.review_reason || []).join(', '),
  };
  renderTable();
}

function cancelEdit() {
  editingId = null;
  editDraft = null;
  renderTable();
}

async function saveEdit(estimateId) {
  const payload = {};
  EDIT_FIELDS.forEach(f => {
    let v = editDraft[f.key];
    if (f.type === 'bool') v = !!v;
    else if (f.type === 'text_list') v = v; // sent as comma-separated string, server splits it
    else if (v === '') v = null;
    payload[f.key] = v;
  });
  payload.confidence_interval = (editDraft.confidence_interval_low !== '' && editDraft.confidence_interval_high !== '')
    ? [Number(editDraft.confidence_interval_low), Number(editDraft.confidence_interval_high)]
    : null;
  payload.time_period = {
    start: editDraft.time_period_start !== '' ? Number(editDraft.time_period_start) : null,
    end: editDraft.time_period_end !== '' ? Number(editDraft.time_period_end) : null,
  };
  payload.test_statistic = (editDraft.test_statistic_type && editDraft.test_statistic_value !== '')
    ? { type: editDraft.test_statistic_type, value: Number(editDraft.test_statistic_value) }
    : null;
  payload.row = editDraft.row_label || null;
  payload.column = editDraft.column_label || null;

  try {
    const res = await fetch(`/api/main-ai/records/${encodeURIComponent(estimateId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      alert(data.error || 'Save failed.');
      return;
    }
  } catch (e) {
    alert('Save failed — check the server log.');
    return;
  }
  editingId = null;
  editDraft = null;
  await loadRecords();
}

async function dedupeRows() {
  dedupeBtn.disabled = true;
  const original = dedupeBtn.textContent;
  dedupeBtn.textContent = 'Checking…';
  try {
    const res = await fetch('/api/main-ai/dedupe', { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Could not check for duplicates.');
      return;
    }
    if (data.removed > 0) {
      await loadRecords();
      alert(`Removed ${data.removed} rounded-duplicate record${data.removed === 1 ? '' : 's'}.`);
    } else {
      alert('No rounded duplicates found.');
    }
  } catch (e) {
    alert('Could not check for duplicates — check the server log.');
  }
  dedupeBtn.textContent = original;
  dedupeBtn.disabled = false;
}

async function addRow() {
  addRowBtn.disabled = true;
  try {
    const res = await fetch('/api/main-ai/records', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper_id: paperFilter.value || '' }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Could not add a row.');
      return;
    }
    await loadRecords();
    startEdit(data.record); // drop straight into edit mode on the new blank row
  } catch (e) {
    alert('Could not add a row — check the server log.');
  }
  addRowBtn.disabled = false;
}

async function deleteRow(estimateId) {
  if (!confirm('Delete this row entirely? This can\'t be undone.')) return;
  try {
    const res = await fetch(`/api/main-ai/records/${encodeURIComponent(estimateId)}`, { method: 'DELETE' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      alert(data.error || 'Delete failed.');
      return;
    }
  } catch (e) {
    alert('Delete failed — check the server log.');
    return;
  }
  if (editingId === estimateId) { editingId = null; editDraft = null; }
  if (expandedId === estimateId) expandedId = null;
  await loadRecords();
}

function editFieldHtml(f) {
  const val = editDraft[f.key];
  const onInput = `data-field="${f.key}"`;
  if (f.type === 'select') {
    const opts = ['<option value="">–</option>'].concat(
      f.options.map(o => `<option value="${o}" ${val === o ? 'selected' : ''}>${o}</option>`)
    ).join('');
    return `<select class="edit-input" ${onInput}>${opts}</select>`;
  }
  if (f.type === 'bool') {
    return `<input type="checkbox" class="edit-input" ${onInput} ${val ? 'checked' : ''}>`;
  }
  if (f.type === 'number') {
    return `<input type="number" step="any" class="edit-input" ${onInput} value="${val ?? ''}">`;
  }
  return `<input type="text" class="edit-input" ${onInput} value="${escapeHtml(val ?? '')}">`;
}

function editRow(r) {
  const fieldsHtml = EDIT_FIELDS.map(f =>
    `<div class="record-detail-item">
      <div class="record-detail-label">${escapeHtml(f.label)}</div>
      ${editFieldHtml(f)}
    </div>`
  ).join('');
  const compositeHtml = `
    <div class="record-detail-item">
      <div class="record-detail-label">Confidence interval [low, high]</div>
      <input type="number" step="any" class="edit-input" data-field="confidence_interval_low" value="${editDraft.confidence_interval_low ?? ''}" style="width:48%">
      <input type="number" step="any" class="edit-input" data-field="confidence_interval_high" value="${editDraft.confidence_interval_high ?? ''}" style="width:48%">
    </div>
    <div class="record-detail-item">
      <div class="record-detail-label">Time period [start, end]</div>
      <input type="number" class="edit-input" data-field="time_period_start" value="${editDraft.time_period_start ?? ''}" style="width:48%">
      <input type="number" class="edit-input" data-field="time_period_end" value="${editDraft.time_period_end ?? ''}" style="width:48%">
    </div>
    <div class="record-detail-item">
      <div class="record-detail-label">Test statistic [type, value]</div>
      <select class="edit-input" data-field="test_statistic_type" style="width:48%">
        ${['', 't', 'z', 'chi2', 'f', 'other'].map(o => `<option value="${o}" ${editDraft.test_statistic_type === o ? 'selected' : ''}>${o || '–'}</option>`).join('')}
      </select>
      <input type="number" step="any" class="edit-input" data-field="test_statistic_value" value="${editDraft.test_statistic_value ?? ''}" style="width:48%">
    </div>
    <div class="record-detail-item">
      <div class="record-detail-label">Table row / column</div>
      <input type="text" class="edit-input" data-field="row_label" value="${escapeHtml(editDraft.row_label ?? '')}" placeholder="row" style="width:48%">
      <input type="text" class="edit-input" data-field="column_label" value="${escapeHtml(editDraft.column_label ?? '')}" placeholder="column" style="width:48%">
    </div>`;
  return `<tr class="record-detail-row editing"><td colspan="17">
      <div class="record-detail-grid">${fieldsHtml}${compositeHtml}</div>
      <div class="record-edit-actions">
        <button class="btn-primary" data-action="save" data-id="${escapeHtml(r.estimate_id)}">Save</button>
        <button class="btn-secondary" data-action="cancel">Cancel</button>
      </div>
    </td></tr>`;
}

function detailRow(r) {
  const items = [
    ['Paper elasticity wording', r.paper_elasticity_wording_raw],
    ['Paper product wording', r.paper_product_wording_raw],
    ['Paper cross-price product wording', r.paper_cross_price_product_wording_raw],
    ['Match justification', r.target_match_justification],
    ['Cross-price food group', r.target_cross_price_food_group || '–'],
    ['Elasticity: raw / derived', r.elasticity_is_raw === null || r.elasticity_is_raw === undefined ? '–' : `${r.elasticity_is_raw ? 'raw' : 'derived'} (${r.elasticity_transformation_type || 'unknown'})`],
    ['Elasticity unit', r.elasticity_unit], ['Unit source', r.unit_source],
    ['Confidence interval', r.confidence_interval ? `[${r.confidence_interval[0]}, ${r.confidence_interval[1]}]` : (r.confidence_interval_reported ? '–' : 'not reported')],
    ['P-value', r.p_value != null ? r.p_value : (r.p_value_reported ? '–' : 'not reported')],
    ['Test statistic', r.test_statistic ? `${r.test_statistic.type} = ${r.test_statistic.value}` : '–'],
    ['Baseline evidence', r.baseline_evidence],
    ['Model type', r.model_type], ['Country/region', r.countries_region],
    ['Time period', fmtTimePeriod(r.time_period)], ['Frequency', r.frequency],
    ['N (obs)', r.n_obs], ['N (units)', r.n_units], ['Data source', r.data_source],
    ['Source', [r.source_type, r.source_location && r.source_location.row ? `row: ${r.source_location.row}` : null,
                r.source_location && r.source_location.column ? `col: ${r.source_location.column}` : null,
                r.source_location && r.source_location.page ? `p.${r.source_location.page}` : null]
                .filter(Boolean).join(' · ') || '–'],
    ['Review reasons', (r.review_reason && r.review_reason.length) ? r.review_reason.join(', ') : '–'],
  ];
  const cells = items.map(([label, val]) =>
    `<div class="record-detail-item"><div class="record-detail-label">${escapeHtml(label)}</div><div class="record-detail-value">${escapeHtml(val)}</div></div>`
  ).join('');
  return `<tr class="record-detail-row"><td colspan="17"><div class="record-detail-grid">${cells}</div></td></tr>`;
}

function renderTable() {
  const records = matchingRecords();
  recordsCount.textContent = `${records.length} of ${allRecords.length} record${allRecords.length === 1 ? '' : 's'}`;
  recordsTableWrap.querySelector('table').hidden = records.length === 0 && allRecords.length === 0;
  recordsEmpty.hidden = allRecords.length !== 0;

  if (!records.length) {
    recordsTbody.innerHTML = '';
    return;
  }

  recordsTbody.innerHTML = records.map(r => {
    const sig = r.significance_stars ? escapeHtml(r.significance_stars) : '';
    const reviewBadge = r.requires_review
      ? '<span class="chunk-tag excluded">needs review</span>'
      : '<span class="chunk-tag kept">ok</span>';
    const matchBadge = r.match_type ? `<span class="chunk-tag match">${escapeHtml(r.match_type)}</span>` : '–';
    const editedBadge = r.manually_added
      ? '<span class="chunk-tag confidence">manual</span>'
      : (r.manually_edited ? '<span class="chunk-tag confidence">edited</span>' : '');
    const isEditing = editingId === r.estimate_id;
    const rowHtml = `
      <tr class="record-row ${expandedId === r.estimate_id ? 'expanded' : ''} ${isEditing ? 'editing' : ''}" data-id="${escapeHtml(r.estimate_id)}">
        <td class="record-expand">${expandedId === r.estimate_id ? '▾' : '▸'}</td>
        <td class="mono">${escapeHtml(r.paper_id)}</td>
        <td>${escapeHtml(r.target_elasticity_type) || '–'}</td>
        <td>${escapeHtml(r.target_product) || '–'}</td>
        <td>${r.target_food_group ? `<span class="chunk-tag match">${escapeHtml(r.target_food_group)}</span>` : '–'}</td>
        <td>${escapeHtml(r.target_cross_price_product) || '–'}</td>
        <td class="record-wording">${escapeHtml(r.paper_elasticity_wording_raw) || '–'}</td>
        <td class="record-wording">${escapeHtml(r.paper_product_wording_raw) || '–'}</td>
        <td>${matchBadge}</td>
        <td>${escapeHtml(r.variable_role) || '–'}</td>
        <td>${escapeHtml(r.estimate_type) || '–'}</td>
        <td class="mono">${fmtCoef(r.coefficient)}</td>
        <td class="mono">${fmtCoef(r.standard_error)}</td>
        <td class="mono">${sig || '–'}</td>
        <td>${escapeHtml(r.specification_status) || '–'}</td>
        <td><div class="review-cell">${reviewBadge}${editedBadge}</div></td>
        <td class="record-actions">
          <button class="chunk-override-btn" data-action="edit" data-id="${escapeHtml(r.estimate_id)}">${isEditing ? 'Editing…' : 'Edit'}</button>
          <button class="chunk-override-btn drop" data-action="delete" data-id="${escapeHtml(r.estimate_id)}">Delete</button>
        </td>
      </tr>`;
    if (isEditing) return rowHtml + editRow(r);
    return expandedId === r.estimate_id ? rowHtml + detailRow(r) : rowHtml;
  }).join('');

  recordsTbody.querySelectorAll('.record-row').forEach(row => {
    row.addEventListener('click', (e) => {
      if (e.target.closest('[data-action="edit"]')) return;
      toggleRow(row.dataset.id);
    });
  });

  recordsTbody.querySelectorAll('[data-action="edit"]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const rec = allRecords.find(r => r.estimate_id === btn.dataset.id);
      if (rec) startEdit(rec);
    });
  });

  recordsTbody.querySelectorAll('[data-action="delete"]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteRow(btn.dataset.id);
    });
  });

  recordsTbody.querySelectorAll('.record-detail-row.editing .edit-input').forEach(input => {
    const evt = input.type === 'checkbox' ? 'change' : 'input';
    input.addEventListener(evt, () => {
      editDraft[input.dataset.field] = input.type === 'checkbox' ? input.checked : input.value;
    });
    input.addEventListener('click', e => e.stopPropagation());
  });

  const saveBtn = recordsTbody.querySelector('[data-action="save"]');
  if (saveBtn) saveBtn.addEventListener('click', (e) => { e.stopPropagation(); saveEdit(saveBtn.dataset.id); });
  const cancelBtn = recordsTbody.querySelector('[data-action="cancel"]');
  if (cancelBtn) cancelBtn.addEventListener('click', (e) => { e.stopPropagation(); cancelEdit(); });
}

paperFilter.addEventListener('change', renderTable);
reviewOnlyFilter.addEventListener('change', renderTable);
addRowBtn.addEventListener('click', addRow);
dedupeBtn.addEventListener('click', dedupeRows);

function showProgressBar() {
  extractProgress.hidden = false;
  extractProgressFill.style.width = '0%';
  extractProgressLabel.textContent = 'Starting…';
}

function hideProgressBar() {
  extractProgress.hidden = true;
}

async function pollProgress() {
  try {
    const res = await fetch('/api/main-ai/extract/progress');
    const p = await res.json();
    const pct = p.total ? Math.round(100 * p.chunk_index / p.total) : 0;
    extractProgressFill.style.width = `${pct}%`;
    extractProgressLabel.textContent = p.paper_id
      ? `Working on "${p.paper_id}" — chunk ${p.chunk_index}/${p.total} (${p.records_found} records so far) — ${pct}%`
      : `Chunk ${p.chunk_index}/${p.total} (${p.records_found} records so far) — ${pct}%`;
    extractStatus.textContent = `Extracting… ${p.extracted}/${p.total} chunks done`;
    await Promise.all([loadPapers(), loadRecords()]);
  } catch (e) {
    // ignore — a single poll failing isn't fatal, the next one will retry
  }
}

extractBtn.addEventListener('click', async () => {
  extractBtn.disabled = true;
  extractStatus.textContent = 'Extracting…';
  extractStatus.classList.add('busy');
  showProgressBar();
  const progressTimer = setInterval(pollProgress, 1500);
  pollProgress();
  try {
    const res = await fetch('/api/main-ai/extract', { method: 'POST' });
    const data = await res.json();
    clearInterval(progressTimer);
    hideProgressBar();
    if (!res.ok) {
      extractStatus.textContent = data.error || 'Extraction failed.';
      extractStatus.classList.remove('busy');
      extractBtn.disabled = false;
      await Promise.all([loadPapers(), loadRecords()]);
      return;
    }
    const dupNote = data.duplicates_removed ? ` (removed ${data.duplicates_removed} rounded duplicate${data.duplicates_removed === 1 ? '' : 's'})` : '';
    extractStatus.textContent = `Extracted ${data.extracted} of ${data.total} chunks — ${data.records} record${data.records === 1 ? '' : 's'} total${dupNote}.`;
    extractStatus.classList.remove('busy');
    await Promise.all([loadPapers(), loadRecords()]);
  } catch (e) {
    clearInterval(progressTimer);
    hideProgressBar();
    extractStatus.textContent = 'Extraction failed — check the server log.';
    extractStatus.classList.remove('busy');
    await Promise.all([loadPapers(), loadRecords()]);
  }
  extractBtn.disabled = false;
});

async function loadFoodGroups() {
  try {
    const res = await fetch('/api/food-groups');
    const data = await res.json();
    const groups = data.food_groups || [];
    EDIT_FIELDS.forEach(f => {
      if (f.key === 'target_food_group' || f.key === 'target_cross_price_food_group') f.options = groups;
    });
  } catch (e) {
    // leave options empty if unreachable — the fields still render as a
    // (currently choice-less) select rather than breaking the edit form
  }
}

loadFoodGroups();
loadPapers();
loadRecords();
