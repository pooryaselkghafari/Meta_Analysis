const dvSelect = document.getElementById('dv-select');
const ivGroups = document.getElementById('iv-groups');
const filterGroups = document.getElementById('filter-groups');
const weightCheckbox = document.getElementById('weight-checkbox');
const labelInput = document.getElementById('label-input');
const runBtn = document.getElementById('run-btn');
const runStatus = document.getElementById('run-status');
const regTableWrap = document.getElementById('regression-table-wrap');
const regTable = document.getElementById('reg-table');
const regEmpty = document.getElementById('regression-empty');

let fieldCatalog = { fields: {}, default_dv: 'coefficient' };
let targets = { elasticities: [], products: [] };
let runs = [];

function escapeHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Fields worth offering as a "restrict to…" filter — the categorical ones a
// user would plausibly want to subset by before regressing (e.g. only
// Maize records, or only OLS-model papers). Numeric/boolean fields aren't
// offered as filters since there's no natural fixed value list for them.
// target_elasticity_type/target_product are rendered as checkbox lists
// sourced from targets.json (the elasticities/products the user originally
// asked to extract, set on the Corpus/upload page) rather than free text —
// so "only Income elasticity" is a checkbox tick, not typing a value that
// has to exactly match. The rest don't come from a user-defined list, so
// they stay free text.
const FILTERABLE = [
  'target_product', 'target_food_group', 'target_elasticity_type', 'model_type',
  'countries_region', 'specification_status', 'source_type',
];
const TARGET_LIST_FIELDS = { target_elasticity_type: 'elasticities', target_product: 'products' };

async function loadTargets() {
  try {
    const res = await fetch('/api/targets');
    targets = await res.json();
  } catch (e) {
    targets = { elasticities: [], products: [] };
  }
}

async function loadFieldCatalog() {
  const res = await fetch('/api/regression/fields');
  fieldCatalog = await res.json();

  const fields = fieldCatalog.fields;
  const keys = Object.keys(fields);

  dvSelect.innerHTML = keys.map(k =>
    `<option value="${k}" ${k === fieldCatalog.default_dv ? 'selected' : ''}>${escapeHtml(fields[k].label)}</option>`
  ).join('');

  const byKind = { numeric: [], categorical: [], boolean: [] };
  keys.forEach(k => { if (byKind[fields[k].kind]) byKind[fields[k].kind].push(k); });

  const groupHtml = (title, list) => list.length ? `
    <div class="iv-group">
      <div class="iv-group-title">${title}</div>
      ${list.map(k => `
        <label class="iv-checkbox">
          <input type="checkbox" class="iv-input" value="${k}"> ${escapeHtml(fields[k].label)}
        </label>`).join('')}
    </div>` : '';

  ivGroups.innerHTML =
    groupHtml('Numeric', byKind.numeric) +
    groupHtml('Categorical', byKind.categorical) +
    groupHtml('Boolean', byKind.boolean);

  // Uncheck the currently-selected DV automatically if the user later picks
  // it as an IV too (the backend rejects that combination anyway).
  dvSelect.addEventListener('change', syncIvAvailability);
  syncIvAvailability();

  filterGroups.innerHTML = FILTERABLE.filter(k => fields[k]).map(k => {
    // A fixed vocab from the field catalog itself (e.g. target_food_group's
    // 8 standard groups) takes priority over a project-defined targets.json
    // list — both render as checkboxes the same way, just from a different
    // source of truth.
    const targetKey = TARGET_LIST_FIELDS[k];
    const targetList = (fields[k].options && fields[k].options.length)
      ? fields[k].options
      : (targetKey ? (targets[targetKey] || []) : []);
    if (targetList.length) {
      return `
        <div class="iv-group">
          <div class="iv-group-title">${escapeHtml(fields[k].label)}</div>
          ${targetList.map(v => `
            <label class="iv-checkbox">
              <input type="checkbox" class="filter-checkbox" data-field="${k}" value="${escapeHtml(v)}"> ${escapeHtml(v)}
            </label>`).join('')}
        </div>`;
    }
    // Fall back to free text if the user hasn't defined a target list for
    // this field yet (e.g. targets.json is empty) — still filterable, just
    // without a fixed set of choices to tick.
    return `
      <div class="iv-group">
        <div class="iv-group-title">${escapeHtml(fields[k].label)}</div>
        <input type="text" class="field-input mono filter-input" data-field="${k}"
               placeholder="comma-separated values, blank = all">
      </div>`;
  }).join('');
}

function syncIvAvailability() {
  const dv = dvSelect.value;
  ivGroups.querySelectorAll('.iv-input').forEach(cb => {
    if (cb.value === dv) { cb.checked = false; cb.disabled = true; }
    else cb.disabled = false;
  });
}

function selectedIvs() {
  return [...ivGroups.querySelectorAll('.iv-input:checked')].map(cb => cb.value);
}

function selectedFilters() {
  const filters = {};
  filterGroups.querySelectorAll('.filter-input').forEach(inp => {
    const v = inp.value.trim();
    if (v) filters[inp.dataset.field] = v.split(',').map(s => s.trim()).filter(Boolean);
  });
  filterGroups.querySelectorAll('.filter-checkbox:checked').forEach(cb => {
    (filters[cb.dataset.field] = filters[cb.dataset.field] || []).push(cb.value);
  });
  return filters;
}

function fmtNum(v, digits) {
  if (v === null || v === undefined) return '–';
  const n = Number(v);
  return Number.isNaN(n) ? '–' : n.toFixed(digits ?? 3);
}

async function loadRuns() {
  const res = await fetch('/api/regression/runs');
  const data = await res.json();
  runs = data.runs || [];
  renderRuns();
}

async function runRegression() {
  const dv = dvSelect.value;
  const ivs = selectedIvs();
  if (!ivs.length) {
    runStatus.textContent = 'Pick at least one independent variable.';
    return;
  }
  runBtn.disabled = true;
  runStatus.textContent = 'Running…';
  runStatus.classList.add('busy');
  try {
    const res = await fetch('/api/regression/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dv, ivs,
        weight_by_precision: weightCheckbox.checked,
        filters: selectedFilters(),
        label: labelInput.value,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      runStatus.textContent = data.error || 'Regression failed.';
      runStatus.classList.remove('busy');
      runBtn.disabled = false;
      return;
    }
    runStatus.textContent = `Fit — N=${data.n_obs}, R²=${fmtNum(data.r_squared, 3)}.`;
    runStatus.classList.remove('busy');
    labelInput.value = '';
    await loadRuns();
  } catch (e) {
    runStatus.textContent = 'Regression failed — check the server log.';
    runStatus.classList.remove('busy');
  }
  runBtn.disabled = false;
}

async function removeRun(runId) {
  if (!confirm('Remove this column from the table?')) return;
  await fetch(`/api/regression/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
  await loadRuns();
}

function renderRuns() {
  regEmpty.hidden = runs.length !== 0;
  regTable.hidden = runs.length === 0;
  if (!runs.length) { regTable.innerHTML = ''; return; }

  // Union of every variable name appearing in any run, in first-seen order,
  // with Intercept always pinned last — matches how a published table lists
  // moderators first and the constant at the bottom.
  const varOrder = [];
  runs.forEach(run => {
    run.rows.forEach(r => {
      if (r.variable !== 'Intercept' && !varOrder.includes(r.variable)) varOrder.push(r.variable);
    });
  });
  if (runs.some(run => run.rows.some(r => r.variable === 'Intercept'))) varOrder.push('Intercept');

  const headerHtml = `
    <thead>
      <tr>
        <th>Variable</th>
        ${runs.map(run => `
          <th>
            <div class="reg-col-head">
              <span>${escapeHtml(run.label)}</span>
              <button class="reg-col-remove" data-id="${escapeHtml(run.id)}" title="Remove this column">&times;</button>
            </div>
          </th>`).join('')}
      </tr>
    </thead>`;

  const cellFor = (run, varName) => {
    const row = run.rows.find(r => r.variable === varName);
    if (!row) return '<td class="reg-cell-empty">–</td>';
    return `<td class="reg-cell">
      <div class="reg-coef">${fmtNum(row.coefficient)}${escapeHtml(row.stars)}</div>
      <div class="reg-se">(${fmtNum(row.std_error)})</div>
    </td>`;
  };

  const bodyRows = varOrder.map(v => `
    <tr>
      <td class="reg-var-name">${escapeHtml(v)}</td>
      ${runs.map(run => cellFor(run, v)).join('')}
    </tr>`).join('');

  const statRow = (label, fmt) => `
    <tr class="reg-stat-row">
      <td class="reg-var-name">${label}</td>
      ${runs.map(run => `<td class="reg-cell mono">${fmt(run)}</td>`).join('')}
    </tr>`;

  const footHtml = `
    <tbody class="reg-stats">
      ${statRow('N', run => run.n_obs)}
      ${statRow('R²', run => fmtNum(run.r_squared))}
      ${statRow('Adj. R²', run => fmtNum(run.adj_r_squared))}
      ${statRow('F-statistic', run => run.f_statistic != null ? `${fmtNum(run.f_statistic, 2)}${run.f_pvalue != null && run.f_pvalue < 0.01 ? '***' : (run.f_pvalue != null && run.f_pvalue < 0.05 ? '**' : (run.f_pvalue != null && run.f_pvalue < 0.10 ? '*' : ''))}` : '–')}
      ${statRow('Weighted (WLS)', run => run.weight_by_precision ? 'yes' : 'no')}
    </tbody>`;

  regTable.innerHTML = headerHtml + `<tbody>${bodyRows}</tbody>` + footHtml;

  regTable.querySelectorAll('.reg-col-remove').forEach(btn => {
    btn.addEventListener('click', () => removeRun(btn.dataset.id));
  });
}

runBtn.addEventListener('click', runRegression);

(async function init() {
  await loadTargets();
  await loadFieldCatalog();
  await loadRuns();
})();
