const dashCards = document.getElementById('dash-cards');
const dashCoverage = document.getElementById('dash-coverage');
const dashRegEmpty = document.getElementById('dash-reg-empty');
const dashRegTable = document.getElementById('dash-reg-table');
const dashFilterBar = document.getElementById('dash-filter-bar');

let targets = { elasticities: [], products: [] };
let foodGroups = [];

function escapeHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fmtNum(v, digits) {
  if (v === null || v === undefined) return '–';
  const n = Number(v);
  return Number.isNaN(n) ? '–' : n.toFixed(digits ?? 3);
}

function pct(part, whole) {
  if (!whole) return '0%';
  return `${Math.round((100 * part) / whole)}%`;
}

function card(label, value, sub) {
  return `
    <div class="dash-card">
      <div class="dash-card-value">${value}</div>
      <div class="dash-card-label">${escapeHtml(label)}</div>
      ${sub ? `<div class="dash-card-sub">${sub}</div>` : ''}
    </div>`;
}

function truncate(s, n) {
  s = String(s ?? '');
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

// Renders a horizontal bar chart as inline SVG — a real chart (axis line,
// bars scaled proportionally to the largest count, value labels at the bar
// end) rather than a list of CSS progress bars.
function barChartSvg(items) {
  const rowH = 26;
  const gap = 6;
  const labelW = 148;
  const valueW = 32;
  const width = 460;
  const chartW = width - labelW - valueW;
  const maxCount = Math.max(...items.map(it => it.count), 1);
  const height = items.length * (rowH + gap) + gap;

  const bars = items.map((it, i) => {
    const y = gap + i * (rowH + gap);
    const barH = rowH - 8;
    const w = Math.max(3, Math.round((it.count / maxCount) * chartW));
    const label = truncate(it.value, 22);
    return `
      <text x="${labelW - 10}" y="${y + barH / 2 + 4}" text-anchor="end" class="dash-chart-label">${escapeHtml(label)}<title>${escapeHtml(it.value)}</title></text>
      <rect x="${labelW}" y="${y}" width="${w}" height="${barH}" rx="3" class="dash-chart-bar"><title>${escapeHtml(it.value)}: ${it.count}</title></rect>
      <text x="${labelW + w + 8}" y="${y + barH / 2 + 4}" class="dash-chart-value mono">${it.count}</text>`;
  }).join('');

  return `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" class="dash-chart-svg">
      <line x1="${labelW}" y1="0" x2="${labelW}" y2="${height}" class="dash-chart-axis" />
      ${bars}
    </svg>`;
}

function coverageBlock(title, items, total) {
  if (!items || !items.length) {
    return `
      <div class="dash-coverage-block">
        <div class="dash-coverage-title">${escapeHtml(title)}</div>
        <div class="dash-coverage-empty">No data yet</div>
      </div>`;
  }
  return `
    <div class="dash-coverage-block">
      <div class="dash-coverage-title">${escapeHtml(title)}</div>
      ${barChartSvg(items)}
    </div>`;
}

function renderRegTable(runs) {
  dashRegEmpty.hidden = runs.length !== 0;
  dashRegTable.hidden = runs.length === 0;
  if (!runs.length) { dashRegTable.innerHTML = ''; return; }

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
        ${runs.map(run => `<th>${escapeHtml(run.label)}</th>`).join('')}
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
      ${statRow('Weighted (WLS)', run => run.weight_by_precision ? 'yes' : 'no')}
    </tbody>`;

  dashRegTable.innerHTML = headerHtml + `<tbody>${bodyRows}</tbody>` + footHtml;
}

async function loadTargets() {
  try {
    const res = await fetch('/api/targets');
    targets = await res.json();
  } catch (e) {
    targets = { elasticities: [], products: [] };
  }
}

async function loadFoodGroupsList() {
  try {
    const res = await fetch('/api/food-groups');
    foodGroups = (await res.json()).food_groups || [];
  } catch (e) {
    foodGroups = [];
  }
}

// Lets the user restrict every stat/chart on the page to one or more of the
// elasticity types / products they originally asked for on the upload page
// (targets.json) — e.g. only "Income elasticity", or only "Maize" — same
// checkbox-from-targets-list pattern as the Regression page's filters. Food
// group is a fixed 8-value vocab (not project-specific), so it's always
// offered once loaded, independent of whether targets.json has anything in it.
function renderFilterBar() {
  if (!targets.elasticities.length && !targets.products.length && !foodGroups.length) {
    dashFilterBar.innerHTML = '';
    return;
  }
  const group = (title, field, values) => values.length ? `
    <div class="dash-filter-group">
      <div class="dash-filter-title">${escapeHtml(title)}</div>
      <div class="dash-filter-options">
        ${values.map(v => `
          <label class="dash-filter-chip">
            <input type="checkbox" class="dash-filter-checkbox" data-field="${field}" value="${escapeHtml(v)}">
            ${escapeHtml(v)}
          </label>`).join('')}
      </div>
    </div>` : '';

  dashFilterBar.innerHTML = `
    ${group('Elasticity type', 'elasticity_type', targets.elasticities)}
    ${group('Product', 'product', targets.products)}
    ${group('Food group', 'food_group', foodGroups)}
    <button class="btn-secondary dash-filter-clear" id="dash-filter-clear" hidden>Clear filters</button>`;

  dashFilterBar.querySelectorAll('.dash-filter-checkbox').forEach(cb => {
    cb.addEventListener('change', () => { updateClearButton(); loadDashboard(); });
  });
  const clearBtn = document.getElementById('dash-filter-clear');
  clearBtn.addEventListener('click', () => {
    dashFilterBar.querySelectorAll('.dash-filter-checkbox:checked').forEach(cb => { cb.checked = false; });
    updateClearButton();
    loadDashboard();
  });
}

function updateClearButton() {
  const clearBtn = document.getElementById('dash-filter-clear');
  if (!clearBtn) return;
  const anyChecked = dashFilterBar.querySelectorAll('.dash-filter-checkbox:checked').length > 0;
  clearBtn.hidden = !anyChecked;
}

function selectedFilterParams() {
  const params = new URLSearchParams();
  dashFilterBar.querySelectorAll('.dash-filter-checkbox:checked').forEach(cb => {
    params.append(cb.dataset.field, cb.value);
  });
  return params;
}

async function loadDashboard() {
  const params = selectedFilterParams();
  const res = await fetch(`/api/dashboard/summary${params.toString() ? '?' + params.toString() : ''}`);
  const d = await res.json();
  const filtered = params.toString().length > 0;

  const coef = d.coefficient_summary || {};
  dashCards.innerHTML = [
    card('Papers in corpus', d.papers_total, `${d.papers_with_records} with at least one extracted estimate`),
    card('Extracted estimates', d.records_total,
      filtered
        ? `of ${d.records_total_unfiltered} total · ${d.records_requires_review} flagged for review`
        : `${d.records_requires_review} flagged for review (${pct(d.records_requires_review, d.records_total)})`),
    card('Chunks screened', `${d.chunks_detected_positive}/${d.chunks_classified}`,
      `passed detection · ${d.chunks_kept} kept of ${d.chunks_total} total`),
    card('Manually edited', d.records_manually_edited,
      `of ${d.records_total} estimate${d.records_total === 1 ? '' : 's'} (${pct(d.records_manually_edited, d.records_total)})`),
    card('Mean coefficient', fmtNum(coef.mean),
      coef.n ? `median ${fmtNum(coef.median)} · range [${fmtNum(coef.min)}, ${fmtNum(coef.max)}] · n=${coef.n}` : 'no numeric estimates yet'),
    card('Regression specs run', (d.regression_runs || []).length, 'saved on the Regression page'),
  ].join('');

  dashCoverage.innerHTML = [
    coverageBlock('Products', d.products, d.records_total),
    coverageBlock('Food groups (standard 8-way)', d.food_groups, d.records_total),
    coverageBlock('Elasticity types', d.elasticity_types, d.records_total),
    coverageBlock('Country / region', d.countries, d.records_total),
    coverageBlock('Data source', d.data_sources, d.records_total),
    coverageBlock('Model type', d.model_types, d.records_total),
    coverageBlock('Specification status', d.specification_status, d.records_total),
  ].join('');

  // Saved regression runs already have their own filters baked in from when
  // they were fit, so the dashboard's filter bar doesn't re-filter them —
  // it only narrows the summary stats/charts above, which are always
  // computed fresh from records.json.
  renderRegTable(d.regression_runs || []);
}

(async function init() {
  await loadTargets();
  await loadFoodGroupsList();
  renderFilterBar();
  await loadDashboard();
})();
