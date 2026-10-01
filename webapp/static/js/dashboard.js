const dashFilterBar = document.getElementById('dash-filter-bar');
const dashMeta = document.getElementById('dash-meta');
const table1Wrap = document.getElementById('table1-wrap');
const table2Wrap = document.getElementById('table2-wrap');
const table3Wrap = document.getElementById('table3-wrap');
const table4Wrap = document.getElementById('table4-wrap');

let availableCountries = [];

function escapeHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fmtNum(v, digits) {
  if (v === null || v === undefined) return '–';
  const n = Number(v);
  return Number.isNaN(n) ? '–' : n.toFixed(digits ?? 3);
}

function fmtPct(v) {
  if (v === null || v === undefined) return '–';
  return `${Number(v).toFixed(1)}%`;
}

function selectedCountries() {
  const sel = document.getElementById('country-filter');
  if (!sel || !sel.value) return [];
  return [sel.value];
}

function renderFilterBar(countries, selected) {
  availableCountries = countries || [];
  if (!availableCountries.length) {
    dashFilterBar.innerHTML = '<div class="dash-filter-empty">No country labels on extracted records yet.</div>';
    return;
  }
  const current = selected || (document.getElementById('country-filter') || {}).value || '';
  dashFilterBar.innerHTML = `
    <label class="dash-filter-label" for="country-filter">Country / region</label>
    <select id="country-filter" class="country-filter-select" title="Filter tables by country / region">
      <option value="">All countries</option>
      ${availableCountries.map(c => `
        <option value="${escapeHtml(c)}" ${c === current ? 'selected' : ''}>${escapeHtml(c)}</option>
      `).join('')}
    </select>`;

  document.getElementById('country-filter').addEventListener('change', () => loadDashboard());
}

function renderMeta(d) {
  const bits = [
    `${d.n_estimates} usable estimates`,
    `${d.n_missing_income} missing real per capita income`,
    `${d.n_requires_review} flagged for review`,
  ];
  if ((d.countries_filter || []).length) {
    bits.unshift(`Filter: ${(d.countries_filter || []).join(', ')}`);
  } else {
    bits.unshift('All countries');
  }
  dashMeta.innerHTML = bits.map(b => `<span class="dash-meta-chip">${escapeHtml(b)}</span>`).join('');
}

function renderTable1(t1) {
  if (!t1 || !(t1.products || []).length) {
    table1Wrap.innerHTML = '<div class="manuscript-empty"><div class="big">No elasticity estimates</div><div>Run Main AI extraction first.</div></div>';
    return;
  }
  const rows = t1.products.map(p => {
    const inc = p.income || {};
    const own = p.own_price || {};
    return `<tr>
      <td>${escapeHtml(p.product)}</td>
      <td>${inc.n || 0}</td>
      <td>${fmtNum(inc.mean)}</td>
      <td>${fmtNum(inc.sd)}</td>
      <td>${fmtNum(inc.min)}</td>
      <td>${fmtNum(inc.max)}</td>
      <td>${own.n || 0}</td>
      <td>${fmtNum(own.mean)}</td>
      <td>${fmtNum(own.sd)}</td>
      <td>${fmtNum(own.min)}</td>
      <td>${fmtNum(own.max)}</td>
    </tr>`;
  }).join('');
  table1Wrap.innerHTML = `
    <table class="chen-table">
      <thead>
        <tr>
          <th rowspan="2">Product group</th>
          <th colspan="5">Income elasticity</th>
          <th colspan="5">Own-price elasticity</th>
        </tr>
        <tr>
          <th>n</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th>
          <th>n</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="chen-footnote">Income estimates: ${t1.n_income || 0}; own-price estimates: ${t1.n_own_price || 0}.</div>`;
}

function renderTable2(t2) {
  if (!t2) {
    table2Wrap.innerHTML = '';
    return;
  }
  const income = t2.real_income || {};
  const sections = (t2.sections || []).map(sec => `
    <div class="chen-char-block">
      <div class="chen-char-title">${escapeHtml(sec.title)}</div>
      <div class="chen-table-wrap">
        <table class="chen-table chen-table-compact">
          <thead><tr><th>Category</th><th>n</th><th>%</th></tr></thead>
          <tbody>
            ${(sec.items || []).map(it => `
              <tr>
                <td>${escapeHtml(it.label)}</td>
                <td>${it.count}</td>
                <td>${fmtPct(it.pct)}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>
    </div>`).join('');

  table2Wrap.innerHTML = `
    <div class="dash-notice">
      <strong>Real per capita income</strong> —
      mean ${fmtNum(income.mean, 1)}
      (SD ${fmtNum(income.sd, 1)}; n=${income.n || 0} of ${t2.n_estimates} estimates).
      Missing income cells stay null and flag the Main AI row for review.
    </div>
    <div class="chen-char-grid">${sections}</div>`;
}

function coefCell(row) {
  if (!row) return '<td class="reg-cell-empty">-</td>';
  const t = row.t != null ? fmtNum(row.t, 2) : null;
  return `<td class="reg-cell">
    <div class="reg-coef">${fmtNum(row.coefficient)}${escapeHtml(row.stars || '')}</div>
    <div class="reg-se">${t != null && t !== '–' ? `[${t}]` : '[-]'}</div>
  </td>`;
}

function renderTable3(t3) {
  const eqs = (t3 && t3.equations) || {};
  const order = ['income', 'own_price', 'cross_price'];
  const labels = { income: 'Income', own_price: 'Own-price', cross_price: 'Cross-price' };

  const notes = [];
  order.forEach(k => {
    (eqs[k] && eqs[k].notes || []).forEach(n => notes.push(`${labels[k]}: ${n}`));
    if (eqs[k] && !eqs[k].ok && eqs[k].error) notes.push(`${labels[k]}: ${eqs[k].error}`);
  });

  const varOrder = [];
  order.forEach(k => {
    const rows = (eqs[k] && eqs[k].rows) || [];
    rows.forEach(r => {
      if (!varOrder.includes(r.variable)) varOrder.push(r.variable);
    });
  });

  if (!varOrder.length) {
    table3Wrap.innerHTML = `
      <div class="manuscript-empty">
        <div class="big">Meta-regression not available yet</div>
        <div>${escapeHtml(notes.join(' ') || 'Need more complete estimates (income covariates help).')}</div>
      </div>`;
    return;
  }

  const body = varOrder.map(v => `
    <tr>
      <td class="reg-var-name">${escapeHtml(v)}</td>
      ${order.map(k => {
        const row = ((eqs[k] && eqs[k].rows) || []).find(r => r.variable === v);
        return coefCell(row);
      }).join('')}
    </tr>`).join('');

  const foot = `
    <tr class="reg-stat-row"><td class="reg-var-name">N</td>
      ${order.map(k => `<td class="reg-cell">${eqs[k] && eqs[k].ok ? eqs[k].n_obs : '-'}</td>`).join('')}</tr>
    <tr class="reg-stat-row"><td class="reg-var-name">R²</td>
      ${order.map(k => `<td class="reg-cell">${eqs[k] && eqs[k].ok ? fmtNum(eqs[k].r_squared) : '-'}</td>`).join('')}</tr>
    <tr class="reg-stat-row"><td class="reg-var-name">Weighting</td>
      ${order.map(k => `<td class="reg-cell">${eqs[k] && eqs[k].ok ? escapeHtml(eqs[k].weighting) : '-'}</td>`).join('')}</tr>`;

  table3Wrap.innerHTML = `
    ${notes.length ? `<div class="dash-notice">${notes.map(n => `<div>${escapeHtml(n)}</div>`).join('')}</div>` : ''}
    <div class="chen-table-wrap">
      <table class="chen-table">
        <thead>
          <tr>
            <th>Equations 1,2,3</th>
            ${order.map(k => `<th>${labels[k]}</th>`).join('')}
          </tr>
        </thead>
        <tbody>${body}</tbody>
        <tbody class="reg-stats">${foot}</tbody>
      </table>
    </div>
    <div class="chen-footnote">Coefficient with significance stars; t-ratio in brackets. Stars: † p&lt;0.10, * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001.</div>`;
}

function renderTable4(t4) {
  if (!t4) {
    table4Wrap.innerHTML = '';
    return;
  }
  const notes = (t4.notes || []).map(n => `<div>${escapeHtml(n)}</div>`).join('');
  const rows = (t4.products || []).map(p => `
    <tr>
      <td>${escapeHtml(p.product)}</td>
      <td>${fmtNum(p.income_elasticity)}</td>
      <td>${fmtNum(p.own_price_elasticity)}</td>
    </tr>`).join('');
  table4Wrap.innerHTML = `
    ${notes ? `<div class="dash-notice">${notes}</div>` : ''}
    <div class="chen-table-wrap">
      <table class="chen-table">
        <thead>
          <tr>
            <th>Product group</th>
            <th>Predicted income elasticity</th>
            <th>Predicted own-price elasticity</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <div class="chen-footnote">
      Evaluated at ln(Y)=${fmtNum(t4.ln_income)}
      ${t4.income_level != null ? `(Y≈${fmtNum(t4.income_level, 1)})` : ''}.
    </div>`;
}

async function loadDashboard() {
  const countries = selectedCountries();
  const params = new URLSearchParams();
  countries.forEach(c => params.append('country', c));
  const res = await fetch(`/api/dashboard/summary${params.toString() ? '?' + params.toString() : ''}`);
  const d = await res.json();
  if (d.error) {
    dashMeta.innerHTML = `<span class="dash-meta-chip">${escapeHtml(d.error)}</span>`;
    return;
  }
  if (!document.getElementById('country-filter') && (d.countries_available || []).length) {
    renderFilterBar(d.countries_available, (d.countries_filter || [])[0] || '');
  } else if (document.getElementById('country-filter') && (d.countries_available || []).length
             && availableCountries.join('\0') !== (d.countries_available || []).join('\0')) {
    renderFilterBar(d.countries_available, selectedCountries()[0] || '');
  }
  renderMeta(d);
  renderTable1(d.table1);
  renderTable2(d.table2);
  renderTable3(d.table3);
  renderTable4(d.table4);
}

(async function init() {
  await loadDashboard();
})();
