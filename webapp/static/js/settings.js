const SLOTS = ['cheap', 'main', 'validation'];
const STAGES = ['classification', 'detection', 'paper_metadata', 'extraction', 'validation'];
const form = document.getElementById('settings-form');
const saveBtn = document.getElementById('save-btn');
const saveStatus = document.getElementById('save-status');

const PROVIDER_LABELS = { anthropic: 'Anthropic (Claude)', openai: 'OpenAI (GPT)', google: 'Google (Gemini)' };
const EFFORT_LABELS = {
  none: 'None — reasoning off',
  minimal: 'Minimal — fastest, cheapest',
  low: 'Low — fast, cheap',
  medium: 'Medium — balanced',
  high: 'High — default on most models',
  xhigh: 'Extra high — hardest coding/agentic tasks',
  max: 'Max — highest capability, no constraints',
};

let availableModels = []; // [{id, label, provider, effort_levels: [...]}, ...]

function modelInfo(modelId) {
  return availableModels.find(m => m.id === modelId);
}

function populateModelSelect(selectEl, currentValue) {
  selectEl.innerHTML = '';
  const values = availableModels.map(m => m.id);
  // If the saved value isn't one of the known models (e.g. an older or
  // hand-edited entry), keep it selectable rather than silently discarding it.
  if (currentValue && !values.includes(currentValue)) {
    const opt = document.createElement('option');
    opt.value = currentValue;
    opt.textContent = `${currentValue} (unrecognized)`;
    selectEl.appendChild(opt);
  }
  // Group by provider so it's obvious which key each option needs.
  const providers = [...new Set(availableModels.map(m => m.provider))];
  providers.forEach(provider => {
    const group = document.createElement('optgroup');
    group.label = PROVIDER_LABELS[provider] || provider;
    availableModels.filter(m => m.provider === provider).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.label;
      group.appendChild(opt);
    });
    selectEl.appendChild(group);
  });
  if (currentValue) selectEl.value = currentValue;
}

function populateEffortSelect(selectEl, statusEl, modelId, currentValue) {
  const info = modelInfo(modelId);
  const levels = (info && info.effort_levels) || [];
  selectEl.innerHTML = '';
  if (!levels.length) {
    selectEl.disabled = true;
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(not applicable)';
    selectEl.appendChild(opt);
    statusEl.textContent = `${modelId || 'This model'} does not support an effort/thinking-level parameter.`;
    return;
  }
  selectEl.disabled = false;
  const defaultOpt = document.createElement('option');
  defaultOpt.value = '';
  defaultOpt.textContent = 'Model default';
  selectEl.appendChild(defaultOpt);
  levels.forEach(level => {
    const opt = document.createElement('option');
    opt.value = level;
    opt.textContent = EFFORT_LABELS[level] || level;
    selectEl.appendChild(opt);
  });
  selectEl.value = currentValue && levels.includes(currentValue) ? currentValue : '';
  statusEl.textContent = '';
}

function wireEffortToModel(slot) {
  const modelSelect = document.getElementById(`${slot}-model`);
  const effortSelect = document.getElementById(`${slot}-effort`);
  const statusEl = document.getElementById(`${slot}-effort-status`);
  const keyStatusEl = document.getElementById(`${slot}-key-status`);
  modelSelect.addEventListener('change', () => {
    // Model changed — re-derive which effort levels are valid; any
    // previously chosen level that no longer applies just resets to default.
    populateEffortSelect(effortSelect, statusEl, modelSelect.value, effortSelect.value);
    // Provider may also have changed — the key field now applies to a
    // different provider's key, so make that explicit rather than implying
    // whatever was typed still refers to the old provider.
    const info = modelInfo(modelSelect.value);
    const provider = info ? info.provider : null;
    keyStatusEl.textContent = provider
      ? `Enter a key for ${PROVIDER_LABELS[provider] || provider}. Switching models may switch which provider's key is needed.`
      : '';
  });
}

async function loadModels() {
  const res = await fetch('/api/models');
  const data = await res.json();
  availableModels = data.models || [];
}

async function loadSettings() {
  await loadModels();
  const res = await fetch('/api/settings');
  const data = await res.json();
  SLOTS.forEach(slot => {
    const s = data[slot] || {};
    const modelSelect = document.getElementById(`${slot}-model`);
    const effortSelect = document.getElementById(`${slot}-effort`);
    const effortStatusEl = document.getElementById(`${slot}-effort-status`);
    const keyInput = document.getElementById(`${slot}-key`);
    const keyStatusEl = document.getElementById(`${slot}-key-status`);
    populateModelSelect(modelSelect, s.model);
    populateEffortSelect(effortSelect, effortStatusEl, modelSelect.value, s.effort);
    keyInput.value = '';
    const providerLabel = PROVIDER_LABELS[s.provider] || s.provider || 'this provider';
    keyStatusEl.textContent = s.has_key
      ? `${providerLabel} key saved (${s.key_preview})`
      : `No ${providerLabel} key saved yet — falls back to the ${s.api_key_env || 'provider'} environment variable.`;
  });
  SLOTS.forEach(wireEffortToModel);
}

async function loadPrompts() {
  const res = await fetch('/api/prompts');
  const data = await res.json();
  STAGES.forEach(stage => {
    const p = (data.prompts && data.prompts[stage]) || { system: '', user: '' };
    document.getElementById(`${stage}-system`).value = p.system || '';
    document.getElementById(`${stage}-user`).value = p.user || '';
    const tokens = (data.tokens && data.tokens[stage]) || [];
    const tokenEl = document.querySelector(`[data-tokens-for="${stage}"]`);
    if (tokenEl) {
      tokenEl.innerHTML = tokens.length
        ? 'Available: ' + tokens.map(t => `<code>{{${t}}}</code>`).join(' ')
        : '';
    }
  });
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  saveBtn.disabled = true;
  saveStatus.textContent = 'Saving…';
  saveStatus.classList.add('busy');

  const settingsPayload = {};
  SLOTS.forEach(slot => {
    const model = document.getElementById(`${slot}-model`).value.trim();
    const key = document.getElementById(`${slot}-key`).value.trim();
    const effortSelect = document.getElementById(`${slot}-effort`);
    const effort = effortSelect.disabled ? '' : effortSelect.value;
    settingsPayload[slot] = { effort: effort || null };
    if (model) settingsPayload[slot].model = model;
    if (key) settingsPayload[slot].api_key = key;
  });

  const promptsPayload = {};
  STAGES.forEach(stage => {
    promptsPayload[stage] = {
      system: document.getElementById(`${stage}-system`).value,
      user: document.getElementById(`${stage}-user`).value,
    };
  });

  try {
    const [settingsRes, promptsRes] = await Promise.all([
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settingsPayload),
      }),
      fetch('/api/prompts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(promptsPayload),
      }),
    ]);
    if (!settingsRes.ok || !promptsRes.ok) throw new Error('save failed');
    saveStatus.textContent = 'Saved.';
    saveStatus.classList.remove('busy');
    await Promise.all([loadSettings(), loadPrompts()]);
  } catch (e) {
    saveStatus.textContent = 'Save failed — check the server log.';
    saveStatus.classList.remove('busy');
  }
  saveBtn.disabled = false;
});

loadSettings();
loadPrompts();
