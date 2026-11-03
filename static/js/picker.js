// Model picker: a text box that takes a model ID or a pasted model-page URL, with autocomplete from
// the provider's public list (OpenRouter's, or Hugging Face's router list) and model info underneath.
import { $, el, esc, fmtContext, fmtPrice } from './util.js';
import { PROVIDERS, catalog, hfModelInfo, modelInfo, on, parseModelInput } from './state.js';

const MAX_RESULTS = 40;

function search(query, provider) {
  const models = catalog[provider] || [];
  // A Hugging Face ID may end in a route (":novita", ":cheapest"): search on the part before it.
  const q = (provider === 'huggingface' ? query.split(':')[0] : query).trim().toLowerCase();
  if (!q) return models.slice(0, MAX_RESULTS);
  const terms = q.split(/\s+/);
  const hits = [];
  for (const model of models) {
    const id = model.id.toLowerCase();
    const name = (model.name || '').toLowerCase();
    if (!terms.every((term) => id.includes(term) || name.includes(term))) continue;
    const short = id.split('/')[1] || id;
    const rank = id === q ? 0 : id.startsWith(q) ? 1 : short.startsWith(q) ? 2 : id.includes(q) ? 3 : 4;
    hits.push([rank, model]);
  }
  hits.sort((a, b) => a[0] - b[0] || a[1].id.localeCompare(b[1].id));
  return hits.slice(0, MAX_RESULTS).map((hit) => hit[1]);
}

function markMatch(id, query) {
  const q = query.trim().toLowerCase();
  const at = q && !/\s/.test(q) ? id.toLowerCase().indexOf(q) : -1;
  if (at < 0) return esc(id);
  return `${esc(id.slice(0, at))}<mark>${esc(id.slice(at, at + q.length))}</mark>${esc(id.slice(at + q.length))}`;
}

// Tokens can cost nothing while each request, image, search or audio clip is still billed: that is not "free".
const OTHER_FEES = '<span class="badge badge-warn" title="Tokens cost nothing, but this model is billed another way (per request, per image, per audio clip, ...). Check its page on OpenRouter.">other fees</span>';

function priceText(model) {
  if (model.providers) {
    // Hugging Face: each provider sets its own price, so the list shows where prices start.
    const count = `${model.providers.length} provider${model.providers.length === 1 ? '' : 's'}`;
    if (model.price_in == null) return `<span>${count} · no price listed</span>`;
    return `<span>from ${fmtPrice(model.price_in)} / ${fmtPrice(model.price_out)} · ${count}</span>`;
  }
  if (model.free) return '<span class="badge badge-free">free</span>';
  return `<span>${fmtPrice(model.price_in)} / ${fmtPrice(model.price_out)}</span>${model.other_fees ? OTHER_FEES : ''}`;
}

const ROUTE_HINT = '<span>Optional route suffix: <code class="inline">:fastest</code> <code class="inline">:cheapest</code> or a provider name</span>';

/** The info line for a Hugging Face model: one provider's numbers when the ID names one, else the range. */
function hfInfoHTML(model) {
  const found = hfModelInfo(model);
  if (!found) {
    const suffix = model.includes(':') ? `<span>Routed via <b style="color:var(--text)">${esc(model.split(':').pop())}</b></span>` : ROUTE_HINT;
    if (catalog.huggingface.length) return `<span style="color:var(--amber)">Not in Hugging Face's router list - double-check the ID.</span>${suffix}`;
    return catalog.errors.huggingface ? `<span>${esc(catalog.errors.huggingface)}</span>${suffix}` : suffix;
  }
  const names = found.providers.map((p) => p.name).join(', ');
  if (found.route) {
    const route = found.route;
    const price = route.price_in == null ? '<span>no price listed - cost will show a dash unless the provider reports one</span>'
      : `<span class="kv">In <b>${fmtPrice(route.price_in)}</b></span><span class="kv">Out <b>${fmtPrice(route.price_out)}</b></span><span>per 1M tokens</span>`;
    return `<span>Routed via <b style="color:var(--text)">${esc(route.name)}</b></span>
      <span class="kv">Context <b>${fmtContext(route.context_length)}</b></span>${price}`;
  }
  const price = found.cheapest
    ? `<span class="kv">From <b>${fmtPrice(found.cheapest.price_in)}</b> in / <b>${fmtPrice(found.cheapest.price_out)}</b> out</span><span>per 1M tokens (${esc(found.cheapest.name)})</span>`
    : '<span>no price listed</span>';
  const routed = found.suffix ? `<span>Routed <b style="color:var(--text)">${esc(found.suffix)}</b></span>` : '';
  return `${routed}<span class="kv">Context up to <b>${fmtContext(found.context_length)}</b></span>${price}
    <span title="${esc(names)}">${found.providers.length} provider${found.providers.length === 1 ? '' : 's'}: ${esc(names)}</span>`;
}

/**
 * @param host      element to render into
 * @param options   { provider, model, showProvider, onChange({provider, model}) }
 * showProvider adds a compact provider dropdown; without it the caller controls the provider.
 */
export function createPicker(host, { provider = 'openrouter', model = '', showProvider = false, onChange }) {
  const state = { provider, model };
  let results = [];
  let active = -1;

  host.innerHTML = `
    <div class="picker">
      <div class="picker-row">
        ${showProvider ? `<select class="select" data-role="provider" aria-label="Provider">${
          PROVIDERS.map((p) => `<option value="${p.id}">${p.label}</option>`).join('')}</select>` : ''}
        <div class="ac">
          <input class="input mono" type="text" data-role="model" dir="ltr" spellcheck="false" autocomplete="off"
                 autocapitalize="off" aria-label="Model ID or URL" role="combobox" aria-expanded="false">
          <div class="ac-list" role="listbox" hidden></div>
        </div>
      </div>
      <div class="model-info" data-role="info"></div>
    </div>`;

  const input = $('[data-role="model"]', host);
  const select = $('[data-role="provider"]', host);
  const listBox = $('.ac-list', host);
  const info = $('[data-role="info"]', host);

  function renderInfo() {
    const example = PROVIDERS.find((p) => p.id === state.provider)?.example || '';
    input.placeholder = `${example} or a model URL`;
    if (!state.model) {
      info.innerHTML = state.provider === 'huggingface'
        ? '<span>Type to search, or paste a huggingface.co link.</span>'
        : '<span>Type to search, or paste an openrouter.ai link.</span>';
      return;
    }
    if (state.provider === 'huggingface') {
      info.innerHTML = hfInfoHTML(state.model);
      return;
    }
    const found = modelInfo('openrouter', state.model);
    if (found) {
      info.innerHTML = `
        <span class="kv">Context <b>${fmtContext(found.context_length)}</b></span>
        ${found.free ? '<span class="badge badge-free">free</span>'
          : `<span class="kv">In <b>${fmtPrice(found.price_in)}</b></span><span class="kv">Out <b>${fmtPrice(found.price_out)}</b></span><span>per 1M tokens</span>${found.other_fees ? OTHER_FEES : ''}`}`;
    } else if (catalog.loaded && catalog.openrouter.length) {
      info.innerHTML = '<span style="color:var(--amber)">Not in OpenRouter\'s public list - double-check the ID.</span>';
    } else {
      info.innerHTML = catalog.error ? `<span>${esc(catalog.error)}</span>` : '';
    }
  }

  function closeList() {
    listBox.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    active = -1;
  }

  function openList() {
    if (!(catalog[state.provider] || []).length) return closeList();
    results = search(input.value, state.provider);
    const typed = input.value.trim().split(state.provider === 'huggingface' ? ':' : '\n')[0];
    if (results.length === 1 && results[0].id === typed) return closeList();
    if (!results.length) {
      const where = state.provider === 'huggingface' ? "Hugging Face's router list" : 'the OpenRouter list';
      listBox.innerHTML = `<div class="ac-empty">No matching models in ${where}. You can still use any ID.</div>`;
    } else {
      listBox.innerHTML = results.map((model, index) => `
        <div class="ac-item" role="option" data-index="${index}" aria-selected="${index === active}">
          <div class="ac-id">${markMatch(model.id, input.value)}</div>
          ${model.name && model.name !== model.id ? `<div class="ac-name">${esc(model.name)}</div>` : ''}
          <div class="ac-meta"><span>${fmtContext(model.context_length)} ctx</span>${priceText(model)}</div>
        </div>`).join('');
    }
    listBox.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }

  function moveActive(step) {
    if (listBox.hidden) { openList(); return; }
    if (!results.length) return;
    active = (active + step + results.length) % results.length;
    listBox.querySelectorAll('.ac-item').forEach((item, index) => {
      item.setAttribute('aria-selected', String(index === active));
      if (index === active) item.scrollIntoView({ block: 'nearest' });
    });
  }

  function commit(nextProvider, nextModel, { fromTyping = false } = {}) {
    const changed = nextProvider !== state.provider || nextModel !== state.model;
    state.provider = nextProvider;
    state.model = nextModel;
    if (select) select.value = nextProvider;
    if (!fromTyping && input.value !== nextModel) input.value = nextModel;
    renderInfo();
    if (changed) onChange?.({ ...state });
  }

  function pick(index) {
    const model = results[index];
    if (!model) return;
    input.value = model.id;
    commit(state.provider, model.id);
    closeList();
  }

  input.addEventListener('input', () => {
    const parsed = parseModelInput(input.value);
    if (parsed.provider && parsed.model) {
      // A pasted URL: switch provider and keep just the model ID in the box.
      input.value = parsed.model;
      commit(parsed.provider, parsed.model);
      closeList();
      return;
    }
    if (parsed.provider) {
      // A URL that has no model in it yet (still being typed): leave the text alone.
      commit(state.provider, '', { fromTyping: true });
      closeList();
      return;
    }
    commit(state.provider, parsed.model, { fromTyping: true });
    active = -1;
    openList();
  });
  input.addEventListener('focus', openList);
  input.addEventListener('blur', () => {
    setTimeout(closeList, 120);
    if (input.value !== state.model) input.value = state.model;   // tidy stray whitespace
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowDown') { event.preventDefault(); moveActive(1); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); moveActive(-1); }
    else if (event.key === 'Enter' && !listBox.hidden && active >= 0) { event.preventDefault(); pick(active); }
    else if (event.key === 'Escape' && !listBox.hidden) { event.stopPropagation(); closeList(); }
  });
  listBox.addEventListener('mousedown', (event) => {
    event.preventDefault();   // keep focus in the input so blur does not fire first
    const item = event.target.closest('.ac-item');
    if (item) pick(Number(item.dataset.index));
  });
  select?.addEventListener('change', () => {
    commit(select.value, state.model);
    closeList();
  });
  on('catalog', renderInfo);

  if (select) select.value = state.provider;
  input.value = state.model;
  renderInfo();

  return {
    get: () => ({ ...state }),
    /** Set the value from outside without firing onChange. */
    set(nextProvider, nextModel) {
      state.provider = nextProvider;
      state.model = nextModel || '';
      if (select) select.value = nextProvider;
      if (input.value !== state.model) input.value = state.model;   // avoid moving the caret mid-typing
      renderInfo();
    },
    focus: () => input.focus(),
  };
}
