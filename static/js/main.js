// Bootstraps the app: theme, tabs, sidebar drawer and each tab module.
import { $, $$, copyText, toast } from './util.js';
import { hydrateIcons, icon } from './icons.js';
import { currentModel, currentPin, emit, loadCatalog, loadServerConfig, on, save, server, settings } from './state.js';
import { initSidebar } from './sidebar.js';
import { initChat } from './chat.js';
import { initBenchmark } from './benchmark.js';
import { initArena } from './arena.js';
import { initSpeed } from './speed.js';
import { initLeaderboard } from './leaderboard.js';

const TABS = ['chat', 'benchmark', 'arena', 'speed', 'leaderboard'];

function applyTheme() {
  document.documentElement.dataset.theme = settings.theme;
  $('#themeToggle').innerHTML = icon(settings.theme === 'dark' ? 'sun' : 'moon');
}

function showTab(name) {
  if (!TABS.includes(name)) name = 'chat';
  settings.tab = name;
  save();
  for (const tab of $$('.tab')) tab.setAttribute('aria-selected', String(tab.dataset.tab === name));
  for (const id of TABS) $(`#tab-${id}`).hidden = id !== name;
  emit('tab', name);
}

function renderModelPill() {
  const model = currentModel();
  const pin = currentPin();
  $('#modelPillText').textContent = model ? `${model}${pin ? ` @ ${pin}` : ''}` : 'no model';
  $('#modelPill').title = model ? `${settings.provider} / ${model}${pin ? ` - pinned to the provider ${pin}` : ''}` : 'Pick a model in the sidebar';
}

function setSidebar(open) {
  $('#app').classList.toggle('sidebar-open', open);
}

async function start() {
  hydrateIcons();
  applyTheme();
  try {
    await loadServerConfig();
  } catch {
    toast('Could not reach the Model Lab server. Is it still running?', 'error', 10000);
  }

  initSidebar();
  initChat();
  initBenchmark();
  initArena();
  initSpeed();
  initLeaderboard();

  $('#themeToggle').addEventListener('click', () => {
    settings.theme = settings.theme === 'dark' ? 'light' : 'dark';
    save();
    applyTheme();
  });
  $('.tabs').addEventListener('click', (event) => {
    const tab = event.target.closest('.tab');
    if (tab) showTab(tab.dataset.tab);
  });
  $('#openSidebar').addEventListener('click', () => setSidebar(true));
  $('#closeSidebar').addEventListener('click', () => setSidebar(false));
  $('#backdrop').addEventListener('click', () => setSidebar(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') setSidebar(false);
  });

  // "Copy" buttons on code blocks anywhere in the app.
  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-copy-code]');
    if (!button) return;
    const code = button.closest('.codeblock')?.querySelector('code')?.textContent ?? '';
    if (await copyText(code)) {
      button.classList.add('copied');
      button.innerHTML = `${icon('check')}<span>Copied</span>`;
      setTimeout(() => {
        button.classList.remove('copied');
        button.innerHTML = `${icon('copy')}<span>Copy</span>`;
      }, 1400);
    }
  });

  on('model', renderModelPill);
  on('pin', renderModelPill);
  on('goto-tab', showTab);
  renderModelPill();
  showTab(settings.tab);
  loadCatalog();
  if (server.mock && !currentModel()) toast('Mock mode: try the model "mock/smart" - no API key needed.', 'info', 7000);
}

start();
