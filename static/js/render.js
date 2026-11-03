// Pieces of UI shared by Chat, Arena and Speed: metric rows, thinking blocks, error notices.
import { esc, fmtCost, fmtInt, fmtSeconds, fmtTps } from './util.js';
import { icon } from './icons.js';
import { renderMarkdown } from './markdown.js';

/** Metrics line shown under a reply. `result` is what runCompletion returns. */
export function metricsHTML(result) {
  const m = result.metrics;
  const parts = [];
  if (m) {
    const approx = m.tokens_estimated ? '~' : '';
    parts.push(`<span title="Time to first token">${icon('clock')}<b>${fmtSeconds(m.ttft)}</b> first token</span>`);
    parts.push(`<span title="Output tokens per second">${icon('zap')}<b>${fmtTps(m.tps)}</b> tok/s</span>`);
    parts.push(`<span title="Total time">${icon('activity')}<b>${fmtSeconds(m.total)}</b> total</span>`);
    parts.push(`<span title="Tokens in / out${m.tokens_estimated ? ' (estimated)' : ''}">${icon('hash')}<b>${approx}${fmtInt(m.tokens_in)}</b> in / <b>${approx}${fmtInt(m.tokens_out)}</b> out</span>`);
    const costTitle = m.cost == null ? 'No price information for this model'
      : m.cost_estimated ? `Estimated cost: an estimate sent by the provider, or worked out from the model price list${m.served_by ? ` (served by ${esc(m.served_by)})` : ''}`
        : `Cost reported by the provider${m.served_by ? ` (served by ${esc(m.served_by)})` : ''}`;
    parts.push(`<span title="${costTitle}">${icon('dollar')}<b>${fmtCost(m.cost, m.cost_estimated)}</b></span>`);
  } else if (result.done && result.elapsed != null) {
    parts.push(`<span>${icon('activity')}<b>${fmtSeconds(result.elapsed)}</b></span>`);
  }
  if (result.truncated) parts.push('<span class="badge badge-warn" title="finish_reason=length">cut off at max tokens</span>');
  if (result.aborted) parts.push('<span class="badge badge-neutral">stopped</span>');
  return parts.join('');
}

export function errorHTML(error) {
  return `${icon('alert')}<span>${esc(error.message || 'Something went wrong.')}</span>`;
}

export function thinkingHTML() {
  return `<details class="thinking" hidden>
    <summary>${icon('chevron-right', 'chev')}${icon('brain')}<span data-role="thinking-label">Thinking...</span></summary>
    <div class="thinking-body" dir="auto" data-role="thinking-body"></div>
  </details>`;
}

/**
 * Paint a (possibly still streaming) reply into a container that holds a thinking block from
 * thinkingHTML() and an element with data-role="body". Auto-opens the thinking block while the
 * model thinks and folds it away once the answer starts; after that the user is in control.
 */
export function paintReply(root, result) {
  const details = root.querySelector('details.thinking');
  const body = root.querySelector('[data-role="body"]');
  if (result.reasoning) {
    details.hidden = false;
    details.querySelector('[data-role="thinking-body"]').textContent = result.reasoning.trim();
    const thinking = !result.content && !result.done;
    const seconds = result.thinkingMs != null ? ` for ${(result.thinkingMs / 1000).toFixed(1)} s` : '';
    details.querySelector('[data-role="thinking-label"]').textContent = thinking ? 'Thinking...' : `Thought${seconds}`;
    if (thinking && !details.dataset.opened) {
      details.open = true;
      details.dataset.opened = '1';
    } else if (!thinking && details.dataset.opened && !details.dataset.closed) {
      details.open = false;
      details.dataset.closed = '1';
    }
  }
  body.innerHTML = renderMarkdown(result.content);
  body.classList.toggle('cursor', !result.done && !result.error);
}
