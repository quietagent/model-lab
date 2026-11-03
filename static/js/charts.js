// Tiny SVG bar chart (no chart library). Colours come from CSS so both themes work.
import { esc } from './util.js';

function niceMax(value) {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const scaled = value / power;
  const step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return step * power;
}

/**
 * @param bars    [{ label, value (number | null), failed }]
 * @param format  (number) => string for value labels
 * @param axisFormat optional formatter for the y-axis labels (defaults to format)
 * @param average optional number drawn as a dashed line
 */
export function barChart({ bars, format, axisFormat = format, average = null }) {
  const width = 720;
  const height = 240;
  const pad = { top: 22, right: 14, bottom: 28, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const peak = Math.max(0, ...bars.map((bar) => bar.value ?? 0));
  const top = niceMax(peak * 1.08);
  const y = (value) => pad.top + plotH - (value / top) * plotH;
  const slot = plotW / Math.max(bars.length, 1);
  const barW = Math.min(64, slot * 0.62);

  let svg = '';
  for (let i = 0; i <= 4; i += 1) {
    const value = (top / 4) * i;
    const lineY = y(value);
    svg += `<line class="${i === 0 ? 'axis' : 'grid'}" x1="${pad.left}" x2="${width - pad.right}" y1="${lineY}" y2="${lineY}"/>`;
    svg += `<text x="${pad.left - 8}" y="${lineY + 4}" text-anchor="end">${esc(axisFormat(value))}</text>`;
  }
  bars.forEach((bar, index) => {
    const centre = pad.left + slot * index + slot / 2;
    const x = centre - barW / 2;
    svg += `<text x="${centre}" y="${height - 9}" text-anchor="middle">${esc(bar.label)}</text>`;
    if (bar.failed || bar.value == null) {
      svg += `<rect class="bar fail" x="${x}" y="${pad.top + plotH - 6}" width="${barW}" height="6" rx="3"/>`;
      svg += `<text class="bar-text" x="${centre}" y="${pad.top + plotH - 12}" text-anchor="middle" style="fill:var(--red)">failed</text>`;
      return;
    }
    const barY = y(bar.value);
    const barH = Math.max(2, pad.top + plotH - barY);
    svg += `<rect class="bar" x="${x}" y="${barY}" width="${barW}" height="${barH}" rx="4"><title>${esc(bar.label)}: ${esc(format(bar.value))}</title></rect>`;
    svg += `<text class="bar-text" x="${centre}" y="${barY - 6}" text-anchor="middle">${esc(format(bar.value))}</text>`;
  });
  if (average != null && bars.length > 1) {
    const lineY = y(average);
    svg += `<line class="avg-line" x1="${pad.left}" x2="${width - pad.right}" y1="${lineY}" y2="${lineY}"/>`;
    svg += `<text class="avg-text" x="${width - pad.right}" y="${lineY - 5}" text-anchor="end">avg ${esc(format(average))}</text>`;
  }
  return `<div class="chart"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Bar chart">${svg}</svg></div>`;
}
