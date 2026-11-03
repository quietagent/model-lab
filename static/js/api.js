// Thin wrappers around the backend. API keys travel only in POST bodies, never in URLs.

async function readError(response) {
  let message = `Request failed (${response.status})`;
  let data = null;
  try {
    data = await response.json();
    if (typeof data.detail === 'string') message = data.detail;
    else if (Array.isArray(data.detail) && data.detail[0]?.msg) message = data.detail[0].msg;
    else if (data.message) message = data.message;
  } catch { /* not JSON */ }
  const error = new Error(message);
  error.status = response.status;
  error.data = data;
  return error;
}

export async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw await readError(response);
  return response.json();
}

export async function sendJSON(method, url, body) {
  const response = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw await readError(response);
  return response.json();
}

/** POST a JSON body and read the reply as server-sent events, calling onEvent for each. */
export async function streamSSE(url, body, onEvent, signal) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await readError(response);
  await readEvents(response, onEvent);
}

/** GET a stream of server-sent events (a benchmark job's progress), calling onEvent for each. */
export async function streamEvents(url, onEvent, signal) {
  const response = await fetch(url, { headers: { Accept: 'text/event-stream' }, signal });
  if (!response.ok) throw await readError(response);
  await readEvents(response, onEvent);
}

async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of block.split('\n')) {
        if (line.startsWith('data:')) onEvent(JSON.parse(line.slice(5)));
      }
    }
  }
}

/**
 * Stream one chat completion. Never throws: the returned object always describes what happened.
 * onUpdate(result) fires as text arrives so callers can render progressively.
 */
export async function runCompletion({ target, messages, system = '', temperature, maxTokens, signal, onUpdate }) {
  const result = {
    content: '', reasoning: '', metrics: null, finishReason: null, truncated: false,
    error: null, aborted: false, done: false, thinkingMs: null, provider: target.provider, model: target.model,
  };
  const started = performance.now();
  let firstThought = null;
  try {
    await streamSSE('/api/chat/stream', {
      target, messages, system, temperature, max_tokens: maxTokens,
    }, (event) => {
      if (event.type === 'start') {
        result.provider = event.provider;
        result.model = event.model;
      } else if (event.type === 'reasoning') {
        firstThought ??= performance.now();
        result.reasoning += event.text;
      } else if (event.type === 'content') {
        if (firstThought != null && result.thinkingMs == null) result.thinkingMs = performance.now() - firstThought;
        result.content += event.text;
      } else if (event.type === 'reclassify') {
        result.reasoning += result.content;
        result.content = '';
      } else if (event.type === 'done') {
        result.metrics = event.metrics;
        result.finishReason = event.finish_reason;
        result.truncated = event.truncated;
      } else if (event.type === 'error') {
        result.error = event;
      }
      onUpdate?.(result);
    }, signal);
  } catch (error) {
    if (error.name === 'AbortError') result.aborted = true;
    else result.error = { message: error.message || 'The request failed.', code: 'network', status: error.status ?? null };
  }
  if (firstThought != null && result.thinkingMs == null) result.thinkingMs = performance.now() - firstThought;
  result.elapsed = (performance.now() - started) / 1000;
  result.done = true;
  onUpdate?.(result);
  return result;
}
