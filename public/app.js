const $ = (id) => document.getElementById(id);
let controller = null;
let timer = null;
let savedResult = null;
let traces = [];
let ready = false;

const labels = {
  starting: 'Preparing your travel team…',
  search_agent_node: 'Comparing flights and places to stay…',
  search_tool_node: 'Flight and hotel search results received.',
  travel_scout_agent_node: 'Researching the places worth your time…',
  travel_scout_tool_node: 'Local research received. Connecting the details…',
  synthesizer_node: 'Bringing your journey together…',
};

function setActivity(text, kind = '') {
  $('activity-label').textContent = text;
  $('activity').className = 'activity ' + kind;
  $('spinner').hidden = kind !== '';
}

function setTab(which) {
  for (const name of ['plan', 'trace']) {
    const active = name === which;
    $(name + '-tab').setAttribute('aria-selected', String(active));
    $(name + '-view').hidden = !active;
  }
}

// Render only text and a small allowlist of Markdown elements. Model HTML is never executed.
function inline(parent, text) {
  const pattern = /\*\*([^*]+)\*\*|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`/g;
  let offset = 0;
  for (const match of text.matchAll(pattern)) {
    parent.append(document.createTextNode(text.slice(offset, match.index)));
    if (match[1]) {
      const strong = document.createElement('strong'); strong.textContent = match[1]; parent.append(strong);
    } else if (match[2]) {
      const link = document.createElement('a'); link.textContent = match[2]; link.href = match[3];
      link.target = '_blank'; link.rel = 'noopener noreferrer'; parent.append(link);
    } else {
      const code = document.createElement('code'); code.textContent = match[4]; parent.append(code);
    }
    offset = match.index + match[0].length;
  }
  parent.append(document.createTextNode(text.slice(offset)));
}

function renderAnswer(markdown) {
  const target = $('answer'); target.replaceChildren();
  let list = null; let pre = null;
  for (const line of markdown.split('\n')) {
    if (line.trim().startsWith('```')) {
      if (pre) pre = null;
      else { pre = document.createElement('pre'); target.append(pre); }
      continue;
    }
    if (pre) { pre.textContent += line + '\n'; continue; }
    if (!line.trim()) { list = null; continue; }
    if (/^\s*---+\s*$/.test(line)) { target.append(document.createElement('hr')); list = null; continue; }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    const bullet = line.match(/^\s*(?:([-*])|\d+\.)\s+(.+)$/);
    if (heading) {
      const el = document.createElement('h' + heading[1].length); inline(el, heading[2]); target.append(el); list = null;
    } else if (bullet) {
      const tag = bullet[1] ? 'UL' : 'OL';
      if (!list || list.tagName !== tag) { list = document.createElement(tag.toLowerCase()); target.append(list); }
      const item = document.createElement('li'); inline(item, bullet[2]); list.append(item);
    } else {
      const p = document.createElement('p'); inline(p, line); target.append(p); list = null;
    }
  }
}

function addTrace(message) {
  traces.push(message);
  $('trace-count').textContent = String(traces.length);
  const details = document.createElement('details'); details.className = 'trace-entry';
  const agent = document.createElement('span'); agent.className = 'trace-agent';
  agent.textContent = message.agent === 'search_agent' ? 'FLIGHTS & STAYS' : 'LOCAL DISCOVERY';
  const summary = document.createElement('summary');
  const calls = message.tool_calls || [];
  summary.textContent = `${traces.length}. ${message.type}` + (message.name ? ` · ${message.name}` : '') +
    (calls.length ? ' → ' + calls.map((c) => c.name).join(', ') : '');
  details.append(summary); summary.prepend(agent);
  if (message.tool_call_id) {
    const id = document.createElement('p'); id.className = 'call-id'; id.textContent = 'Call ID: ' + message.tool_call_id; details.append(id);
  }
  if (message.content) {
    const pre = document.createElement('pre');
    try { pre.textContent = JSON.stringify(JSON.parse(message.content), null, 2); }
    catch { pre.textContent = message.content; }
    if (message.truncated) pre.textContent += '\n[Observation truncated for display]';
    details.append(pre);
  }
  if (calls.length) {
    const pre = document.createElement('pre'); pre.textContent = JSON.stringify(calls, null, 2); details.append(pre);
  }
  $('trace-list').append(details);
}

function showRouting(tasks) {
  $('agent-pills').replaceChildren();
  for (const task of tasks) {
    const pill = document.createElement('span'); pill.className = 'agent-pill';
    pill.textContent = task.target_agent === 'search_agent' ? '✈ Flights & stays' : '⌖ Local discovery';
    pill.title = task.focus; $('agent-pills').append(pill);
  }
}

function resetView(sample = false) {
  $('results').hidden = false;
  savedResult = null; traces = []; $('trace-list').replaceChildren(); $('trace-count').textContent = '0';
  $('agent-pills').replaceChildren(); $('answer').replaceChildren(); $('empty-state').hidden = true;
  $('run-state').hidden = false; $('download-result').hidden = true; $('sample-banner').hidden = !sample;
  $('plan-loading').hidden = false; $('form-error').hidden = true; setTab('plan');
  $('results').focus({preventScroll: true});
  $('results').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function finish(data) {
  savedResult = {query: $('trip-query').value, ...data, trace: traces};
  showRouting(data.tasks || []); renderAnswer(data.answer);
  $('plan-loading').hidden = true; $('download-result').hidden = false;
  setActivity($('sample-banner').hidden ? 'Your plan is ready.' : 'Example plan', 'complete');
}

function eventReceived(item) {
  if (item.event === 'progress') setActivity(labels[item.data.node] || 'Your travel team is working…');
  if (item.event === 'routing') showRouting(item.data.tasks);
  if (item.event === 'trace') addTrace(item.data);
  if (item.event === 'result') finish(item.data);
  if (item.event === 'error') throw new Error(item.data.message);
}

async function createPlan(event) {
  event.preventDefault();
  if (controller || !ready) return;
  const query = $('trip-query').value.trim();
  if (query.length < 5) { $('form-error').textContent = 'Tell us a little more about the trip you have in mind.'; $('form-error').hidden = false; return; }
  resetView(); controller = new AbortController(); $('submit-plan').disabled = true; $('cancel-plan').hidden = false;
  $('preview-sample').disabled = true; $('elapsed').textContent = '0s';
  const start = Date.now(); timer = setInterval(() => { $('elapsed').textContent = Math.floor((Date.now() - start) / 1000) + 's'; }, 1000);
  setActivity(labels.starting);
  try {
    const headers = {'Content-Type': 'application/json'};
    if ($('access-key').value) headers.Authorization = 'Bearer ' + $('access-key').value;
    const response = await fetch('/api/plan', {method: 'POST', headers, body: JSON.stringify({query}), signal: controller.signal});
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : 'The request could not be started. Please check your trip details.');
    }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const {done, value} = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      let newline;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1);
        if (line.trim()) eventReceived(JSON.parse(line));
      }
      if (done) { if (buffer.trim()) eventReceived(JSON.parse(buffer)); break; }
    }
    if (!savedResult) throw new Error('The connection ended before your plan was ready. Please try again.');
  } catch (error) {
    setActivity(error.name === 'AbortError' ? 'Planning stopped. You can start a new request.' : error.message, 'error');
    $('plan-loading').hidden = true;
    if (error.name !== 'AbortError') { $('form-error').textContent = error.message; $('form-error').hidden = false; }
  } finally {
    clearInterval(timer); controller = null; $('submit-plan').disabled = !ready; $('cancel-plan').hidden = true; $('preview-sample').disabled = false;
  }
}

$('plan-form').addEventListener('submit', createPlan);
$('cancel-plan').addEventListener('click', () => controller?.abort());
$('trip-query').addEventListener('input', () => { $('char-count').textContent = $('trip-query').value.length + ' / 6000'; });
for (const name of ['plan', 'trace']) $(name + '-tab').addEventListener('click', () => setTab(name));
document.querySelector('.tabs').addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { const next = $('plan-tab').getAttribute('aria-selected') === 'true' ? 'trace' : 'plan'; setTab(next); $(next + '-tab').focus(); }
});
for (const chip of document.querySelectorAll('[data-starter]')) chip.addEventListener('click', () => {
  const departure = new Date(); departure.setDate(departure.getDate() + 45);
  const arrival = new Date(departure); arrival.setDate(arrival.getDate() + 1);
  const returning = new Date(departure); returning.setDate(returning.getDate() + 5);
  const fmt = (d) => d.toISOString().slice(0, 10);
  const starters = {
    city: 'Create a relaxed three-day Lisbon itinerary for two adults. We love walkable neighborhoods, architecture, and local food. Ask me for any essential missing details.',
    flights: `Find round-trip economy flights from YYZ to CDG for two adults, departing ${fmt(departure)} and returning ${fmt(returning)}, with at most one stop. Find one hotel room for ${fmt(arrival)}–${fmt(returning)}, no children, sorted by price. Flag any mismatch with actual arrival dates. Show USD quotes.`,
    food: 'Help two adults explore Kyoto through its food and culture over four days. Include a relaxed itinerary, local restaurants, and seasonal considerations. Ask about dates before researching time-sensitive details.',
  };
  $('trip-query').value = starters[chip.dataset.starter]; $('trip-query').dispatchEvent(new Event('input')); $('trip-query').focus();
});
$('preview-sample').addEventListener('click', async () => {
  if (controller) return;
  try {
    const response = await fetch('/sample.json'); if (!response.ok) throw new Error('Example unavailable');
    const sample = await response.json(); resetView(true); $('trip-query').value = sample.query; $('trip-query').dispatchEvent(new Event('input'));
    for (const message of sample.trace) addTrace(message);
    finish(sample); $('elapsed').textContent = '';
  } catch { $('form-error').textContent = 'The example could not be loaded. Please refresh and try again.'; $('form-error').hidden = false; }
});
$('download-result').addEventListener('click', () => {
  if (!savedResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(savedResult, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = 'travel-plan.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function checkStatus() {
  try {
    const response = await fetch('/api/status'); if (!response.ok) throw new Error('Unavailable');
    const status = await response.json(); ready = status.ready; $('submit-plan').disabled = !ready;
    $('connection-label').textContent = ready ? 'Your travel team is ready' : 'Explore an example plan';
    $('connection-dot').classList.toggle('ready', ready); $('access-field').hidden = !status.access_required;
    if (!ready) { $('setup-note').textContent = 'Live planning is unavailable. You can view an example plan below.'; $('setup-note').hidden = false; }
  } catch {
    ready = false; $('submit-plan').disabled = true; $('connection-label').textContent = 'Live planner unavailable';
    $('setup-note').textContent = 'The live planner is unavailable. Try the example below while we reconnect.'; $('setup-note').hidden = false;
  }
}
checkStatus();
