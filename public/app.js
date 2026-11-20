const $ = (id) => document.getElementById(id);
let controller = null;
let timer = null;
let savedResult = null;
let ready = false;
let activeQuery = "";

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

function resetView(sample = false) {
  $('results').hidden = false;
  savedResult = null; $('answer').replaceChildren(); $('empty-state').hidden = true;
  $('run-state').hidden = false; $('download-result').hidden = true; $('sample-banner').hidden = !sample;
  $('plan-loading').hidden = false; $('form-error').hidden = true;
  $('results').focus({preventScroll: true});
  $('results').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function finish(data, remember = true) {
  savedResult = {answer: data.answer, query: activeQuery || $('trip-query').value};
  if (remember && $('sample-banner').hidden) rememberPlan(savedResult);
  renderAnswer(data.answer);
  $('plan-loading').hidden = true; $('download-result').hidden = false;
  setActivity($('sample-banner').hidden ? 'Your plan is ready.' : 'Example plan', 'complete');
}

function eventReceived(item) {
  if (item.event === 'progress') setActivity(labels[item.data.node] || 'Your travel team is working…');
  if (item.event === 'result') finish(item.data);
  if (item.event === 'error') throw new Error(item.data.message);
}

async function createPlan(event) {
  event.preventDefault();
  if (controller || !ready) return;
  const query = buildQuery();
  if (!query) return;
  activeQuery = query;
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
    activeQuery = sample.query; finish(sample); $('elapsed').textContent = '';
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

function buildQuery() {
  const destination = $('destination').value.trim();
  const notes = $('trip-query').value.trim();
  const fail = (message, field) => { $('form-error').textContent = message; $('form-error').hidden = false; $(field).focus(); return null; };
  if (!destination && notes.length < 5) return fail('Choose a destination or tell us a little about the trip you have in mind.', 'destination');
  const start = $('depart-date').value, end = $('return-date').value;
  if (start && end && end < start) return fail('Your return date should be on or after your departure.', 'return-date');
  if ((start && start < localToday()) || (end && end < localToday())) return fail('Choose upcoming dates for your trip.', start && start < localToday() ? 'depart-date' : 'return-date');
  const parts = [];
  if (destination) parts.push('Plan a trip to ' + destination + '.');
  if ($('origin').value.trim()) parts.push('Departing from ' + $('origin').value.trim() + '. Include flight options.');
  if (start) parts.push('Departure date: ' + start + '.');
  if (end) parts.push('Return date: ' + end + '.');
  parts.push('Travelers: ' + $('travelers').value + ' adults.');
  if ($('budget').value.trim()) parts.push('Total trip budget: ' + $('budget').value.trim() + '.');
  const interests = [...document.querySelectorAll('[data-interest][aria-pressed="true"]')].map(el => el.dataset.interest);
  if (interests.length) parts.push('Interests: ' + interests.join(', ') + '.');
  if (notes) parts.push(notes);
  parts.push('Include places to stay and a practical itinerary. Clearly identify estimates and ask for essential missing details. Do not book anything.');
  const query = parts.join('\n');
  if (query.length > 6000) return fail('Please shorten your trip notes a little before planning.', 'trip-query');
  return query;
}
function localToday() {
  const now = new Date();
  return [now.getFullYear(), String(now.getMonth()+1).padStart(2,'0'), String(now.getDate()).padStart(2,'0')].join('-');
}
const draftFields = ['destination','origin','depart-date','return-date','travelers','budget','trip-query'];
let recentPlans = [];
function readLocal(key, fallback) {
  try { const value = localStorage.getItem(key); return value ? JSON.parse(value) : fallback; } catch { return fallback; }
}
function writeLocal(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); return true; }
  catch { $('storage-note').textContent = 'Browser storage is unavailable. Export plans you want to keep.'; $('storage-note').hidden = false; return false; }
}
function captureDraft() {
  return Object.fromEntries(draftFields.map(id => [id, $(id).value]));
}
function persistDraft() {
  writeLocal('travel-planner-draft-v1', captureDraft());
}
function rememberPlan(plan) {
  const title = $('destination').value.trim() || plan.query.split('\n')[0].slice(0,80);
  const item = {title, query: plan.query, answer: plan.answer, tasks: plan.tasks || [], date: new Date().toISOString(), draft: captureDraft()};
  recentPlans = [item, ...recentPlans.filter(p => p.query !== item.query)].slice(0,5);
  writeLocal('travel-planner-plans-v1', recentPlans); renderRecentPlans();
}
function renderRecentPlans() {
  $('saved-count').textContent = String(recentPlans.length);
  if (!recentPlans.length) return;
  $('recent-plans').replaceChildren();
  for (const item of recentPlans) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'recent-card';
    const icon = document.createElement('span'); icon.className = 'recent-icon'; icon.textContent = '⌖'; icon.setAttribute('aria-hidden','true');
    const title = document.createElement('strong'); title.textContent = item.title;
    const date = document.createElement('p'); date.textContent = new Date(item.date).toLocaleDateString(undefined,{month:'short',day:'numeric'}) + ' · Open plan';
    button.append(icon,title,date);
    button.addEventListener('click', () => {
      if (controller) return;
      if (item.draft) for (const id of draftFields) if (typeof item.draft[id] === 'string') $(id).value = item.draft[id];
      activeQuery = item.query; resetView(); finish(item, false); $('elapsed').textContent = ''; persistDraft();
    });
    $('recent-plans').append(button);
  }
}
function updateProfile() {
  const name = $('traveler-name').value.trim();
  $('greeting').textContent = name ? 'WELCOME BACK, ' + name.toUpperCase() : 'A LITTLE INSPIRATION, JUST FOR YOU';
  $('profile-label').textContent = name ? name + '’s travel space' : 'Your travel space';
  $('profile-initial').textContent = name ? Array.from(name)[0].toUpperCase() : 'Y';
}
function initializeWorkspace() {
  const profile = readLocal('travel-planner-profile-v1', {});
  if (typeof profile.name === 'string') $('traveler-name').value = profile.name.slice(0,40);
  updateProfile();
  $('traveler-name').addEventListener('input', () => { updateProfile(); writeLocal('travel-planner-profile-v1',{name:$('traveler-name').value}); });
  const draft = readLocal('travel-planner-draft-v1', {});
  for (const id of draftFields) {
    if (typeof draft[id] === 'string') $(id).value = draft[id];
    $(id).addEventListener('input',persistDraft); $(id).addEventListener('change',persistDraft);
  }
  $('depart-date').min = localToday(); $('return-date').min = localToday();
  $('depart-date').addEventListener('change', () => { $('return-date').min = $('depart-date').value || localToday(); });
  const interests = readLocal('travel-planner-interests-v1', []);
  for (const button of document.querySelectorAll('[data-interest]')) {
    button.setAttribute('aria-pressed', String(Array.isArray(interests) && interests.includes(button.dataset.interest)));
    button.addEventListener('click', () => {
      button.setAttribute('aria-pressed',String(button.getAttribute('aria-pressed') !== 'true'));
      writeLocal('travel-planner-interests-v1',[...document.querySelectorAll('[data-interest][aria-pressed="true"]')].map(el=>el.dataset.interest));
    });
  }
  const stored = readLocal('travel-planner-plans-v1', []);
  recentPlans = Array.isArray(stored) ? stored.filter(p=>p && typeof p.query === 'string' && typeof p.answer === 'string' && typeof p.title === 'string').slice(0,5) : [];
  renderRecentPlans();
  $('new-trip').addEventListener('click', () => {
    if (controller) { $('cancel-plan').focus(); return; }
    for (const id of draftFields) if (!['origin','travelers','budget'].includes(id)) $(id).value = '';
    $('return-date').min = localToday(); $('results').hidden = true; $('form-error').hidden = true; persistDraft();
    $('destination').scrollIntoView({behavior:'smooth',block:'center'}); $('destination').focus({preventScroll:true});
  });
}

initializeWorkspace();
