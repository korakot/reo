// reo live: microphone -> AudioWorklet (live-worklet.js) -> worker (live-worker.js, runs reo) -> dial and charts.
const $ = (id) => document.getElementById(id);
const L = { ctx: null, stream: null, node: null, worker: null, last: null, shown: 0, hist: [], thr: 6.0, running: false };
const MAX = 10; // dial range, syllables per second
try { const t = parseFloat(localStorage.getItem('reo-thr')); if (t >= 3 && t <= 9) L.thr = t; } catch {}
$('thr').value = L.thr; $('thrv').textContent = L.thr.toFixed(1) + ' /s';
$('thr').addEventListener('input', () => {
  L.thr = parseFloat($('thr').value); $('thrv').textContent = L.thr.toFixed(1) + ' /s';
  try { localStorage.setItem('reo-thr', L.thr); } catch {}
});
window.__reoLive = () => L.last && { now: L.last.now, rate: L.last.rate, speaking: L.last.speaking, session: L.last.session, thr: L.thr };

$('go').addEventListener('click', () => (L.running ? stop() : start()));

async function start() {
  $('msg').textContent = '';
  try {
    L.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
  } catch (err) { $('msg').textContent = 'Microphone not available: ' + err.message; return; }
  L.ctx = new AudioContext();
  await L.ctx.audioWorklet.addModule('live-worklet.js');
  const src = L.ctx.createMediaStreamSource(L.stream);
  L.node = new AudioWorkletNode(L.ctx, 'reo-capture');
  const mute = L.ctx.createGain(); mute.gain.value = 0;
  src.connect(L.node); L.node.connect(mute); mute.connect(L.ctx.destination); // keeps the node running, silently
  L.worker = new Worker('live-worker.js', { type: 'module' });
  L.worker.postMessage({ type: 'init', sr: L.ctx.sampleRate });
  L.node.port.onmessage = (e) => L.worker.postMessage({ type: 'audio', x: e.data }, [e.data.buffer]);
  L.worker.onmessage = (e) => receive(e.data);
  await L.ctx.resume();
  L.running = true; L.hist = []; L.last = null;
  $('go').textContent = '■ Stop'; $('go').className = 'stop';
  $('state').textContent = 'Listening… say a few sentences.';
}

function stop() {
  L.running = false;
  try { L.stream.getTracks().forEach((t) => t.stop()); } catch {}
  try { L.ctx.close(); } catch {}
  if (L.worker) L.worker.terminate();
  $('go').textContent = '● Start'; $('go').className = 'go';
  $('state').textContent = 'Stopped. Press Start to go again.';
}

function receive(m) {
  L.last = m;
  L.hist.push({ t: m.now, rate: m.rate, speaking: m.speaking });
  while (L.hist.length && L.hist[0].t < m.now - 60) L.hist.shift();
  const s = m.session;
  $('sRate').textContent = s.rate ? s.rate.toFixed(2) : '–';
  $('sSyl').textContent = s.syllables;
  $('sTime').textContent = s.speech < 60 ? Math.round(s.speech) + ' s' : `${Math.floor(s.speech / 60)}:${String(Math.round(s.speech % 60)).padStart(2, '0')}`;
  if (m.rate === null) $('state').textContent = 'Listening… say a few sentences.';
  else if (!m.speaking) $('state').textContent = 'Pause: holding your last speed.';
  else $('state').textContent = m.rate > L.thr ? 'Too fast: slow down a little.' : m.rate > L.thr - 1 ? 'Getting fast.' : 'Good pace.';
  drawStrip(); drawHistory();
}

// ------------------------------------------------------------------ drawing

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function setup(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return { g, w, h };
}
const zone = (v) => (v > L.thr ? css('--warn') : v > L.thr - 1 ? css('--near') : css('--ok'));

function drawGauge() {
  const { g, w, h } = setup($('gauge'));
  // radius chosen so the number and its label fit under the pivot: cy + 0.34 R + 18 <= h - 4
  const cx = w / 2, R = Math.min(w / 2 - 16, (h - 38) / 1.34), cy = R + 16, th = Math.max(14, R * 0.12);
  const ang = (v) => Math.PI + (Math.min(MAX, Math.max(0, v)) / MAX) * Math.PI;
  // coloured zones: fine, getting close, too fast
  const arc = (a, b, col) => { g.beginPath(); g.arc(cx, cy, R - th / 2, ang(a), ang(b)); g.strokeStyle = col; g.lineWidth = th; g.stroke(); };
  arc(0, Math.max(0, L.thr - 1), css('--ok')); arc(Math.max(0, L.thr - 1), L.thr, css('--near')); arc(L.thr, MAX, css('--warn'));
  // ticks and labels
  g.strokeStyle = css('--muted'); g.fillStyle = css('--muted'); g.lineWidth = 1.5;
  g.font = '13px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  for (let v = 0; v <= MAX; v++) {
    const a = ang(v), r1 = R - th - 4, r2 = r1 - (v % 2 ? 6 : 11);
    g.beginPath(); g.moveTo(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a)); g.lineTo(cx + r2 * Math.cos(a), cy + r2 * Math.sin(a)); g.stroke();
    if (v % 2 === 0) g.fillText(v, cx + (r2 - 12) * Math.cos(a), cy + (r2 - 12) * Math.sin(a));
  }
  // needle
  const live = L.last && L.last.rate !== null, v = L.shown, a = ang(v);
  g.globalAlpha = live && L.last.speaking ? 1 : 0.45;
  g.strokeStyle = css('--needle'); g.lineWidth = 4; g.lineCap = 'round';
  g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + (R - th - 18) * Math.cos(a), cy + (R - th - 18) * Math.sin(a)); g.stroke();
  g.fillStyle = css('--needle'); g.beginPath(); g.arc(cx, cy, 8, 0, 2 * Math.PI); g.fill();
  g.globalAlpha = 1;
  // the number
  g.textBaseline = 'alphabetic';
  g.fillStyle = live ? zone(v) : css('--muted');
  g.font = `700 ${Math.round(R * 0.30)}px system-ui, sans-serif`;
  g.fillText(live ? v.toFixed(1) : '–', cx, cy + R * 0.34);
  g.fillStyle = css('--muted'); g.font = '13px system-ui, sans-serif';
  g.fillText('syllables per second', cx, cy + R * 0.34 + 18);
}

function drawStrip() {
  const m = L.last; if (!m) return;
  const { g, w, h } = setup($('strip'));
  const t1 = m.now, t0 = t1 - 8, X = (t) => ((t - t0) / 8) * w;
  let lo = Infinity, hi = -Infinity;
  for (const v of m.env) { if (v < lo) lo = v; if (v > hi) hi = v; }
  lo = Math.max(lo, hi - 40);
  const Y = (v) => h - 6 - ((Math.max(v, lo) - lo) / (hi - lo || 1)) * (h - 14);
  g.fillStyle = css('--accent-soft');
  for (const [a, b] of m.stretches) g.fillRect(X(Math.max(a, t0)), 2, X(Math.min(b, t1)) - X(Math.max(a, t0)), h - 4);
  g.strokeStyle = css('--env'); g.lineWidth = 1.3; g.beginPath();
  m.env.forEach((v, i) => { const t = m.envStart + i * 0.01; if (t < t0) return; i ? g.lineTo(X(t), Y(v)) : g.moveTo(X(t), Y(v)); });
  g.stroke();
  g.fillStyle = m.rate !== null && m.rate > L.thr ? css('--warn') : css('--accent');
  for (const t of m.peaks) {
    const i = Math.round((t - m.envStart) / 0.01);
    if (i >= 0 && i < m.env.length) { g.beginPath(); g.arc(X(t), Y(m.env[i]), 4, 0, 2 * Math.PI); g.fill(); }
  }
}

function drawHistory() {
  const m = L.last; if (!m) return;
  const { g, w, h } = setup($('history'));
  const l = 28, t1 = m.now, t0 = t1 - 60, X = (t) => l + ((t - t0) / 60) * (w - l - 6), Y = (v) => h - 18 - (Math.min(v, MAX) / MAX) * (h - 26);
  g.font = '12px system-ui, sans-serif'; g.fillStyle = css('--muted'); g.strokeStyle = css('--line'); g.lineWidth = 1;
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (let v = 0; v <= MAX; v += 2) { const y = Math.round(Y(v)) + 0.5; g.beginPath(); g.moveTo(l, y); g.lineTo(w - 6, y); g.stroke(); g.fillText(v, l - 6, y); }
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (let s = 0; s <= 60; s += 15) g.fillText(s ? `−${s} s` : 'now', X(t1 - s), h - 14);
  g.strokeStyle = css('--warn'); g.setLineDash([6, 4]); g.beginPath(); g.moveTo(l, Y(L.thr)); g.lineTo(w - 6, Y(L.thr)); g.stroke(); g.setLineDash([]);
  g.lineWidth = 3; g.lineCap = 'round';
  for (let i = 1; i < L.hist.length; i++) {
    const a = L.hist[i - 1], b = L.hist[i];
    if (a.rate === null || b.rate === null) continue;
    g.globalAlpha = b.speaking ? 1 : 0.35;
    g.strokeStyle = zone(b.rate);
    g.beginPath(); g.moveTo(X(a.t), Y(a.rate)); g.lineTo(X(b.t), Y(b.rate)); g.stroke();
  }
  g.globalAlpha = 1;
}

function frame() {
  const target = L.last && L.last.rate !== null ? L.last.rate : 0;
  L.shown += (target - L.shown) * 0.12;
  drawGauge();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
addEventListener('resize', () => { drawStrip(); drawHistory(); });
