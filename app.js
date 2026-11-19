// reo demo page: open or record audio, analyse each channel in a worker, draw rate per utterance.
import { readWav, fastShare, HOP } from './reo.js';

const $ = (id) => document.getElementById(id);
const S = { audio: null, results: [], ch: 0, sel: null, thr: 6.0, name: '', job: 0, playing: null };
try { const t = parseFloat(localStorage.getItem('reo-thr')); if (t >= 3 && t <= 9) S.thr = t; } catch {}
$('thr').value = S.thr;

// ------------------------------------------------------------------ input

$('file').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) load(f, f.name); e.target.value = ''; });
const drop = $('drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); const f = e.dataTransfer.files[0]; if (f) load(f, f.name); });

let rec = null;
$('rec').addEventListener('click', async () => {
  if (rec) { rec.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (err) { status('Microphone not available: ' + err.message); return; }
  const chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const t0 = Date.now();
  const tick = setInterval(() => { $('rec').textContent = `■ Stop  ${fmtTime((Date.now() - t0) / 1000)}`; }, 250);
  rec.onstop = () => {
    clearInterval(tick); stream.getTracks().forEach((t) => t.stop());
    $('rec').classList.remove('on'); $('rec').textContent = '● Record'; rec = null;
    load(new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' }), 'recording');
  };
  rec.start(500);
  $('rec').classList.add('on'); $('rec').textContent = '■ Stop  0:00';
  status('Recording… speak for at least a few seconds, then press Stop.');
});

async function decode(buf) {
  try { return readWav(buf); } catch {}
  // not a WAV reo reads itself: let the browser decode it, at 16 kHz (plenty for 300-3300 Hz)
  const ctx = new OfflineAudioContext(1, 1, 16000);
  const ab = await ctx.decodeAudioData(buf.slice(0));
  return { channels: Array.from({ length: ab.numberOfChannels }, (_, c) => ab.getChannelData(c)), sr: ab.sampleRate };
}

let worker = null;
async function load(blob, name) {
  const job = ++S.job;
  status('Reading ' + name + '…');
  let audio;
  try { audio = await decode(await blob.arrayBuffer()); } catch (err) { status('Could not read this file: ' + (err.message || err)); return; }
  if (job !== S.job) return;
  const dur = audio.channels[0].length / audio.sr;
  if (dur < 1) { status('This recording is shorter than a second.'); return; }
  S.audio = audio; S.name = name; S.results = []; S.ch = 0; S.sel = null;
  $('results').hidden = true;
  if (worker) worker.terminate();
  worker = new Worker('worker.js', { type: 'module' });
  const n = audio.channels.length, prog = new Array(n).fill(0);
  let done = 0;
  const t0 = performance.now();
  progress(0, n);
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.id !== job) return;
    if (m.error) { status('Analysis failed: ' + m.error); return; }
    if (m.progress !== undefined) { prog[m.channel] = m.progress; progress(prog.reduce((a, b) => a + b, 0) / n, n); return; }
    S.results[m.channel] = m.result; prog[m.channel] = 1;
    if (++done === n) {
      status(`${name}: ${fmtTime(dur)}, ${n === 1 ? 'mono' : n + ' channels'}, ${audio.sr} Hz · analysed in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
      show();
    }
  };
  // channels one after another in one worker (copies, so the page keeps its audio for playback)
  audio.channels.forEach((x, c) => worker.postMessage({ id: job, channel: c, x: new Float32Array(x), sr: audio.sr }));
}

function status(t) { $('status').textContent = t; }
function progress(f, n) {
  $('status').innerHTML = `Counting syllables${n > 1 ? ` in ${n} channels` : ''}… <span class="bar"><i style="width:${Math.round(f * 100)}%"></i></span>`;
}

// ------------------------------------------------------------------ results

function show() {
  $('results').hidden = false;
  const tabs = $('tabs');
  tabs.innerHTML = '';
  if (S.results.length > 1) {
    const names = S.results.length === 2 ? ['Left channel', 'Right channel'] : S.results.map((_, c) => `Channel ${c + 1}`);
    S.results.forEach((r, c) => {
      const b = document.createElement('button');
      b.role = 'tab'; b.textContent = `${names[c]} · ${r.rate ? r.rate.toFixed(1) : '–'} /s`;
      b.setAttribute('aria-selected', c === S.ch);
      b.onclick = () => { S.ch = c; S.sel = null; show(); };
      tabs.appendChild(b);
    });
  }
  const r = S.results[S.ch];
  const stat = (v, label, cls = '') => `<div class="stat ${cls}"><b>${v}</b><span>${label}</span></div>`;
  $('stats').innerHTML =
    stat(r.rate ? r.rate.toFixed(2) : '–', 'syllables per second of speech', 'big') +
    stat(r.median_interval ? Math.round(r.median_interval * 1000) + ' ms' : '–', 'median time from one syllable to the next') +
    stat(r.syllables, 'syllables counted') +
    stat(fmtTime(r.speech_time), `speaking time (of ${fmtTime(r.duration)})`) +
    stat(r.pause_share !== null ? Math.round(r.pause_share * 100) + '%' : '–', 'of the talk is pauses');
  $('fileinfo').textContent = `${S.name}${S.results.length > 1 ? ' · ' + (S.ch === 0 ? 'left' : S.ch === 1 ? 'right' : 'channel ' + (S.ch + 1)) : ''}`;
  if (S.sel === null) {
    // start on the fastest rated utterance, so there is something to hear
    let best = -1, bi = null;
    r.utterances.forEach((u, i) => { if (u.rate !== null && u.rate > best) { best = u.rate; bi = i; } });
    S.sel = bi;
  }
  updateThreshold();
  drawDetail();
}

$('thr').addEventListener('input', () => {
  S.thr = parseFloat($('thr').value);
  try { localStorage.setItem('reo-thr', S.thr); } catch {}
  updateThreshold();
});

function updateThreshold() {
  $('thrv').textContent = S.thr.toFixed(1) + ' /s';
  const r = S.results[S.ch];
  if (!r) return;
  const fs = fastShare(r, S.thr);
  const nFast = r.utterances.filter((u) => u.rate !== null && u.rate > S.thr).length;
  $('fast').textContent = fs === null ? '' : `${Math.round(fs * 100)}% of speech above the line (${nFast} utterance${nFast === 1 ? '' : 's'})`;
  $('fast').style.color = fs > 0 ? 'var(--warn)' : 'var(--muted)';
  drawTimeline();
}

// ------------------------------------------------------------------ drawing

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function setup(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.font = '12px system-ui, sans-serif';
  return { g, w, h };
}

const TL = { l: 34, r: 8, t: 10, b: 22 };
function tlScale() {
  const r = S.results[S.ch], cv = $('timeline');
  const w = cv.clientWidth, h = cv.clientHeight;
  const ymax = Math.max(9, ...r.utterances.map((u) => (u.rate || 0) + 0.5));
  return {
    x: (t) => TL.l + (t / r.duration) * (w - TL.l - TL.r), t: (x) => ((x - TL.l) / (w - TL.l - TL.r)) * r.duration,
    y: (v) => h - TL.b - (v / ymax) * (h - TL.t - TL.b), ymax, w, h,
  };
}

function drawTimeline() {
  const r = S.results[S.ch];
  if (!r) return;
  const { g, w, h } = setup($('timeline'));
  const s = tlScale();
  const ink = css('--muted'), line = css('--line');
  // grid
  g.strokeStyle = line; g.fillStyle = ink; g.lineWidth = 1; g.textAlign = 'right'; g.textBaseline = 'middle';
  for (let v = 0; v <= s.ymax; v += 2) {
    const y = Math.round(s.y(v)) + 0.5;
    g.beginPath(); g.moveTo(TL.l, y); g.lineTo(w - TL.r, y); g.stroke();
    g.fillText(v, TL.l - 6, y);
  }
  g.textAlign = 'center'; g.textBaseline = 'top';
  const step = niceStep(r.duration / 6);
  for (let t = 0; t <= r.duration; t += step) g.fillText(fmtTime(t), s.x(t), h - TL.b + 6);
  // bars
  r.utterances.forEach((u, i) => {
    const x0 = s.x(u.start), x1 = Math.max(s.x(u.end), x0 + 2);
    if (u.rate === null) {
      g.fillStyle = css('--short'); g.fillRect(x0, s.y(0) - 6, x1 - x0, 6);
    } else {
      const fast = u.rate > S.thr;
      g.fillStyle = fast ? css('--warn') : css('--accent');
      g.globalAlpha = i === S.sel ? 1 : 0.75;
      g.fillRect(x0, s.y(u.rate), x1 - x0, s.y(0) - s.y(u.rate));
      g.globalAlpha = 1;
    }
    if (i === S.sel) { g.strokeStyle = css('--ink'); g.lineWidth = 2; g.strokeRect(x0 - 1, s.y(u.rate || 0.6) - 1, x1 - x0 + 2, s.y(0) - s.y(u.rate || 0.6) + 2); }
  });
  // overall rate and threshold
  if (r.rate) {
    g.strokeStyle = ink; g.setLineDash([2, 3]); g.lineWidth = 1;
    const y = s.y(r.rate); g.beginPath(); g.moveTo(TL.l, y); g.lineTo(w - TL.r, y); g.stroke();
    g.setLineDash([]); g.fillStyle = ink; g.textAlign = 'left'; g.textBaseline = 'bottom';
    g.fillText('average ' + r.rate.toFixed(1), TL.l + 4, y - 2);
  }
  g.strokeStyle = css('--warn'); g.setLineDash([6, 4]); g.lineWidth = 1.5;
  const yt = s.y(S.thr); g.beginPath(); g.moveTo(TL.l, yt); g.lineTo(w - TL.r, yt); g.stroke();
  g.setLineDash([]);
  g.fillStyle = css('--warn'); g.textAlign = 'right'; g.textBaseline = 'bottom';
  g.fillText('too fast', w - TL.r - 2, yt - 2);
}

function niceStep(x) {
  const c = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];
  return c.find((v) => v >= x) || 3600;
}

$('timeline').addEventListener('click', (e) => {
  const r = S.results[S.ch];
  if (!r) return;
  const s = tlScale(), rect = e.currentTarget.getBoundingClientRect();
  const t = s.t(e.clientX - rect.left), tol = s.t(TL.l + 6) - s.t(TL.l);
  let best = null, bd = Infinity;
  r.utterances.forEach((u, i) => {
    const d = t < u.start ? u.start - t : t > u.end ? t - u.end : 0;
    if (d < bd) { bd = d; best = i; }
  });
  if (best !== null && bd <= tol) { S.sel = best; drawTimeline(); drawDetail(); play(); }
});

function drawDetail() {
  const r = S.results[S.ch];
  const u = r && S.sel !== null ? r.utterances[S.sel] : null;
  $('detailCard').hidden = !u;
  if (!u) return;
  const n = r.peaks.filter((t) => t >= u.start && t < u.end).length;
  $('detailTitle').textContent = `${fmtSpan(u.start, u.end)} · ${n} syllables in ${u.speech.toFixed(1)} s of speech` +
    (u.rate !== null ? ` = ${u.rate.toFixed(1)} /s` : ' (too short for a rate)');
  const { g, w, h } = setup($('detail'));
  const a = Math.max(0, u.start - 0.3), b = Math.min(r.duration, u.end + 0.3);
  const X = (t) => 8 + ((t - a) / (b - a)) * (w - 16);
  const i0 = Math.floor(a / HOP), i1 = Math.min(r.env.length - 1, Math.ceil(b / HOP));
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i <= i1; i++) { lo = Math.min(lo, r.env[i]); hi = Math.max(hi, r.env[i]); }
  lo = Math.max(lo, hi - 45);
  const Y = (v) => h - 20 - ((Math.max(v, lo) - lo) / (hi - lo || 1)) * (h - 34);
  // speech stretches
  g.fillStyle = css('--accent-soft');
  for (const [sa, sb] of r.stretches) if (sb > a && sa < b) g.fillRect(X(Math.max(sa, a)), 4, X(Math.min(sb, b)) - X(Math.max(sa, a)), h - 24);
  // loudness
  g.strokeStyle = css('--env'); g.lineWidth = 1.5; g.beginPath();
  for (let i = i0; i <= i1; i++) { const x = X(i * HOP), y = Y(r.env[i]); i === i0 ? g.moveTo(x, y) : g.lineTo(x, y); }
  g.stroke();
  // syllables
  const fast = u.rate !== null && u.rate > S.thr;
  g.fillStyle = fast ? css('--warn') : css('--accent');
  for (const t of r.peaks) {
    if (t < a || t > b) continue;
    const x = X(t), y = Y(r.env[Math.round(t / HOP)]);
    g.beginPath(); g.arc(x, y, 4, 0, 2 * Math.PI); g.fill();
  }
  // time axis
  g.fillStyle = css('--muted'); g.textAlign = 'center'; g.textBaseline = 'top';
  const step = niceStep((b - a) / 5) <= 1 ? 0.5 : niceStep((b - a) / 5);
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) g.fillText(step < 1 ? t.toFixed(1) + ' s' : fmtTime(t), X(t), h - 16);
  if (S.playing) {
    const t = S.playing.from + (S.playing.ctx.currentTime - S.playing.t0);
    if (t <= S.playing.to) { g.strokeStyle = css('--ink'); g.lineWidth = 1; g.beginPath(); g.moveTo(X(t), 4); g.lineTo(X(t), h - 20); g.stroke(); }
  }
}

// ------------------------------------------------------------------ playback

let actx = null;
function play() {
  const r = S.results[S.ch];
  if (!r || S.sel === null) return;
  const u = r.utterances[S.sel];
  if (S.playing) { try { S.playing.src.stop(); } catch {} S.playing = null; }
  actx = actx || new AudioContext();
  actx.resume();
  const x = S.audio.channels[S.ch], sr = S.audio.sr;
  const from = Math.max(0, u.start - 0.15), to = Math.min(r.duration, u.end + 0.15);
  const seg = x.subarray(Math.floor(from * sr), Math.ceil(to * sr));
  const buf = actx.createBuffer(1, seg.length, sr);
  buf.copyToChannel(new Float32Array(seg), 0);
  const src = actx.createBufferSource();
  src.buffer = buf; src.connect(actx.destination); src.start();
  S.playing = { src, ctx: actx, t0: actx.currentTime, from, to };
  src.onended = () => { if (S.playing && S.playing.src === src) { S.playing = null; drawDetail(); } };
  const anim = () => { if (S.playing && S.playing.src === src) { drawDetail(); requestAnimationFrame(anim); } };
  requestAnimationFrame(anim);
}
$('play').addEventListener('click', play);

// ------------------------------------------------------------------ CSV

$('csv').addEventListener('click', () => {
  const rows = [['channel', 'start_s', 'end_s', 'speech_s', 'syllables', 'rate_per_s', 'above_line']];
  S.results.forEach((r, c) => r.utterances.forEach((u) => rows.push([c, u.start.toFixed(2), u.end.toFixed(2), u.speech.toFixed(2), u.syllables,
    u.rate === null ? '' : u.rate.toFixed(2), u.rate === null ? '' : u.rate > S.thr ? 1 : 0])));
  const blob = new Blob([rows.map((r) => r.join(',')).join('\n') + '\n'], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = (S.name.replace(/\.[^.]+$/, '') || 'reo') + '-utterances.csv';
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

function fmtSpan(a, b) {
  return b < 60 ? `${a.toFixed(1)}–${b.toFixed(1)} s` : `${fmtTime(a)}–${fmtTime(b)}`;
}

function fmtTime(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return m ? `${m}:${String(Math.floor(s)).padStart(2, '0')}` : `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
}

addEventListener('resize', () => { if (S.results.length) { drawTimeline(); drawDetail(); } });
