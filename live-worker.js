// Live mode worker: downsamples the microphone to about 16 kHz, keeps the last 10 s, and four times a second runs
// reo on them. The live rate is the syllables over the most recent RECENT seconds of speech, so it follows the
// speaker within a few seconds and does not drop to zero in a pause. Syllables older than COMMIT seconds are added
// to the session total once, so the total does not count a syllable twice as the window slides.
import { analyse, HOP } from './reo.js';

const KEEP = 10, TICK = 0.25, COMMIT = 4, SHOW = 8;
let RECENT = 4; // seconds of speech behind the live rate (init can change it)
let sr0 = 0, D = 1, sr = 0, fir = null, hist = null, phase = 0;
let ring = null, n = 0, total = 0, sinceTick = 0;     // ring holds the last KEEP s at sr; total = samples ever
let committed = 0, sessionSyl = 0, sessionSpeech = 0;  // session totals, up to time `committed` (s)

function setup(rate) {
  sr0 = rate; D = Math.max(1, Math.floor(rate / 14000)); sr = rate / D;
  // low-pass for the decimation: Hann-windowed sinc, cut-off at 0.45 of the new rate
  const taps = D === 1 ? 1 : 16 * D + 1, fc = 0.45 / D, h = (taps - 1) / 2;
  fir = new Float64Array(taps); let s = 0;
  for (let i = 0; i < taps; i++) {
    const t = i - h, w = taps === 1 ? 1 : 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1));
    fir[i] = (t === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * t) / (Math.PI * t)) * w; s += fir[i];
  }
  for (let i = 0; i < taps; i++) fir[i] /= s;
  hist = new Float64Array(taps); phase = 0;
  ring = new Float32Array(Math.round(KEEP * sr)); n = 0; total = 0; sinceTick = 0;
  committed = 0; sessionSyl = 0; sessionSpeech = 0;
}

function push(x) {
  const L = fir.length;
  for (let i = 0; i < x.length; i++) {
    hist.copyWithin(0, 1); hist[L - 1] = x[i];
    if (++phase < D) continue;
    phase = 0;
    let y = 0;
    for (let k = 0; k < L; k++) y += fir[k] * hist[L - 1 - k];
    ring[n % ring.length] = y; n++; total++; sinceTick++;
  }
}

function tick() {
  const len = Math.min(n, ring.length), now = total / sr, t0 = now - len / sr;
  const x = new Float32Array(len), start = n - len;
  for (let i = 0; i < len; i++) x[i] = ring[(start + i) % ring.length];
  const r = analyse(x, sr);
  const peaks = r.peaks.map((t) => t + t0), st = r.stretches.map(([a, b]) => [a + t0, b + t0]);
  // live rate: the most recent RECENT s of speech
  let need = RECENT, cnt = 0, got = 0;
  for (let j = st.length - 1; j >= 0 && need > 0; j--) {
    const [a, b] = st[j], from = Math.max(a, b - need);
    got += b - from; need -= b - from;
    cnt += peaks.filter((t) => t >= from && t < b).length;
  }
  const lastEnd = st.length ? st[st.length - 1][1] : -Infinity;
  const speaking = now - lastEnd < 0.6;
  // session totals: commit what is now older than COMMIT s
  const upto = now - COMMIT;
  if (upto > committed) {
    const from = Math.max(committed, t0);
    sessionSyl += peaks.filter((t) => t >= from && t < upto).length;
    for (const [a, b] of st) sessionSpeech += Math.max(0, Math.min(b, upto) - Math.max(a, from));
    committed = upto;
  }
  // what the page draws: the last SHOW s of loudness, syllables and speech
  const i0 = Math.max(0, r.env.length - Math.round(SHOW / HOP));
  postMessage({
    now, rate: got >= 1 ? cnt / got : null, recentSpeech: got, speaking, lastEnd,
    env: Array.from(r.env.subarray(i0)), envStart: t0 + i0 * HOP,
    peaks: peaks.filter((t) => t >= now - SHOW), stretches: st.filter(([, b]) => b >= now - SHOW),
    session: { syllables: sessionSyl, speech: sessionSpeech, rate: sessionSpeech >= 2 ? sessionSyl / sessionSpeech : null },
    sr, sr0,
  });
}

globalThis.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'init') { setup(m.sr); if (m.recent) RECENT = m.recent; return; }
  if (m.type === 'audio') {
    push(m.x);
    if (sinceTick >= TICK * sr && n >= 0.5 * sr) { sinceTick = 0; tick(); }
  }
};
