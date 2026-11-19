// reo.js (เร็ว): how fast someone speaks, in syllables per second, from the audio alone (no transcript).
// The twin of reo.py: same method, same settings, same numbers. Runs in the browser and in Node, no
// dependencies. Method and settings: see reo.py and korakot/pat wiki/syllable-detection.md.
//
//   import { analyse, readWav } from './reo.js';
//   const { channels, sr } = readWav(arrayBuffer);
//   const r = analyse(channels[0], sr);   // r.rate (syll/s), r.utterances, r.peaks, ...

export const HOP = 0.010;
// pat's detector settings plus speed adaptation (see DEFAULT in reo.py for why).
export const DEFAULT = {
  smooth: 6, mindip: 2.0, act_db: 25.0, nuc_db: 25.0, voi_thr: 0.45, min_pause: 0.12, min_run: 0.04,
  flat_db: 1.0, floor_db: 6.0, floor_pct: 10, use_hi: true, hi_db: 8.0, hi_max: 0.2, min_voiced: 0.03,
  adapt: true, adapt_c: 24.0, adapt_min: 2.0, adapt_max: 14.0, adapt_group: 0.5, adapt_iter: 8,
};
export const RATE = { utt_pause: 0.5, min_speech: 1.0 };

// ------------------------------------------------------------------ FFT (radix 2, complex, in place)

class FFT {
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2); this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { this.cos[i] = Math.cos((2 * Math.PI * i) / n); this.sin[i] = -Math.sin((2 * Math.PI * i) / n); }
  }
  run(re, im) {
    const n = this.n, rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let s = 0; s < n; s += size) {
        for (let k = 0; k < half; k++) {
          const c = this.cos[k * step], d = this.sin[k * step];
          const a = s + k, b = a + half;
          const tr = re[b] * c - im[b] * d, ti = re[b] * d + im[b] * c;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    }
  }
}

const pow2 = (n) => 1 << Math.ceil(Math.log2(n));
const hann = (n) => Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));

// ------------------------------------------------------------------ features (10 ms frames)

// Per-frame levels in dB (full band, 300-3300 Hz 'dj', 2.5 kHz up 'hi') and voicing strength 'voi'.
// Frame i is centred on sample i*hop. onProgress(fraction) is called now and then on long input.
// The spectrum window is 20 ms (pat uses 32 ms): a shorter window keeps the dips between fast syllables.
export function features(x, sr, onProgress, win = 0.020) {
  const hop = Math.round(HOP * sr);
  const nf = 1 + Math.floor(x.length / hop);
  const ns = Math.round(win * sr), nfs = pow2(ns), ws = hann(ns), fs = new FFT(nfs);
  const nv = Math.floor(0.040 * sr), nfv = pow2(2 * nv), wv = hann(nv), fv = new FFT(nfv);
  const binHz = sr / nfs;
  const band = (lo, hi) => { const b = []; for (let k = 0; k <= nfs / 2; k++) { const f = k * binHz; if (f >= lo && f < hi) b.push(k); } return [b[0], b[b.length - 1]]; };
  const B = { full: band(60, sr / 2), dj: band(300, 3300), hi: band(2500, sr / 2) };
  const lo = Math.floor(sr / 500), hiLag = Math.floor(sr / 70);
  // the window's own autocorrelation, for Boersma's correction
  const re = new Float64Array(Math.max(nfs, nfv)), im = new Float64Array(Math.max(nfs, nfv));
  const acorr = (n, fft) => { // in place on re/im, length n; result in re[0..]
    fft.run(re, im);
    for (let k = 0; k < n; k++) { re[k] = re[k] * re[k] + im[k] * im[k]; im[k] = 0; }
    fft.run(re, im); // power spectrum is real and even, so a forward FFT gives n times the inverse
  };
  re.fill(0); im.fill(0); re.set(wv);
  acorr(nfv, fv);
  const rw = new Float64Array(nv); for (let i = 0; i < nv; i++) rw[i] = re[i] / re[0];
  const out = { full: new Float64Array(nf), dj: new Float64Array(nf), hi: new Float64Array(nf), voi: new Float64Array(nf) };
  const at = (i) => (i >= 0 && i < x.length ? x[i] : 0);
  const pw = new Float64Array(nfs / 2 + 1);
  for (let i = 0; i < nf; i++) {
    const c = i * hop;
    // spectrum: 32 ms Hann, zero-padded to a power of two
    re.fill(0, 0, nfs); im.fill(0, 0, nfs);
    const s0 = c - Math.floor(ns / 2);
    for (let j = 0; j < ns; j++) re[j] = at(s0 + j) * ws[j];
    fs.run(re, im);
    for (let k = 0; k <= nfs / 2; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    for (const key of ['full', 'dj', 'hi']) {
      const [a, b] = B[key]; let s = 0;
      for (let k = a; k <= b; k++) s += pw[k];
      out[key][i] = 10 * Math.log10(s + 1e-10);
    }
    // voicing: normalised autocorrelation over 40 ms, best lag for F0 70-500 Hz
    re.fill(0, 0, nfv); im.fill(0, 0, nfv);
    const v0 = c - Math.floor(nv / 2);
    let mean = 0;
    for (let j = 0; j < nv; j++) mean += at(v0 + j);
    mean /= nv;
    for (let j = 0; j < nv; j++) re[j] = (at(v0 + j) - mean) * wv[j];
    acorr(nfv, fv);
    const r0 = re[0] / nfv + 1e-12;
    let best = -Infinity;
    for (let L = lo; L <= hiLag; L++) { const v = re[L] / nfv / r0 / rw[L]; if (v > best) best = v; }
    out.voi[i] = best;
    if (onProgress && i % 2000 === 0) onProgress(i / nf);
  }
  return out;
}

// ------------------------------------------------------------------ syllable nuclei (pat's detector)

function smooth(y, k) {
  if (k <= 1) return Float64Array.from(y);
  const s = k / 4, h = Math.floor(2 * s + 1);
  const g = []; let tot = 0;
  for (let r = -h; r <= h; r++) { const v = Math.exp(-0.5 * (r / s) ** 2); g.push(v); tot += v; }
  const n = y.length, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let r = -h; r <= h; r++) acc += g[r + h] * y[Math.min(n - 1, Math.max(0, i + r))];
    out[i] = acc / tot;
  }
  return out;
}

// numpy-style percentile (linear interpolation)
function percentile(y, q) {
  const a = Float64Array.from(y).sort();
  const pos = (q / 100) * (a.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

function runs(mask) {
  const out = []; let a = -1;
  for (let i = 0; i <= mask.length; i++) {
    const m = i < mask.length && mask[i];
    if (m && a < 0) a = i;
    if (!m && a >= 0) { out.push([a, i]); a = -1; }
  }
  return out; // [a, b)
}

function rangeMin(env, a, b) { let m = Infinity; for (let i = a; i <= b; i++) if (env[i] < m) m = env[i]; return m; }
const dip = (env, a, b) => Math.min(env[a], env[b]) - rangeMin(env, a, b);

function valley(env, a, b, flat) {
  let m = Infinity, i0 = a;
  for (let i = a; i <= b; i++) if (env[i] < m) { m = env[i]; i0 = i; }
  let lo = i0, hi = i0;
  while (lo > a && env[lo - 1] <= m + flat) lo--;
  while (hi < b && env[hi + 1] <= m + flat) hi++;
  return (lo + hi) / 2;
}

// Syllable nuclei. Returns { units, stretches, env }; times in frames. A unit is { start, end, peak }.
export function detect(F, opts = {}) {
  const p = { ...DEFAULT, ...opts };
  const n = F.full.length;
  if (n < 3) return { units: [], stretches: [], env: new Float64Array(n) };
  const full = smooth(F.full, 3);
  const voiced = Array.from(F.voi, (v) => v >= p.voi_thr);
  const env = smooth(F.dj, p.smooth);
  const thr = Math.max(percentile(full, 99) - p.act_db, percentile(full, p.floor_pct) + p.floor_db);
  const gap = Math.round(p.min_pause / HOP), mr = Math.round(p.min_run / HOP);
  let st = [];
  for (const [a, b] of runs(Array.from(full, (v) => v > thr))) {
    if (st.length && a - st[st.length - 1][1] < gap) st[st.length - 1][1] = b;
    else st.push([a, b]);
  }
  st = st.filter(([a, b]) => b - a >= mr);
  if (p.use_hi) { // extend starts back over onset frication and aspiration
    const hi = smooth(F.hi, 3);
    const hthr = Math.max(percentile(hi, p.floor_pct) + p.hi_db, percentile(hi, 99) - 40);
    const lim = Math.round(p.hi_max / HOP), ext = [];
    for (let [a, b] of st) {
      const lo = Math.max(ext.length ? ext[ext.length - 1][1] : 0, a - lim);
      while (a - 1 >= lo && hi[a - 1] > hthr) a--;
      ext.push([a, b]);
    }
    st = ext;
  }
  const nref = percentile(env, 99);
  const vrun = new Float64Array(n);
  for (const [a, b] of runs(voiced)) for (let i = a; i < b; i++) vrun[i] = b - a;
  const mv = p.min_voiced / HOP;
  // units in stretch [sa, sb) (frames); e is an envelope whose index 0 is frame off
  const nuclei = (e, off, sa, sb) => {
    const pk = [];
    for (let i = Math.max(1, sa - off); i < Math.min(sb - off, e.length - 1); i++) if (e[i] > e[i - 1] && e[i] >= e[i + 1]) pk.push(i);
    while (pk.length > 1) { // drop the lower peak of the pair with the shallowest dip until every dip >= mindip
      let d = Infinity, j = 0;
      for (let q = 0; q + 1 < pk.length; q++) { const x = dip(e, pk[q], pk[q + 1]); if (x < d) { d = x; j = q; } }
      if (d >= p.mindip) break;
      pk.splice(e[pk[j]] < e[pk[j + 1]] ? j : j + 1, 1);
    }
    const nuc = pk.filter((i) => {
      if (!(e[i] > nref - p.nuc_db)) return false;
      let v = false, r = 0;
      for (let j = Math.max(0, i + off - 1); j <= Math.min(n - 1, i + off + 1); j++) { v = v || voiced[j]; r = Math.max(r, vrun[j]); }
      return v && r >= mv;
    });
    return nuc.map((i, k) => ({
      start: k ? valley(e, nuc[k - 1], i, p.flat_db) + off : sa,
      end: k + 1 < nuc.length ? valley(e, i, nuc[k + 1], p.flat_db) + off : sb,
      peak: i + off,
    }));
  };
  const units = [];
  if (!p.adapt) {
    for (const [sa, sb] of st) units.push(...nuclei(env, 0, sa, sb));
    return { units, stretches: st, env };
  }
  // speed adaptation, per utterance (stretches joined across pauses shorter than adapt_group)
  const groups = [];
  for (const [a, b] of st) {
    const g = groups[groups.length - 1];
    if (g && (a - g[g.length - 1][1]) * HOP < p.adapt_group) g.push([a, b]); else groups.push([[a, b]]);
  }
  const margin = Math.floor(p.adapt_max / 2) + 4;
  for (const g of groups) {
    let T = 0;
    for (const [a, b] of g) T += b - a;
    T *= HOP;
    const lo = Math.max(0, g[0][0] - margin), hi = Math.min(n, g[g.length - 1][1] + margin);
    let k = p.smooth, us = [];
    for (let it = 0; it < p.adapt_iter; it++) {
      const e = k === p.smooth ? env.subarray(lo, hi) : smooth(F.dj.slice(lo, hi), k);
      us = [];
      for (const [sa, sb] of g) us.push(...nuclei(e, lo, sa, sb));
      if (!us.length) break;
      const k2 = Math.min(p.adapt_max, Math.max(p.adapt_min, (p.adapt_c * T) / us.length));
      if (Math.abs(k2 - k) < 0.05) break;
      k = k2;
    }
    units.push(...us);
  }
  return { units, stretches: st, env };
}

// ------------------------------------------------------------------ rate

const median = (a) => { const s = Float64Array.from(a).sort(); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

// Articulation rate and per-utterance rates from detected units and speech stretches (frames).
export function rate(units, stretches, duration, opts = {}) {
  const { utt_pause, min_speech } = { ...RATE, ...opts };
  const peaks = units.map((u) => u.peak * HOP);
  const st = stretches.map(([a, b]) => [a * HOP, b * HOP]);
  let speech = 0;
  for (const [a, b] of st) speech += b - a;
  const iv = [];
  for (const [a, b] of st) {
    const q = peaks.filter((t) => t >= a && t < b);
    for (let i = 1; i < q.length; i++) iv.push(q[i] - q[i - 1]);
  }
  const utts = [];
  for (const [a, b] of st) {
    const u = utts[utts.length - 1];
    if (u && a - u.end < utt_pause) { u.end = b; u.speech += b - a; } else utts.push({ start: a, end: b, speech: b - a });
  }
  for (const u of utts) {
    u.syllables = peaks.filter((t) => t >= u.start && t < u.end).length;
    u.rate = u.speech >= min_speech ? u.syllables / u.speech : null;
  }
  const rated = utts.filter((u) => u.rate !== null).map((u) => u.rate);
  return {
    duration, speech_time: speech, syllables: peaks.length,
    rate: speech > 0 ? peaks.length / speech : null,
    median_interval: iv.length ? median(iv) : null,
    pause_share: st.length ? 1 - speech / (st[st.length - 1][1] - st[0][0]) : null,
    utterance_rate_median: rated.length ? median(rated) : null,
    utterance_rate_p90: rated.length ? percentile(rated, 90) : null,
    utterances: utts, peaks, stretches: st,
  };
}

// Analyse one channel (Float32Array or array of samples in -1..1). opts overrides DEFAULT and RATE.
export function analyse(x, sr, opts = {}, onProgress) {
  const F = features(x, sr, onProgress);
  const { units, stretches, env } = detect(F, opts);
  const r = rate(units, stretches, x.length / sr, opts);
  r.env = env; // the 300-3300 Hz loudness curve (dB, 10 ms frames), for drawing
  return r;
}

// Share of rated speech time in utterances faster than threshold (syllables per second).
export function fastShare(r, threshold) {
  const rated = r.utterances.filter((u) => u.rate !== null);
  const tot = rated.reduce((s, u) => s + u.speech, 0);
  return tot ? rated.filter((u) => u.rate > threshold).reduce((s, u) => s + u.speech, 0) / tot : null;
}

// ------------------------------------------------------------------ WAV files

function ulaw(b) {
  const u = ~b & 0xff;
  let t = ((u & 0x0f) << 3) + 0x84;
  t <<= (u & 0x70) >> 4;
  return (u & 0x80 ? 0x84 - t : t - 0x84) / 32768;
}
function alaw(b) {
  const a = b ^ 0x55;
  let t = (a & 0x0f) << 4;
  const seg = (a & 0x70) >> 4;
  t = seg === 0 ? t + 8 : seg === 1 ? t + 0x108 : (t + 0x108) << (seg - 1);
  return (a & 0x80 ? t : -t) / 32768;
}

// RIFF WAVE (ArrayBuffer or Uint8Array) -> { channels: [Float32Array], sr }.
// PCM 8/16/24/32, float 32/64, mu-law and A-law (common in call recordings). Throws on anything else.
export function readWav(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const str = (p, n) => String.fromCharCode(...u8.subarray(p, p + n));
  if (u8.length < 12 || str(0, 4) !== 'RIFF' || str(8, 4) !== 'WAVE') throw new Error('not a WAV file');
  let p = 12, fmt = null, data = null;
  while (p + 8 <= u8.length) {
    const id = str(p, 4), size = dv.getUint32(p + 4, true);
    if (id === 'fmt ') {
      let tag = dv.getUint16(p + 8, true);
      if (tag === 0xfffe) tag = dv.getUint16(p + 32, true);
      fmt = { tag, ch: dv.getUint16(p + 10, true), sr: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    }
    if (id === 'data') { data = [p + 8, size === 0 || size === 0xffffffff ? u8.length - p - 8 : Math.min(size, u8.length - p - 8)]; break; }
    p += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error('WAV without fmt or data');
  const { tag, ch, sr, bits } = fmt, w = bits / 8, n = Math.floor(data[1] / (w * ch));
  const channels = Array.from({ length: ch }, () => new Float32Array(n));
  const rd = {
    '1/8': (o) => (u8[o] - 128) / 128, '1/16': (o) => dv.getInt16(o, true) / 32768,
    '1/24': (o) => { let v = u8[o] | (u8[o + 1] << 8) | (u8[o + 2] << 16); if (v >= 1 << 23) v -= 1 << 24; return v / (1 << 23); },
    '1/32': (o) => dv.getInt32(o, true) / 2 ** 31, '3/32': (o) => dv.getFloat32(o, true), '3/64': (o) => dv.getFloat64(o, true),
    '7/8': (o) => ulaw(u8[o]), '6/8': (o) => alaw(u8[o]),
  }[`${tag}/${bits}`];
  if (!rd) throw new Error(`unsupported WAV format ${tag} (${bits} bit)`);
  for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) channels[c][i] = rd(data[0] + (i * ch + c) * w);
  return { channels, sr };
}
