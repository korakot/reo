"""reo (เร็ว): how fast someone speaks, in syllables per second, from the audio alone (no transcript).

Method (korakot/pat wiki/syllable-detection.md): find syllable nuclei as peaks of loudness in the 300-3300 Hz band
(de Jong & Wempe 2009, with that band and pat's settings), keep peaks that are voiced, and divide the number of
syllables by the time spent speaking (pauses of 120 ms or more left out). That is the articulation rate.

Needs only numpy. The JS twin, reo.js, gives the same numbers on the same audio.

    python reo.py call.wav                 # one line per file (and per channel)
    python reo.py *.wav --csv files.csv --utterances utt.csv --threshold 6.5
    python reo.py --spans spans.csv --span-csv span_rates.csv       # rows: file,start,end[,channel][,label]
    python reo.py call.wav --window 10 --windows every10s.csv

    import reo
    r = reo.analyse_file('call.wav')[0]    # first channel
    r['rate'], r['utterances']
    reo.span(r, 30.0, 45.0)['rate']                # any stretch of time, from the same analysis
"""
import argparse, csv, os, shutil, struct, subprocess, sys
import numpy as np

HOP = 0.010
# pat's detector settings, plus speed adaptation (adapt): with one fixed smoothing width, slow speech is
# over-counted (long syllables split) and fast speech under-counted (syllables merge). So each utterance is
# smoothed over a width in proportion to its own syllable length, adapt_c / rate frames, found by iterating
# until the rate settles. On audio played faster or slower that keeps the count the same (validate/tempo.py).
DEFAULT = dict(smooth=6, mindip=2.0, act_db=30.0, nuc_db=25.0, voi_thr=0.42, min_pause=0.12, min_run=0.04,
               flat_db=1.0, floor_db=6.0, floor_pct=10, use_hi=True, hi_db=8.0, hi_max=0.2, min_voiced=0.03,
               adapt=True, adapt_c=24.0, adapt_min=2.0, adapt_max=14.0, adapt_group=0.5, adapt_iter=8,
               denoise=True, noise_pct=20.0, noise_alpha=1.0, noise_range=50.0, act_band='dj')
# utterances: speech joined across pauses shorter than utt_pause; a rate is given only with min_speech of speech
RATE = dict(utt_pause=0.5, min_speech=1.0)


# ---------------------------------------------------------------- features (10 ms frames)

def _pow2(n):
    return 1 << int(np.ceil(np.log2(n)))


def _hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def features(x, sr, chunk=4096, win=0.020):
    """Per-frame levels in dB (full band, 300-3300 Hz 'dj', 2.5 kHz up 'hi') and voicing strength 'voi'.
    Frame i is centred on sample i*hop. The spectrum window is 20 ms (pat uses 32 ms): a shorter window keeps the
    dips between fast syllables (validate/tempo.py)."""
    x = np.asarray(x, dtype=np.float64)
    hop = int(round(HOP * sr))
    nf = 1 + len(x) // hop
    ns = int(round(win * sr)); nfs = _pow2(ns); ws = _hann(ns)
    f = np.fft.rfftfreq(nfs, 1.0 / sr)
    bands = {'full': (f >= 60) & (f < sr / 2), 'dj': (f >= 300) & (f < 3300), 'hi': (f >= 2500) & (f < sr / 2)}
    nv = int(0.040 * sr); nfv = _pow2(2 * nv); wv = _hann(nv)
    rw = np.fft.irfft(np.abs(np.fft.rfft(wv, nfv)) ** 2)[:nv]; rw = rw / rw[0]
    lo, hi = int(sr / 500), int(sr / 70)
    pad = max(ns, nv)
    xp = np.concatenate([np.zeros(pad), x, np.zeros(2 * pad)])
    out = {k: np.empty(nf) for k in ('full', 'dj', 'hi', 'voi')}
    for i0 in range(0, nf, chunk):
        idx = hop * np.arange(i0, min(nf, i0 + chunk))[:, None] + pad
        S = xp[idx - ns // 2 + np.arange(ns)[None, :]]
        S = (S - S.mean(1, keepdims=True)) * ws  # frame mean removed: a DC offset would leak into the low bands
        P = np.abs(np.fft.rfft(S, nfs, axis=1)) ** 2
        for k, m in bands.items():
            out[k][i0:i0 + len(idx)] = 10 * np.log10(P[:, m].sum(1) + 1e-10)
        V = xp[idx - nv // 2 + np.arange(nv)[None, :]]
        V = (V - V.mean(1, keepdims=True)) * wv
        R = np.fft.irfft(np.abs(np.fft.rfft(V, nfv, axis=1)) ** 2, axis=1)[:, :hi + 1]
        out['voi'][i0:i0 + len(idx)] = (R[:, lo:] / (R[:, :1] + 1e-12) / rw[lo:hi + 1]).max(1)
    return out


# ---------------------------------------------------------------- syllable nuclei (pat's detector)

def _smooth(y, k):
    if k <= 1: return np.asarray(y, float)
    s = k / 4.0; h = int(2 * s + 1)
    g = np.exp(-0.5 * (np.arange(-h, h + 1) / s) ** 2); g /= g.sum()
    yp = np.concatenate([np.full(h, y[0]), y, np.full(h, y[-1])])
    return np.convolve(yp, g, mode='valid')


def _runs(mask):
    d = np.diff(np.concatenate([[0], np.asarray(mask, int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))  # [a, b)


def _dip(env, a, b):
    return min(env[a], env[b]) - env[a:b + 1].min()


def _valley(env, a, b, flat):
    seg = env[a:b + 1]; m = seg.min(); i0 = int(np.argmin(seg)); lo = hi = i0
    while lo > 0 and seg[lo - 1] <= m + flat: lo -= 1
    while hi < len(seg) - 1 and seg[hi + 1] <= m + flat: hi += 1
    return a + (lo + hi) / 2.0


def denoise(F, pct=20.0, alpha=1.0, rng=50.0):
    """Subtract each band's noise power (its pct-th percentile over the recording, times alpha) before the levels
    are used, with a floor rng dB under the band's 99th percentile. Without this, steady noise raises the level
    floor that separates speech from pauses, and the quiet parts of speech are counted as pauses."""
    out = dict(F)
    for k in ('full', 'dj', 'hi'):
        P = 10 ** (np.asarray(F[k], float) / 10)
        N = np.percentile(P, pct) * alpha
        top = np.percentile(P, 99)
        out[k] = 10 * np.log10(np.maximum(P - N, 0) + top * 10 ** (-rng / 10))
    return out


def detect(F, **kw):
    """Syllable nuclei. Returns (units, stretches); times in frames. A unit is dict(start, end, peak)."""
    p = dict(DEFAULT); p.update(kw)
    n = len(F['full'])
    if n < 3: return [], []
    if p['denoise']: F = denoise(F, p['noise_pct'], p['noise_alpha'], p['noise_range'])
    full = _smooth(F[p['act_band']], 3)
    voiced = F['voi'] >= p['voi_thr']
    env = _smooth(F['dj'], p['smooth'])
    thr = max(np.percentile(full, 99) - p['act_db'], np.percentile(full, p['floor_pct']) + p['floor_db'])
    gap = int(round(p['min_pause'] / HOP)); mr = int(round(p['min_run'] / HOP))
    st = []
    for a, b in _runs(full > thr):
        if st and a - st[-1][1] < gap: st[-1] = (st[-1][0], b)
        else: st.append((a, b))
    st = [(a, b) for a, b in st if b - a >= mr]
    if p['use_hi']:  # extend starts back over onset frication and aspiration
        hi = _smooth(F['hi'], 3)
        hact = hi > max(np.percentile(hi, p['floor_pct']) + p['hi_db'], np.percentile(hi, 99) - 40)
        lim = int(round(p['hi_max'] / HOP)); ext = []
        for a, b in st:
            lo = max(ext[-1][1] if ext else 0, a - lim)
            while a - 1 >= lo and hact[a - 1]: a -= 1
            ext.append((a, b))
        st = ext
    nref = np.percentile(env, 99)
    vrun = np.zeros(n)
    for a, b in _runs(voiced): vrun[a:b] = b - a
    mv = p['min_voiced'] / HOP

    def nuclei(e, off, sa, sb):
        """Units in stretch [sa, sb) (frames); e is an envelope whose index 0 is frame off."""
        pk = [i for i in range(max(1, sa - off), min(sb - off, len(e) - 1)) if e[i] > e[i - 1] and e[i] >= e[i + 1]]
        while len(pk) > 1:  # drop the lower peak of the pair with the shallowest dip until every dip >= mindip
            d, j = min((_dip(e, pk[j], pk[j + 1]), j) for j in range(len(pk) - 1))
            if d >= p['mindip']: break
            pk.pop(j if e[pk[j]] < e[pk[j + 1]] else j + 1)
        nuc = [i for i in pk if e[i] > nref - p['nuc_db'] and voiced[max(0, i + off - 1):i + off + 2].any()
               and vrun[max(0, i + off - 1):i + off + 2].max() >= mv]
        out = []
        for k, i in enumerate(nuc):
            a = _valley(e, nuc[k - 1], i, p['flat_db']) + off if k else sa
            b = _valley(e, i, nuc[k + 1], p['flat_db']) + off if k + 1 < len(nuc) else sb
            out.append(dict(start=a, end=b, peak=i + off))
        return out

    if not p['adapt']:
        return [u for sa, sb in st for u in nuclei(env, 0, sa, sb)], st
    # speed adaptation, per utterance (stretches joined across pauses shorter than adapt_group)
    groups = []
    for a, b in st:
        if groups and (a - groups[-1][-1][1]) * HOP < p['adapt_group']: groups[-1].append((a, b))
        else: groups.append([(a, b)])
    margin = int(p['adapt_max'] / 2) + 4   # wider than the widest kernel's half-width, so edges match a full smooth
    units = []
    for g in groups:
        T = sum(b - a for a, b in g) * HOP
        lo, hi = max(0, g[0][0] - margin), min(n, g[-1][1] + margin)
        k, us = float(p['smooth']), []
        for _ in range(p['adapt_iter']):
            e = env[lo:hi] if k == p['smooth'] else _smooth(F['dj'][lo:hi], k)
            us = [u for sa, sb in g for u in nuclei(e, lo, sa, sb)]
            if not us: break
            k2 = min(p['adapt_max'], max(p['adapt_min'], p['adapt_c'] * T / len(us)))
            if abs(k2 - k) < 0.05: break
            k = k2
        units += us
    return units, st


# ---------------------------------------------------------------- rate

def rate(units, stretches, duration, utt_pause=RATE['utt_pause'], min_speech=RATE['min_speech']):
    """Articulation rate and per-utterance rates from detected units and speech stretches (frames)."""
    peaks = np.array([u['peak'] for u in units], float) * HOP
    st = [(a * HOP, min(b * HOP, duration)) for a, b in stretches]  # the last frame can reach past the end
    speech = sum(b - a for a, b in st)
    iv = []
    for a, b in st:
        q = peaks[(peaks >= a) & (peaks < b)]
        iv += list(np.diff(q))
    utts = []
    for a, b in st:
        if utts and a - utts[-1]['end'] < utt_pause:
            utts[-1]['end'] = b; utts[-1]['speech'] += b - a
        else:
            utts.append(dict(start=a, end=b, speech=b - a))
    for u in utts:
        u['syllables'] = int(((peaks >= u['start']) & (peaks < u['end'])).sum())
        u['rate'] = u['syllables'] / u['speech'] if u['speech'] >= min_speech else None
    rated = [u['rate'] for u in utts if u['rate'] is not None]
    return dict(
        duration=duration, speech_time=speech, syllables=len(peaks),
        rate=len(peaks) / speech if speech > 0 else None,
        median_interval=float(np.median(iv)) if iv else None,
        pause_share=1 - speech / (st[-1][1] - st[0][0]) if st else None,
        utterance_rate_median=float(np.median(rated)) if rated else None,
        utterance_rate_p90=float(np.percentile(rated, 90)) if rated else None,
        utterances=utts, peaks=[float(t) for t in peaks], stretches=[(float(a), float(b)) for a, b in st])


def analyse(x, sr, **kw):
    """Analyse one channel. Keyword arguments override DEFAULT (detector) and RATE (utterances)."""
    det = {k: v for k, v in kw.items() if k in DEFAULT}
    rt = {k: v for k, v in kw.items() if k in RATE}
    F = features(x, sr)
    units, st = detect(F, **det)
    return rate(units, st, len(x) / sr, **rt)


def span(r, start, end, min_speech=0.5):
    """Rate over [start, end) seconds of an analysed recording r: the syllables whose peak lies in the span, over
    the speech inside it. Measuring a span of the whole-file analysis (rather than cutting the audio first) keeps
    the level thresholds of the whole recording, so spans of one file are measured alike. A rate is given when the
    span holds at least min_speech seconds of speech."""
    speech = float(sum(max(0.0, min(b, end) - max(a, start)) for a, b in r['stretches']))
    q = [t for t in r['peaks'] if start <= t < end]
    iv = []
    for a, b in r['stretches']:
        p = [t for t in q if a <= t < b]
        iv += [y - x for x, y in zip(p, p[1:])]
    return dict(start=start, end=end, speech=speech, syllables=len(q),
                rate=len(q) / speech if speech >= min_speech and speech > 0 else None,
                median_interval=float(np.median(iv)) if iv else None)


def windows(r, size, step=None, min_speech=0.5):
    """span() over consecutive windows of size seconds (every step seconds, default = size) across the recording."""
    step = step or size
    out, t = [], 0.0
    while t < r['duration']:
        out.append(span(r, t, min(t + size, r['duration']), min_speech)); t += step
    return out


def fast_share(r, threshold):
    """Share of rated speech time in utterances faster than threshold (syllables per second)."""
    rated = [u for u in r['utterances'] if u['rate'] is not None]
    tot = sum(u['speech'] for u in rated)
    return sum(u['speech'] for u in rated if u['rate'] > threshold) / tot if tot else None


# ---------------------------------------------------------------- audio files

def _ulaw(b):
    u = ~b.astype(np.int32) & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t <<= (u & 0x70) >> 4
    return np.where(u & 0x80, 0x84 - t, t - 0x84) / 32768.0


def _alaw(b):
    a = b.astype(np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0)))
    return np.where(a & 0x80, t, -t) / 32768.0


def read_wav(data):
    """RIFF WAVE bytes -> (channels [C, N] float, sr). PCM 8/16/24/32, float 32/64, mu-law, A-law."""
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE': raise ValueError('not a WAV file')
    p, fmt, raw = 12, None, None
    while p + 8 <= len(data):
        cid, size = data[p:p + 4], struct.unpack('<I', data[p + 4:p + 8])[0]
        if cid == b'fmt ':
            tag, ch, sr = struct.unpack('<HHI', data[p + 8:p + 16]); bits = struct.unpack('<H', data[p + 22:p + 24])[0]
            if tag == 0xFFFE: tag = struct.unpack('<H', data[p + 32:p + 34])[0]
            fmt = (tag, ch, sr, bits)
        if cid == b'data':
            raw = data[p + 8:p + 8 + size] if size not in (0, 0xFFFFFFFF) else data[p + 8:]
            break
        p += 8 + size + (size & 1)
    if fmt is None or raw is None: raise ValueError('WAV without fmt or data')
    tag, ch, sr, bits = fmt
    w = bits // 8
    raw = raw[:len(raw) // (w * ch) * w * ch]
    if tag == 1 and bits == 8: x = (np.frombuffer(raw, np.uint8).astype(float) - 128) / 128
    elif tag == 1 and bits == 16: x = np.frombuffer(raw, '<i2') / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16); x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
    elif tag == 1 and bits == 32: x = np.frombuffer(raw, '<i4') / 2.0 ** 31
    elif tag == 3 and bits == 32: x = np.frombuffer(raw, '<f4').astype(float)
    elif tag == 3 and bits == 64: x = np.frombuffer(raw, '<f8')
    elif tag == 7: x = _ulaw(np.frombuffer(raw, np.uint8))
    elif tag == 6: x = _alaw(np.frombuffer(raw, np.uint8))
    else: raise ValueError(f'unsupported WAV format {tag} ({bits} bit)')
    return x.reshape(-1, ch).T, sr


def read_audio(path):
    """Any audio file -> (channels [C, N], sr). WAV is read directly; other formats go through ffmpeg."""
    with open(path, 'rb') as f: data = f.read()
    try:
        return read_wav(data)
    except ValueError:
        if not shutil.which('ffmpeg'): raise ValueError(f'{path}: not a WAV file I can read, and ffmpeg is not installed')
        out = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'wav', '-acodec', 'pcm_s16le', '-'],
                             capture_output=True, check=True).stdout
        return read_wav(out)


def analyse_file(path, channels=None, **kw):
    """List of results, one per channel (all channels, or the indices given)."""
    X, sr = read_audio(path)
    chans = range(len(X)) if channels is None else channels
    out = []
    for c in chans:
        r = analyse(X[c], sr, **kw); r['file'] = path; r['channel'] = c; r['sr'] = sr
        out.append(r)
    return out


def _f(v, d=2):
    return '' if v is None else f'{v:.{d}f}'


def main(argv=None):
    ap = argparse.ArgumentParser(description='Speaking rate (syllables per second of speech) from audio files.')
    ap.add_argument('files', nargs='*')
    ap.add_argument('--channel', default='all', help="'all' (default) or a channel number, 0 = left")
    ap.add_argument('--threshold', type=float, default=None, help='flag utterances faster than this (syll/s)')
    ap.add_argument('--csv', help='write one row per file and channel')
    ap.add_argument('--utterances', help='write one row per utterance')
    ap.add_argument('--spans', help='CSV of spans to measure: file,start,end[,channel][,label] (seconds); its files '
                    'are analysed too')
    ap.add_argument('--span-csv', help='write one row per span (with --spans)')
    ap.add_argument('--window', type=float, help='also measure consecutive windows of this many seconds')
    ap.add_argument('--windows', help='write one row per window (with --window)')
    ap.add_argument('--mindip', type=float, default=DEFAULT['mindip'])
    a = ap.parse_args(argv)
    chans = None if a.channel == 'all' else [int(a.channel)]
    want = []
    if a.spans:
        with open(a.spans, newline='') as f: want = list(csv.DictReader(f))
        a.files = list(dict.fromkeys(a.files + [w['file'] for w in want]))
    srows, wrows = [], []
    cols = ['file', 'channel', 'duration', 'speech_time', 'syllables', 'rate', 'median_interval_ms',
            'pause_share', 'utterance_rate_median', 'utterance_rate_p90'] + (['fast_share'] if a.threshold else [])
    rows, urows = [], []
    for path in a.files:
        try:
            res = analyse_file(path, chans, mindip=a.mindip)
        except Exception as e:  # keep going over a batch
            print(f'{path}: {e}', file=sys.stderr); continue
        for r in res:
            row = dict(file=path, channel=r['channel'], duration=_f(r['duration']), speech_time=_f(r['speech_time']),
                       syllables=r['syllables'], rate=_f(r['rate']),
                       median_interval_ms=_f(r['median_interval'] and r['median_interval'] * 1000, 0),
                       pause_share=_f(r['pause_share']), utterance_rate_median=_f(r['utterance_rate_median']),
                       utterance_rate_p90=_f(r['utterance_rate_p90']))
            if a.threshold: row['fast_share'] = _f(fast_share(r, a.threshold))
            rows.append(row)
            for u in r['utterances']:
                urows.append(dict(file=path, channel=r['channel'], start=_f(u['start']), end=_f(u['end']),
                                  speech=_f(u['speech']), syllables=u['syllables'], rate=_f(u['rate']),
                                  fast='' if a.threshold is None or u['rate'] is None else int(u['rate'] > a.threshold)))
            for w in want:
                if w['file'] == path and (not w.get('channel') or int(w['channel']) == r['channel']):
                    sp = span(r, float(w['start']), float(w['end']))
                    srows.append(dict(file=path, channel=r['channel'], label=w.get('label', ''), start=w['start'], end=w['end'],
                                      speech=_f(sp['speech']), syllables=sp['syllables'], rate=_f(sp['rate']),
                                      median_interval_ms=_f(sp['median_interval'] and sp['median_interval'] * 1000, 0)))
                    print(f"  span {w.get('label') or ''} {w['start']}-{w['end']} s ch{r['channel']}: {srows[-1]['rate'] or '-'} syll/s"
                          f" ({sp['syllables']} syllables in {sp['speech']:.2f} s of speech)")
            if a.window:
                for sp in windows(r, a.window):
                    wrows.append(dict(file=path, channel=r['channel'], start=_f(sp['start']), end=_f(sp['end']),
                                      speech=_f(sp['speech']), syllables=sp['syllables'], rate=_f(sp['rate']),
                                      median_interval_ms=_f(sp['median_interval'] and sp['median_interval'] * 1000, 0)))
            print(f"{os.path.basename(path)} ch{r['channel']}: {row['rate']} syll/s over {row['speech_time']} s of speech"
                  f" ({r['syllables']} syllables; median interval {row['median_interval_ms']} ms"
                  + (f"; {float(row['fast_share'] or 0) * 100:.0f}% of speech above {a.threshold}" if a.threshold else '') + ')')
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
    if a.utterances:
        with open(a.utterances, 'w', newline='') as f:
            w = csv.DictWriter(f, ['file', 'channel', 'start', 'end', 'speech', 'syllables', 'rate', 'fast'])
            w.writeheader(); w.writerows(urows)
    cols = ['file', 'channel', 'start', 'end', 'speech', 'syllables', 'rate', 'median_interval_ms']
    if a.span_csv:
        with open(a.span_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, cols[:2] + ['label'] + cols[2:]); w.writeheader(); w.writerows(srows)
    if a.windows:
        with open(a.windows, 'w', newline='') as f:
            w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(wrows)


if __name__ == '__main__':
    main()
