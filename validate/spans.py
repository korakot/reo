"""Span check: is the rate right on stretches of a long recording, not only on whole files?

Long recordings with a known truth are made by joining each speaker's sentences in order (LOTUS: one file per
speaker and microphone, about 3 minutes; TSync-1: blocks of 50 sentences; GigaSpeech 2: one file per video), or
taken as they are (CCOST interviews). Every file is analysed once; then windows of 5, 10, 30 and 60 s are measured
two ways and compared with the truth inside the window:

  in file   reo.span() on the whole-file analysis (how the library measures spans);
  cut out   the window's audio cut out and analysed on its own.

The truth counts a syllable when its vowel's midpoint is in the window, over the marked speech inside the window
(pauses of 120 ms or more left out). Windows with less than 40% speech are skipped.

    python3 -I validate/spans.py lotus  /path/to/LOTUS   out/spans_lotus.csv
    python3 -I validate/spans.py tsync1 /path/to/TSync1  out/spans_tsync1.csv
    python3 -I validate/spans.py giga   /path/to/giga    out/spans_giga.csv
    python3 -I validate/spans.py ccost  /path/to/C-COST  out/spans_ccost.csv
    python3 -I validate/spans.py summary out/spans_*.csv
"""
import csv, glob, os, sys
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import reo
from check import htk, VOWELS

LENGTHS = [5, 10, 30, 60]


def join(parts):
    """parts: [(wav path, [(s, e, ph)])] -> (x, sr, vowel midpoints, speech intervals)."""
    xs, mids, segs, off, sr0 = [], [], [], 0.0, None
    for wav, ph in parts:
        X, sr = reo.read_audio(wav)
        if sr0 is None: sr0 = sr
        if sr != sr0: continue
        xs.append(X[0])
        for s, e, p in ph:
            segs.append((s + off, e + off))
            if p in VOWELS: mids.append(off + (s + e) / 2)
        off += len(X[0]) / sr
    spans = []
    for a, b in sorted(segs):
        if spans and a - spans[-1][1] < 0.12: spans[-1][1] = max(spans[-1][1], b)
        else: spans.append([a, b])
    return np.concatenate(xs), sr0, np.array(mids), spans


def files(corpus, root):
    if corpus == 'lotus':
        for cond, sub, labdir in [('CC', 'C/Clean', 'CClab'), ('UO', 'U/Office', 'UOlab')]:
            for d in sorted(glob.glob(os.path.join(root, 'PD', sub, 'Wav', '*'))):
                parts = []
                for w in sorted(glob.glob(os.path.join(d, '*.wav'))):
                    lab = os.path.join(root, 'PD', sub, labdir, os.path.basename(w)[:-4] + '.lab')
                    if os.path.exists(lab): parts.append((w, htk(lab)))
                spk = os.path.basename(d)[2:6]
                if parts: yield dict(id=f'{spk}-{cond}', sex=spk[0], parts=parts)
    elif corpus == 'tsync1':
        labs = sorted(glob.glob(os.path.join(root, 'lab_ph', '*.lab')))
        for k in range(0, len(labs), 50):
            parts = [(os.path.join(root, 'wav', os.path.basename(l)[:-4] + '.wav'), htk(l)) for l in labs[k:k + 50]]
            yield dict(id=f'tsync1-{k // 50:02d}', sex='F', parts=parts)
    elif corpus == 'giga':
        import pandas as pd
        d = pd.read_parquet(os.path.join(root, 'riang_phones.parquet'))
        d = d[d['seg'].str.startswith('dev/')]
        d['video'] = d['seg'].str.split('/').str[1].str.split('-').str[0]
        d['idx'] = d['seg'].str.split('-').str[1].astype(int)
        for video, g in d.groupby('video'):
            parts = []
            for idx, s in g.groupby('idx'):
                wav = os.path.join(root, 'dev', video, f'{video}-{idx}.wav')
                if not os.path.exists(wav): continue
                ph = [(r.start, r.end, 'a' if k == 0 else 'x-') for _, q in s.groupby('syl') for k, r in enumerate(q.itertuples())]
                parts.append((wav, ph))
            if parts: yield dict(id=f'giga-{video}', sex='', parts=parts)
    elif corpus == 'ccost':
        from ccost_read import interviews
        for f in interviews(root):
            f['ccost'] = True
            yield f


def truth_in(mids, spans, a, b):
    n = int(((mids >= a) & (mids < b)).sum())
    sp = sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans)
    return n, sp


def run_turns(f):
    """CCOST: each interviewee turn is a span; the interviewer's unlabelled speech is around it."""
    X, sr = reo.read_audio(f['wav'])
    x = X[0]
    r = reo.analyse(x, sr)
    mids = np.array(f['mids'])
    spans = []
    for a, b in sorted(f['phones']):
        if spans and a - spans[-1][1] < 0.12: spans[-1][1] = max(spans[-1][1], b)
        else: spans.append([a, b])
    rows = []
    for a, b in f['turns']:
        n, sp = truth_in(mids, spans, a, b)
        if sp < 1.0 or n == 0: continue
        q = reo.span(r, a, b)
        c = reo.analyse(x[int(round(a * sr)):int(round(b * sr))], sr)
        rows.append(dict(file=f['id'], sex=f['sex'], length='turn', start=round(a, 2), end=round(b, 2), true_syl=n,
                         true_speech=round(sp, 3), syl=q['syllables'], speech=round(q['speech'], 3), rate=q['rate'],
                         cut_rate=c['rate'], speaker=f['speaker'], age=f['age']))
    return rows


def run(f):
    if f.get('ccost'): return run_turns(f)
    if 'x' in f: x, sr, mids, spans = f['x'], f['sr'], f['mids'], f['spans']
    else: x, sr, mids, spans = join(f['parts'])
    dur = len(x) / sr
    r = reo.analyse(x, sr)
    n, sp = truth_in(mids, spans, 0, dur)
    rows = [dict(file=f['id'], sex=f['sex'], length='whole', start=0, end=round(dur, 2), true_syl=n, true_speech=round(sp, 3),
                 syl=r['syllables'], speech=round(r['speech_time'], 3), rate=r['rate'], cut_rate=r['rate'])]
    for L in LENGTHS:
        for a in np.arange(0, dur - L + 1e-9, L):
            b = a + L
            n, sp = truth_in(mids, spans, a, b)
            if sp < 0.4 * L or n == 0: continue
            q = reo.span(r, a, b)
            c = reo.analyse(x[int(round(a * sr)):int(round(b * sr))], sr)
            rows.append(dict(file=f['id'], sex=f['sex'], length=L, start=round(a, 2), end=round(b, 2), true_syl=n,
                             true_speech=round(sp, 3), syl=q['syllables'], speech=round(q['speech'], 3), rate=q['rate'],
                             cut_rate=c['rate']))
    return rows


def speakers(paths):
    """CCOST: each speaker's rate over all their turns (spans), measured against true, by sex."""
    rows = [r for p in paths for r in csv.DictReader(open(p)) if r['length'] == 'turn' and r['rate']]
    if not rows: return
    by = {}
    for r in rows: by.setdefault(r['speaker'], []).append(r)
    t = np.array([sum(float(r['true_syl']) for r in v) / sum(float(r['true_speech']) for r in v) for v in by.values()])
    m = np.array([sum(float(r['syl']) for r in v) / sum(float(r['speech']) for r in v) for v in by.values()])
    sx = np.array([v[0]['sex'] for v in by.values()])
    rk = lambda a: np.argsort(np.argsort(a))
    print(f'  per speaker (all their turns): {len(by)} speakers, true {np.median(t):.2f}/s (range {t.min():.2f}-{t.max():.2f}), '
          f'rate error {np.median(m / t - 1) * 100:+.1f}%, |error| median {np.median(np.abs(m / t - 1)) * 100:.1f}%, '
          f'r={np.corrcoef(t, m)[0, 1]:.2f}, rank rho={np.corrcoef(rk(t), rk(m))[0, 1]:.2f}')
    for s in sorted(set(sx)):
        k = sx == s
        print(f'    sex={s} {k.sum()} speakers: rate error {np.median(m[k] / t[k] - 1) * 100:+.1f}%, r={np.corrcoef(t[k], m[k])[0, 1]:.2f}')


def summary(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    for c in sorted({os.path.basename(p).replace('spans_', '').replace('.csv', '') for p in paths}):
        rs = [r for p in paths if c in os.path.basename(p) for r in csv.DictReader(open(p))]
        nfiles = len({r['file'] for r in rs})
        print(f'{c}: {nfiles} files')
        for L in ['whole'] + [str(l) for l in LENGTHS] + ['turn']:
            q = [r for r in rs if r['length'] == L and r['rate'] and r['cut_rate']]
            if len(q) < 3: continue
            t = np.array([float(r['true_syl']) / float(r['true_speech']) for r in q])
            m = np.array([float(r['rate']) for r in q]); cut = np.array([float(r['cut_rate']) for r in q])
            e = m / t - 1; d = np.abs(cut / m - 1)
            label = 'whole file' if L == 'whole' else 'turns' if L == 'turn' else f'{L} s windows'
            print(f'  {label:14s} n={len(q):5d}  true {np.median(t):.2f}/s  rate error {np.median(e) * 100:+5.1f}%  '
                  f'|error| median {np.median(np.abs(e)) * 100:4.1f}%, 90% under {np.percentile(np.abs(e), 90) * 100:4.1f}%  '
                  f'r={np.corrcoef(t, m)[0, 1]:.2f}   in file vs cut out: median {np.median(d) * 100:.1f}%, '
                  f'90% under {np.percentile(d, 90) * 100:.1f}%')
            for s in sorted({r['sex'] for r in q} - {''}):
                k = np.array([r['sex'] == s for r in q])
                if k.sum() >= 3:
                    print(f'    sex={s} n={k.sum():5d}  rate error {np.median(e[k]) * 100:+5.1f}%  '
                          f'|error| median {np.median(np.abs(e[k])) * 100:4.1f}%  r={np.corrcoef(t[k], m[k])[0, 1]:.2f}')


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summary(sys.argv[2:]); speakers(sys.argv[2:]); sys.exit()
    corpus, root, out = sys.argv[1:4]
    fs = list(files(corpus, root))
    with Pool(4) as p: res = [r for rs in p.imap_unordered(run, fs) for r in rs]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(res[0]))
        w.writeheader(); w.writerows(res)
    summary([out]); speakers([out])
