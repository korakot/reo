"""Check reo's rate against hand-marked syllables: TSync-1, LOTUS PD and CCOST.

The true rate of a recording is its number of syllables (one vowel per syllable in the hand-marked phones) divided
by its speaking time (the marked speech, with pauses of 120 ms or more left out, the same rule reo uses).

    python3 -I validate/check.py tsync1 /path/to/TSync1 out/tsync1.csv
    python3 -I validate/check.py lotus  /path/to/LOTUS  out/lotus.csv
    python3 -I validate/check.py ccost  /path/to/C-COST out/ccost.csv
    python3 -I validate/check.py summary out/*.csv

Where to get the corpora: README.md, section Accuracy.
"""
import csv, glob, os, re, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reo as sr_

VOWELS = {'a', 'aa', 'i', 'ii', 'u', 'uu', 'e', 'ee', 'o', 'oo', 'x', 'xx', 'v', 'vv', 'q', 'qq', '@', '@@',
          'ia', 'iia', 'ua', 'uua', 'va', 'vva'}
SILENCE = {'sil', 'sp', '.sil', 'sli', 'spsp', 'spa', 'spk', ''}


def htk(path):
    out = []
    for line in open(path, encoding='latin-1'):
        p = line.replace('\r', '').split()
        if len(p) >= 3:
            ph = p[2].lstrip('_/')
            if ph.lower() in SILENCE or ph.lower().startswith('sp'): continue
            out.append((int(p[0]) / 1e7, int(p[1]) / 1e7, ph))
    return out


def truth(segs, min_pause=0.12):
    """(syllables, speaking time) from non-silent segments."""
    n = sum(1 for _, _, ph in segs if ph in VOWELS)
    spans = []
    for a, b, _ in sorted(segs):
        if spans and a - spans[-1][1] < min_pause: spans[-1][1] = max(spans[-1][1], b)
        else: spans.append([a, b])
    return n, sum(b - a for a, b in spans)


def items(corpus, root):
    if corpus == 'tsync1':
        for lab in sorted(glob.glob(os.path.join(root, 'lab_ph', '*.lab'))):
            b = os.path.basename(lab)[:-4]
            yield dict(id=b, speaker='tsync1', sex='F', cond='', wav=os.path.join(root, 'wav', b + '.wav'), lab=lab)
    elif corpus == 'lotus':
        for cond, sub, labdir in [('CC', 'C/Clean', 'CClab'), ('UO', 'U/Office', 'UOlab')]:
            for w in sorted(glob.glob(os.path.join(root, 'PD', sub, 'Wav', '*', '*.wav'))):
                b = os.path.basename(w)[:-4]
                lab = os.path.join(root, 'PD', sub, labdir, b + '.lab')
                if os.path.exists(lab):
                    yield dict(id=b, speaker=b[2:6], sex=b[2], cond=cond, wav=w, lab=lab)
    elif corpus == 'ccost':
        yield from ccost_items(root)
    elif corpus == 'giga':
        yield from giga_items(root)


def giga_items(root):
    """GigaSpeech 2 Thai dev: YouTube speech. The truth is riang's alignment of the human transcripts
    (korakot/vox-gigaspeech2-th-devtest riang_phones.parquet, in root): syllables are exact, times machine-made."""
    import pandas as pd
    d = pd.read_parquet(os.path.join(root, 'riang_phones.parquet'))
    d = d[d['seg'].str.startswith('dev/')]
    for seg, g in d.groupby('seg', sort=True):
        split, rest = seg.split('/'); video = rest.split('-')[0]
        wav = os.path.join(root, split, video, rest + '.wav')
        if not os.path.exists(wav): continue
        # one pseudo-vowel per syllable, spanning the syllable's phones, so truth() counts syllables
        segs = [(r.start, r.end, 'a' if k == 0 else 'x-') for _, s in g.groupby('syl') for k, r in enumerate(s.itertuples())]
        yield dict(id=seg, speaker='giga/' + video, sex='', cond='', wav=wav, segs=segs)


def ccost_items(root):
    raise NotImplementedError('filled in once the CCOST layout is known')


def run(it):
    X, sr = sr_.read_audio(it['wav'])
    r = sr_.analyse(X[0], sr)
    n, sp = truth(it['segs'] if 'segs' in it else htk(it['lab']))
    return dict(id=it['id'], speaker=it['speaker'], sex=it['sex'], cond=it['cond'],
                true_syl=n, true_speech=round(sp, 3), true_rate=round(n / sp, 3) if sp else '',
                syl=r['syllables'], speech=round(r['speech_time'], 3), rate=round(r['rate'], 3) if r['rate'] else '')


def summary(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    rows = [r for r in rows if r['true_rate'] and r['rate'] and float(r['true_speech']) >= 1.0]
    def stats(rs, label):
        t = np.array([float(r['true_rate']) for r in rs]); d = np.array([float(r['rate']) for r in rs])
        ts = np.array([float(r['true_syl']) for r in rs]); ds = np.array([float(r['syl']) for r in rs])
        print(f"{label:28s} n={len(rs):5d}  true {np.median(t):.2f} syll/s  measured {np.median(d):.2f}  "
              f"count {np.median(ds / ts - 1) * 100:+5.1f}%  rate {np.median(d / t - 1) * 100:+5.1f}% "
              f"(|err| median {np.median(np.abs(d / t - 1)) * 100:4.1f}%)  r={np.corrcoef(t, d)[0, 1]:.2f}")
    def speakers(rs, label):
        by = {}
        for r in rs: by.setdefault(r['speaker'], []).append(r)
        if len(by) < 3: return
        t = np.array([sum(float(r['true_syl']) for r in v) / sum(float(r['true_speech']) for r in v) for v in by.values()])
        d = np.array([sum(float(r['syl']) for r in v) / sum(float(r['speech']) for r in v) for v in by.values()])
        rho = np.corrcoef(np.argsort(np.argsort(t)), np.argsort(np.argsort(d)))[0, 1]
        print(f"{label:28s} speakers={len(by):3d}  speaker rates r={np.corrcoef(t, d)[0, 1]:.2f}  rank rho={rho:.2f}  "
              f"rate error {np.median(d / t - 1) * 100:+5.1f}%")
    groups = {}
    for r in rows:
        corpus = ('tsync1' if r['speaker'] == 'tsync1' else 'lotus' if re.match(r'[FM]\d{3}$', r['speaker'])
                  else 'giga' if r['speaker'].startswith('giga/') else 'ccost')
        groups.setdefault(corpus, []).append(r)
    for c, rs in groups.items():
        stats(rs, c)
        for key in ('sex', 'cond'):
            vals = sorted({r[key] for r in rs})
            if len(vals) > 1:
                for v in vals: stats([r for r in rs if r[key] == v], f'  {c} {key}={v}')
        speakers(rs, f'  {c} per speaker')
        for s in sorted({r['sex'] for r in rs}):
            speakers([r for r in rs if r['sex'] == s], f'  {c} per speaker, sex={s}')


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summary(sys.argv[2:]); sys.exit()
    corpus, root, out = sys.argv[1:4]
    its = list(items(corpus, root))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with Pool(4) as p: res = p.map(run, its, chunksize=8)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(res[0])); w.writeheader(); w.writerows(res)
    summary([out])
