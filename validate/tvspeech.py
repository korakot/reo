"""Syllable-count check on TVSpeech (typhoon-ai/TVSpeech, CC BY): real, fast YouTube talk with human transcripts.

TVSpeech has no timings, so this checks the count only: reo's syllables against the syllables in the transcript.
Thai is read with sakot (korakot/sakot, pip install from git); a Latin word counts its vowel groups, an all-caps
acronym one syllable per letter (NGO = เอ็น-จี-โอ), digits one per digit. The rate check uses reo's own
speaking time, so its error is the count error.

    python3 -I validate/tvspeech.py /path/to/test-00000-of-00001.parquet out/tvspeech.csv
"""
import csv, io, os, re, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reo

_sakot = None


def thai_syllables(text):
    global _sakot
    if _sakot is None:
        import sakot; _sakot = sakot
    n = 0
    for _, readings in _sakot.pronounce_text(text):
        if readings: n += len(max(readings, key=lambda r: r[1])[0].split('|'))
    return n


def latin_syllables(w):
    if w.isupper() and len(w) <= 5: return len(w)
    return max(1, len(re.findall(r'[aeiouy]+', w.lower().rstrip('e') or w.lower())))


def count(text):
    text = re.sub(r'\[[^\]]*\]', ' ', text)  # [ns] and other tags
    n = 0
    for tok in re.findall(r'[฀-๿]+|[A-Za-z]+|\d', text):
        if re.match(r'[฀-๿]', tok): n += thai_syllables(tok)
        elif tok.isdigit(): n += 1
        else: n += latin_syllables(tok)
    return n


def run(row):
    X, sr = reo.read_wav(row['audio']['bytes'])
    r = reo.analyse(X[0], sr)
    n = count(row['sentence'])
    return dict(id=row['audio_id'], speaker=row['audio_id'].split('_')[0], sex='', cond='',
                true_syl=n, true_speech=round(r['speech_time'], 3), true_rate=round(n / r['speech_time'], 3),
                syl=r['syllables'], speech=round(r['speech_time'], 3), rate=round(r['rate'], 3))


if __name__ == '__main__':
    import pyarrow.parquet as pq
    src, out = sys.argv[1:3]
    rows = pq.read_table(src).to_pylist()
    with Pool(4) as p: res = p.map(run, rows, chunksize=4)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(res[0])); w.writeheader(); w.writerows(res)
    t = np.array([r['true_syl'] for r in res], float); d = np.array([r['syl'] for r in res], float)
    rate = np.array([r['true_rate'] for r in res])
    print(f'TVSpeech n={len(res)}  transcript rate median {np.median(rate):.2f} syll/s '
          f'(10th-90th pct {np.percentile(rate, 10):.2f}-{np.percentile(rate, 90):.2f})  '
          f'count error median {np.median(d / t - 1) * 100:+.1f}%  |err| median {np.median(np.abs(d / t - 1)) * 100:.1f}%  '
          f'r(rate)={np.corrcoef(rate, d / np.array([r["speech"] for r in res]))[0, 1]:.2f}')
    for lo, hi in [(0, 4.5), (4.5, 5.5), (5.5, 6.5), (6.5, 99)]:
        m = (rate >= lo) & (rate < hi)
        if m.sum(): print(f'  transcript rate {lo:.1f}-{hi:.1f}: n={m.sum():3d}  count error median {np.median(d[m] / t[m] - 1) * 100:+.1f}%')
