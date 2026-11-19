"""Speed test: the same recordings played faster and slower (ffmpeg rubberband, pitch kept).

The words stay the same, so a perfect detector finds the same number of syllables at every speed and its rate
grows exactly with the speed factor. Any drop in the count at high factors is syllables lost to fast speech.

    python3 -I validate/tempo.py out/tempo.csv TSYNC1_DIR LOTUS_DIR TVSPEECH_PARQUET
"""
import csv, glob, os, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reo

FACTORS = [0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0]


def stretch(wav_bytes, f):
    with tempfile.NamedTemporaryFile(suffix='.wav') as t:
        t.write(wav_bytes); t.flush()
        flt = 'anull' if f == 1.0 else f'rubberband=tempo={f}'
        out = subprocess.run(['ffmpeg', '-v', 'error', '-i', t.name, '-af', flt, '-ac', '1', '-f', 'wav',
                              '-acodec', 'pcm_s16le', '-'], capture_output=True, check=True).stdout
    return reo.read_wav(out)


def run(it):
    out = []
    for f in FACTORS:
        X, sr = stretch(it['bytes'], f)
        r = reo.analyse(X[0], sr)
        out.append(dict(id=it['id'], set=it['set'], sex=it['sex'], factor=f, syl=r['syllables'],
                        speech=round(r['speech_time'], 3), rate=round(r['rate'] or 0, 3)))
    return out


def sample(tsync, lotus, tvs):
    import pyarrow.parquet as pq
    its = []
    for w in sorted(glob.glob(os.path.join(tsync, 'wav', '*.wav')))[::65][:40]:
        its.append(dict(id=os.path.basename(w), set='tsync1 (F, read)', sex='F', bytes=open(w, 'rb').read()))
    for spk in ['F001', 'F002', 'F004', 'F005', 'M025', 'M026', 'M027', 'M028']:
        for w in sorted(glob.glob(os.path.join(lotus, 'PD', 'C', 'Clean', 'Wav', '*', f'CC{spk}*.wav')))[:8]:
            its.append(dict(id=os.path.basename(w), set=f'lotus {spk[0]} (read)', sex=spk[0], bytes=open(w, 'rb').read()))
    rows = pq.read_table(tvs).to_pylist()
    for r in rows[::10]:
        its.append(dict(id=r['audio_id'], set='tvspeech (YouTube)', sex='', bytes=r['audio']['bytes']))
    return its


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        rows = list(csv.DictReader(open(sys.argv[2])))
    else:
        out = sys.argv[1]
        its = sample(*sys.argv[2:5])
        with Pool(4) as p: res = [r for rs in p.map(run, its, chunksize=1) for r in rs]
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, list(res[0])); w.writeheader(); w.writerows(res)
        rows = [{k: str(v) for k, v in r.items()} for r in res]
    by = {}
    for r in rows: by.setdefault(r['id'], {})[float(r['factor'])] = r
    sets = sorted({r['set'] for r in rows})
    print('syllables found, relative to normal speed (1.0 = none lost); in brackets the measured rate at that speed')
    print('set'.ljust(22) + ''.join(f'{f:>14}' for f in FACTORS))
    for s in sets:
        ids = [i for i, v in by.items() if v[1.0]['set'] == s]
        line = s.ljust(22)
        for f in FACTORS:
            c = np.median([float(by[i][f]['syl']) / float(by[i][1.0]['syl']) for i in ids])
            rt = np.median([float(by[i][f]['rate']) for i in ids])
            line += f'{c:8.2f} ({rt:3.1f})'
        print(line)
