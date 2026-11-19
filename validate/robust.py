"""Consistency check: the same speech recorded or stored differently should get the same rate.

Each recording is changed in one way at a time (level, clipping, sample rate, telephone line, low-bitrate MP3, noise,
hum, DC offset), analysed again, and compared with the original: the whole-file rate and the rate in each 10 s
window (reo.windows). Recordings: LOTUS speakers (sentences joined, as in spans.py), TVSpeech clips (YouTube talk).

    python3 -I validate/robust.py out/robust.csv LOTUS_DIR TVSPEECH_PARQUET
    python3 -I validate/robust.py summary out/robust.csv
"""
import csv, io, os, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import reo

FF = {  # ffmpeg filters and output options; the result is decoded back to WAV
    '8 kHz': ['-ar', '8000'],
    '44.1 kHz': ['-ar', '44100'],
    'telephone (300-3400 Hz, 8 kHz, mu-law)': ['-af', 'highpass=f=300,lowpass=f=3400', '-ar', '8000', '-acodec', 'pcm_mulaw'],
    'MP3 24 kbit/s': ['-ar', '16000', '-acodec', 'libmp3lame', '-b:a', '24k', '-f', 'mp3'],
}


def wav_bytes(x, sr):
    b = io.BytesIO()
    y = np.clip(x, -1, 1)
    pcm = (y * 32767).astype('<i2').tobytes()
    import struct
    b.write(b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16)
            + b'data' + struct.pack('<I', len(pcm)) + pcm)
    return b.getvalue()


def ffmpeg(x, sr, args):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'in.wav'); mid = os.path.join(d, 'mid' + ('.mp3' if 'mp3' in args else '.wav'))
        open(src, 'wb').write(wav_bytes(x, sr))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src] + args + [mid], check=True)
        out = subprocess.run(['ffmpeg', '-v', 'error', '-i', mid, '-f', 'wav', '-acodec', 'pcm_s16le', '-'],
                             capture_output=True, check=True).stdout
    X, s = reo.read_wav(out)
    return X[0], s


def variants(x, sr, seed):
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(x[np.abs(x) > 0.02 * np.abs(x).max()] ** 2))  # level of the speech, not the silences
    t = np.arange(len(x)) / sr
    yield 'level -20 dB', x * 0.1, sr
    yield 'level +12 dB, clipped', np.clip(x * 4, -1, 1), sr
    for name, args in FF.items():
        y, s = ffmpeg(x, sr, args); yield name, y, s
    for snr in (20, 10):
        yield f'white noise, SNR {snr} dB', x + rng.normal(0, rms / 10 ** (snr / 20), len(x)), sr
    yield 'mains hum 50 Hz (-20 dB)', x + 0.1 * rms * np.sqrt(2) * np.sin(2 * np.pi * 50 * t), sr
    yield 'DC offset 0.05', x + 0.05, sr


def run(item):
    x, sr = item['x'], item['sr']
    r0 = reo.analyse(x, sr)
    w0 = reo.windows(r0, 10)
    out = []
    for name, y, s in variants(x, sr, hash(item['id']) & 0xffff):
        r = reo.analyse(y, s)
        w = reo.windows(r, 10)
        wc = [b['rate'] / a['rate'] - 1 for a, b in zip(w0, w) if a['rate'] and b['rate']]
        out.append(dict(id=item['id'], set=item['set'], variant=name, rate0=r0['rate'], rate=r['rate'],
                        change=r['rate'] / r0['rate'] - 1 if r['rate'] and r0['rate'] else '',
                        win_abs_change_median=float(np.median(np.abs(wc))) if wc else '', windows=len(wc)))
    return out


def items(lotus, tvs):
    from spans import files, join
    out = []
    for f in files('lotus', lotus):
        if f['id'].endswith('-CC') and f['id'][1:4] in ('001', '002', '004', '005', '025', '026', '027', '028'):
            x, sr, _, _ = join(f['parts']); out.append(dict(id=f['id'], set=f'LOTUS {f["sex"]}, read', x=x, sr=sr))
    import pyarrow.parquet as pq
    for r in pq.read_table(tvs).to_pylist()[::19]:
        X, sr = reo.read_wav(r['audio']['bytes']); out.append(dict(id=r['audio_id'], set='TVSpeech, YouTube', x=X[0], sr=sr))
    return out


def summary(rows):
    print('rate change against the original recording: whole file (median, worst) and 10 s windows (median |change|)')
    for v in dict.fromkeys(r['variant'] for r in rows):
        line = f'{v:40s}'
        for s in sorted({r['set'] for r in rows}):
            q = [r for r in rows if r['variant'] == v and r['set'] == s and r['change'] != '']
            c = np.array([float(r['change']) for r in q]); wcm = np.median([float(r['win_abs_change_median']) for r in q if r['win_abs_change_median'] != ''])
            line += f'  {s[:10]:10s} {np.median(c) * 100:+5.1f}% (worst {c[np.argmax(np.abs(c))] * 100:+5.1f}%) win {wcm * 100:4.1f}%'
        print(line)


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summary(list(csv.DictReader(open(sys.argv[2])))); sys.exit()
    out = sys.argv[1]
    its = items(*sys.argv[2:4])
    with Pool(4) as p: res = [r for rs in p.imap_unordered(run, its) for r in rs]
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, list(res[0])); w.writeheader(); w.writerows(res)
    summary([{k: str(v) for k, v in r.items()} for r in res])
