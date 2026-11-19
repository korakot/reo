"""Parity test: reo.py and reo.js must give the same syllables, times and rates on the same files,
including span() and windows().

    python3 test/parity.py [<wav> ...]

With no files it makes test signals with ffmpeg (8 and 16 kHz, stereo, mu-law, A-law, 24-bit) from a source
WAV given in REO_TEST_WAV, or from TSync-1/LOTUS files given as arguments.
"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reo

HERE = os.path.dirname(os.path.abspath(__file__))


def variants(src, d):
    """Encodings call recordings come in, made from one source file."""
    out = []
    for name, args in [('s16_8k', ['-ar', '8000']), ('s16_16k', ['-ar', '16000']),
                       ('ulaw_8k', ['-ar', '8000', '-acodec', 'pcm_mulaw']), ('alaw_8k', ['-ar', '8000', '-acodec', 'pcm_alaw']),
                       ('s24_44k', ['-ar', '44100', '-acodec', 'pcm_s24le']), ('f32_48k', ['-ar', '48000', '-acodec', 'pcm_f32le'])]:
        p = os.path.join(d, f'{name}.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src] + args + [p], check=True)
        out.append(p)
    p = os.path.join(d, 'stereo_8k.wav')  # left: the source, right: the source delayed and quieter
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-filter_complex',
                    '[0:a]asplit[a][b];[b]adelay=700,volume=0.4[c];[a][c]amerge=inputs=2', '-ar', '8000', p], check=True)
    out.append(p)
    return out


def main(files):
    with tempfile.TemporaryDirectory() as d:
        if not files or all(f.endswith('.wav') and len(files) == 1 for f in files) and os.environ.get('REO_VARIANTS'):
            files = variants(files[0] if files else os.environ['REO_TEST_WAV'], d)
        js = json.loads(subprocess.run(['node', os.path.join(HERE, 'parity.mjs')] + files, capture_output=True, check=True, text=True).stdout)
        bad = 0
        for f in files:
            for c, r in enumerate(reo.analyse_file(f)):
                j = js[f][c]
                pw = [[w['syllables'], w['speech'], w['rate'], w['median_interval']] for w in reo.windows(r, 5)]
                ps = [[q['syllables'], q['speech'], q['rate']] for q in (reo.span(r, a, b) for a, b in [(0.3, 2.1), (1.0, 7.7), (2.5, 30)])]
                close = lambda u, v: all((x is None and y is None) or (x is not None and y is not None and abs(x - y) < 1e-9)
                                         for x, y in zip(u, v)) and len(u) == len(v)
                same = (r['syllables'] == j['syllables'] and len(r['utterances']) == j['utterances']
                        and all(abs(a - b) < 1e-9 for a, b in zip(r['peaks'], j['peaks']))
                        and abs((r['rate'] or 0) - (j['rate'] or 0)) < 1e-9
                        and len(pw) == len(j['windows']) and all(close(u, v) for u, v in zip(pw, j['windows']))
                        and all(close(u, v) for u, v in zip(ps, j['spans'])))
                bad += not same
                print(f"{'OK  ' if same else 'DIFF'} {os.path.basename(f)} ch{c}: py {r['syllables']} syl {r['rate'] or 0:.4f}/s, "
                      f"js {j['syllables']} syl {j['rate'] or 0:.4f}/s")
        print(f'{len(files)} files, {bad} differ')
        return bad


if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1:]) else 0)
