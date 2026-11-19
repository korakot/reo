"""Span properties that must hold on any recording (Python; test/parity.py checks JS gives the same spans):

- a span over the whole recording equals the whole-file result;
- windows of any size split the syllables and the speaking time exactly (their sums equal the totals);
- two adjacent spans add up to the span that covers both.

    python3 test/spans.py <wav> [<wav> ...]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reo


def check(path):
    bad = []
    for r in reo.analyse_file(path):
        whole = reo.span(r, 0, r['duration'] + 1)
        if whole['syllables'] != r['syllables'] or abs(whole['speech'] - r['speech_time']) > 1e-9:
            bad.append('whole span != whole file')
        for size in (1, 2.5, 10, 30):
            w = reo.windows(r, size)
            if sum(q['syllables'] for q in w) != r['syllables']: bad.append(f'{size} s windows: syllables do not add up')
            if abs(sum(q['speech'] for q in w) - r['speech_time']) > 1e-6: bad.append(f'{size} s windows: speech does not add up')
        d = r['duration']
        for m in (d * 0.3, d * 0.5, d * 0.71):
            a, b, ab = reo.span(r, 0, m), reo.span(r, m, d + 1), reo.span(r, 0, d + 1)
            if a['syllables'] + b['syllables'] != ab['syllables'] or abs(a['speech'] + b['speech'] - ab['speech']) > 1e-9:
                bad.append(f'split at {m:.2f} s does not add up')
    print(('OK  ' if not bad else 'FAIL') + f' {os.path.basename(path)}' + ('' if not bad else ': ' + '; '.join(sorted(set(bad)))))
    return not bad


if __name__ == '__main__':
    ok = [check(p) for p in sys.argv[1:]]
    print(f'{sum(ok)} of {len(ok)} files pass')
    sys.exit(0 if all(ok) else 1)
