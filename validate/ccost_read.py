"""CCOST interviews (Pittayaporn et al., Zenodo doi:10.5281/zenodo.17366698): real conversation, 49 speakers.

Only the interviewee is transcribed. The interviewer's turns, between the labelled ones, are in the same audio
but unmarked, so a check on CCOST measures spans: each interviewee turn (tier TRN), measured inside the whole
interview. The truth is the hand-corrected syllable tier (MAS) and phone tier (MAU).
"""
import csv, glob, os, re

PAUSE = re.compile(r'^(<.*>|\[.*\]|)$')


def read_textgrid(path):
    t = open(path, encoding='utf-8').read()
    tiers = {}
    for tr in re.split(r'\n\s*item \[\d+\]:', t)[1:]:
        name = re.search(r'name = "(.*)"', tr).group(1)
        tiers[name] = [(float(s), float(e), x.strip()) for s, e, x in
                       re.findall(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*)"', tr)]
    return tiers


def interviews(root):
    meta = {}
    mp = os.path.join(os.path.dirname(root.rstrip('/')), 'CCOST_metadata.csv')
    if os.path.exists(mp):
        for r in csv.DictReader(open(mp)): meta[r['bundle']] = r
    for wav in sorted(glob.glob(os.path.join(root, '*', '*_INT_*.wav'))):
        b = os.path.basename(wav)[:-4]
        tg = wav[:-4] + '.TextGrid'
        if not os.path.exists(tg): continue
        T = read_textgrid(tg)
        if not {'TRN', 'MAS', 'MAU'} <= set(T): continue
        mids = [(s + e) / 2 for s, e, x in T['MAS'] if not PAUSE.match(x)]
        phones = [(s, e) for s, e, x in T['MAU'] if not PAUSE.match(x)]
        turns = [(s, e) for s, e, x in T['TRN'] if x and not PAUSE.match(x)]
        m = meta.get(b, {})
        yield dict(id=b, speaker='ccost/' + m.get('Speaker', b.split('_')[1]), sex=m.get('Sex', '')[:1],
                   age=m.get('Age_Group', ''), wav=wav, mids=mids, phones=phones, turns=turns)
