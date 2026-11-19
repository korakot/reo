#!/bin/sh
# Rerun every check behind the README's numbers. Corpora (outside the repo):
#   TSYNC1    TSync-1, the TSync1 folder of TSync1Corpus.zip (github.com/korakot/corpus releases v1.0)
#   LOTUS     LOTUS PD, the LOTUS folder of AIFORTHAI-LotusCorpus.zip (same release)
#   GIGA      a folder with GigaSpeech 2 Thai dev/ (speechcolab/gigaspeech2, gated) and riang_phones.parquet
#             (korakot/vox-gigaspeech2-th-devtest)
#   TVSPEECH  typhoon-ai/TVSpeech data/test-00000-of-00001.parquet
#   CCOST     the C-COST folder of C-COST.zip (Zenodo doi:10.5281/zenodo.17366698), CCOST_metadata.csv beside it
# Results go to $OUT (default out/). About 20 minutes on 4 cores. Needs numpy, pandas, pyarrow, ffmpeg (with
# rubberband and libmp3lame), and sakot (pip install "sakot @ git+https://github.com/korakot/sakot") for TVSpeech.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-out}
mkdir -p "$OUT"
python3 -I validate/check.py tsync1 "$TSYNC1" "$OUT/tsync1.csv"
python3 -I validate/check.py lotus "$LOTUS" "$OUT/lotus.csv"
python3 -I validate/check.py giga "$GIGA" "$OUT/giga.csv"
python3 validate/tvspeech.py "$TVSPEECH" "$OUT/tvspeech.csv"
python3 validate/tempo.py "$OUT/tempo.csv" "$TSYNC1" "$LOTUS" "$TVSPEECH"
python3 -I validate/spans.py lotus "$LOTUS" "$OUT/spans_lotus.csv"
python3 -I validate/spans.py tsync1 "$TSYNC1" "$OUT/spans_tsync1.csv"
python3 -I validate/spans.py giga "$GIGA" "$OUT/spans_giga.csv"
python3 -I validate/spans.py ccost "$CCOST" "$OUT/spans_ccost.csv"
python3 -I validate/robust.py "$OUT/robust.csv" "$LOTUS" "$TVSPEECH"
