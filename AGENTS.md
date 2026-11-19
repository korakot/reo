# reo

reo measures speaking rate (syllables per second of speech) from audio alone. It is deliberately narrow: one number, per recording and per utterance, so it is easy to hand over and to check. Read `README.md` first; it holds the method and every accuracy result.

Two implementations, one behaviour. `reo.py` (numpy) and `reo.js` (no dependencies) must give the same syllables, times and rates on the same audio, so a batch run in Python matches what the demo page shows. Any change to one goes into the other in the same commit, and `python3 test/parity.py <wav files>` must pass (`REO_VARIANTS=1 python3 test/parity.py one.wav` also checks 8–48 kHz, μ-law, A-law, 24-bit and stereo).

A change to the detector is judged on the checks in `validate/`, not on one file: the hand-marked corpora (`check.py`), real fast speech (`tvspeech.py`, GigaSpeech 2 in `check.py`), and the speed test (`tempo.py`), where the count should stay the same from 0.6× to 2× speed. Update the README tables when the numbers move. Corpora stay outside the repo; no audio is committed.

The page (`index.html`, `app.js`, `worker.js`) is served by GitHub Pages from `main`, so a push to `main` is a deploy. Test it in headless Chromium before pushing.

Merge each finished step to `main` right away (as in `korakot/memory`): other sessions clone `main`.

Origin: the detector is korakot/pat's (`wiki/syllable-detection.md`); reo adds the 2 dB dip, the 20 ms window and the speed adaptation. Started 19 Nov 2026 for a coworker who needs to tell whether call-centre agents speak too fast.
