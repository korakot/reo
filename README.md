# reo (เร็ว)

**How fast is someone speaking?** reo counts the syllables in a recording and divides by the time spent speaking, so you get **syllables per second of speech** (พยางค์ต่อวินาที), for the whole recording and for each utterance. It needs no transcript and no speech recogniser, and it was tested on Thai.

- **Demo:** https://korakot.github.io/reo/ — open a recording or record yourself. The analysis runs in the browser; nothing is uploaded.
- **Python** (`reo.py`, needs only numpy) for batches of files.
- **JavaScript** (`reo.js`, no dependencies) for the browser and Node.

The two give the same numbers on the same audio (`test/parity.py` checks syllable for syllable).

## Use it

**Python**, on a folder of call recordings:

```bash
pip install "reo @ git+https://github.com/korakot/reo"
reo calls/*.wav --csv files.csv --utterances utterances.csv --threshold 6
```

or without installing, `python reo.py calls/*.wav ...`. `files.csv` has one row per file and channel: the rate, the median time from one syllable to the next, the speaking time, the share of pauses, the median and 90th-percentile utterance rate, and, with `--threshold`, the share of speech in utterances faster than that. `utterances.csv` has one row per utterance, so you can find the fast stretches. WAV files are read directly, including the μ-law and A-law encodings that phone systems use; other formats go through ffmpeg if it is installed.

```python
import reo
left, right = reo.analyse_file('call.wav')        # one result per channel
left['rate'], left['utterances'][0]               # 4.8, {'start': 0.4, 'end': 3.1, 'speech': 2.5, 'syllables': 12, 'rate': 4.8}
```

**JavaScript:**

```js
import { analyse, readWav } from '@korakot/reo';   // or './reo.js'
const { channels, sr } = readWav(arrayBuffer);
const r = analyse(channels[0], sr);                 // r.rate, r.utterances, r.peaks (syllable times)
```

`node cli.mjs call.wav --threshold 6` does the same from the command line for WAV files.

**Stereo call recordings:** most call systems record the agent on one channel and the customer on the other. reo measures each channel on its own, so the agent's speed is not mixed with the customer's. A mono recording of two people gives one rate for both.

## What the number means

reo finds each syllable as a peak of loudness in the 300–3300 Hz band that is voiced, which is the method of de Jong & Wempe (2009) with the band and settings tested for Thai in [korakot/pat](https://github.com/korakot/pat) (`wiki/syllable-detection.md`). Their standard Praat script, on full-band loudness, misses 16% of Thai syllables in running speech; the 300–3300 Hz band misses about 5%.

Pauses of 120 ms or more are left out of the speaking time, so the rate is the **articulation rate**: how fast the words come while the person is talking, separate from how much they pause (also reported, as the share of pauses). An **utterance** is speech between pauses of half a second or more; it gets a rate when it holds at least one second of speech.

reo also reports the **median time from one syllable to the next**, which is close to 1 / rate and is what you get from syllable timestamps.

**Speed adaptation.** With one fixed amount of smoothing, a detector over-counts slow speech (a long syllable splits in two) and under-counts fast speech (two short syllables merge). reo smooths each utterance in proportion to its own syllable length, found by iterating until the rate settles, and uses a 20 ms analysis window (pat uses 32 ms). On the speed test below this cut the syllables lost at double speed from 9–25% to 4–10%, and the over-count at 0.6× speed from up to 18% to up to 9%.

**Where to set "too fast"** is a judgement, not something reo knows. Have a few people rate some clips as too fast or fine, run reo on them, and put the line between the two groups. For orientation: people reading sentences aloud measure about 3.5 per second, read news 4.8, YouTube talk about 5.

## Accuracy

Checked against syllables marked by hand (LOTUS, TSync-1), against riang's alignment of human transcripts (GigaSpeech 2), and against transcripts alone (TVSpeech). The true rate counts one syllable per vowel and leaves out pauses of 120 ms or more, like reo.

| Speech | Speakers | Utterances | True rate | Count error | Rate error | Per speaker |
|---|---|---|---|---|---|---|
| Read sentences, close and far microphone (LOTUS PD) | 12 women, 12 men | 1,680 | 3.46 /s | 0.0% | +0.1% | r = 0.95 (women 0.84, men 0.96) |
| Read news, 8 kHz (TSync-1) | 1 woman | 2,600 | 4.75 /s | −2.2% | +0.6% | |
| YouTube (GigaSpeech 2 dev) | 21 videos | 6,604 | 4.99 /s | 0.0% | +4.9%* | rank ρ = 0.93 |
| YouTube talk (TVSpeech) | many | 570 | 5.09 /s | −1.3% | (count only) | |

\* reo's speaking time is shorter than the alignment's, whose phones likely run over short pauses (the transcripts mark none). The syllable count, which is exact in that truth, shows no bias.

The errors are medians over utterances. A single utterance of a few seconds is off by about 4–10% (median absolute error), so judge a person on many utterances, not one.

**Fast and slow.** By true speed, on real YouTube speech (GigaSpeech 2):

| True rate | Utterances | Count error | Measured rate |
|---|---|---|---|
| under 3 /s | 127 | 0% | 3.5 |
| 3–4 | 660 | 0% | 4.4 |
| 4–5 | 2,522 | 0% | 5.0 |
| 5–6 | 2,314 | 0% | 5.6 |
| 6–7 | 776 | −5% | 6.1 |
| over 7 | 205 | −12% | 6.5 |

So reo is exact up to about 6 syllables per second and then starts to merge syllables: very fast speech reads somewhat slower than it is, but still faster than everything slower. Part of the drop is the binning itself (an utterance whose true rate is over-estimated lands in the top bin), which is why the speed test is the cleaner check.

**Speed test** (`validate/tempo.py`): the same recordings played faster and slower with pitch kept, so the words, and the true count, stay the same. Syllables found relative to normal speed, and the measured rate in brackets:

| | 0.6× | 0.8× | 1.0× | 1.25× | 1.5× | 1.75× | 2.0× |
|---|---|---|---|---|---|---|---|
| LOTUS women (read) | 1.06 (2.4) | 1.01 (3.0) | 1.00 (3.6) | 1.00 (4.4) | 0.98 (4.9) | 1.00 (5.7) | 0.93 (6.4) |
| LOTUS men (read) | 1.00 (2.3) | 1.00 (3.1) | 1.00 (3.7) | 1.00 (4.3) | 0.95 (5.4) | 0.94 (6.1) | 0.96 (6.4) |
| TSync-1 (woman, news) | 1.09 (3.3) | 1.10 (4.3) | 1.00 (4.9) | 1.07 (6.4) | 1.01 (7.1) | 1.00 (8.0) | 0.94 (8.9) |
| TVSpeech (YouTube) | 1.07 (3.3) | 1.02 (4.2) | 1.00 (5.0) | 0.97 (6.0) | 0.95 (7.0) | 0.92 (7.9) | 0.90 (8.7) |

**Limits.** Untested on real call-centre audio: telephone bandwidth is covered (TSync-1 is 8 kHz and the band stops at 3300 Hz), but line noise, hold music and crosstalk are not. Music and loud noise add false syllables. Children's speech and other languages are untested.

To rerun the checks: `validate/check.py` (TSync-1, LOTUS, GigaSpeech 2), `validate/tvspeech.py`, `validate/tempo.py`. Each says where its corpus comes from. No audio is kept in this repository.

## Credit

The detector comes from [korakot/pat](https://github.com/korakot/pat), after de Jong, N. H. & Wempe, T. (2009), *Praat script to detect syllable nuclei and measure speech rate automatically*, Behavior Research Methods 41:385. Test data: LOTUS and TSync-1 (NECTEC), GigaSpeech 2 (SpeechColab, with riang's alignment from korakot/vox), TVSpeech (Typhoon, SCB 10X).
