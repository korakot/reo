# reo (เร็ว)

**How fast is someone speaking?** reo counts the syllables in a recording and divides by the time spent speaking, so you get **syllables per second of speech** (พยางค์ต่อวินาที), for the whole recording and for each utterance. It needs no transcript and no speech recogniser, and it was tested on Thai.

- **Demo:** https://korakot.github.io/reo/ — open a recording or record yourself. The analysis runs in the browser; nothing is uploaded.
- **Live:** https://korakot.github.io/reo/live.html — talk and watch a dial show your speed over your last 4 seconds of speech, turning red past your "too fast" line.
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

**Spans.** To measure part of a recording, such as the stretch where an agent reads a script, analyse the whole file once and ask for the span. Syllables count when their peak falls inside it; speaking time is the speech inside it.

```python
r = reo.analyse_file('call.wav')[0]
reo.span(r, 62.0, 95.5)        # {'start': 62.0, 'end': 95.5, 'speech': 27.1, 'syllables': 141, 'rate': 5.2, 'median_interval': 0.18}
reo.windows(r, 10)             # the same for every 10 s of the recording
```

```js
span(r, 62.0, 95.5).rate;  windows(r, 10);
```

From the command line: `reo --spans spans.csv --span-csv rates.csv`, where `spans.csv` has the columns `file,start,end` and optionally `channel,label`; `reo call.wav --window 10 --windows every10s.csv`; `node cli.mjs call.wav --span 62:95.5 --window 10`. On the demo page, drag across the chart. Measuring a span inside the whole-file analysis, rather than cutting the audio first, keeps the level and noise estimates of the whole recording, so every span of a file is measured on the same footing (the two agree within about 1% for spans of 10 s and more, see Accuracy).

**Stereo call recordings:** most call systems record the agent on one channel and the customer on the other. reo measures each channel on its own, so the agent's speed is not mixed with the customer's. A mono recording of two people gives one rate for both.

## What the number means

reo finds each syllable as a peak of loudness in the 300–3300 Hz band that is voiced, which is the method of de Jong & Wempe (2009) with the band and settings tested for Thai in [korakot/pat](https://github.com/korakot/pat) (`wiki/syllable-detection.md`). Their standard Praat script, on full-band loudness, misses 16% of Thai syllables in running speech; the 300–3300 Hz band misses about 5%. reo accepts a peak as voiced at a slightly lower voicing strength than pat (0.42 against 0.45), because conversation is often weakly voiced (creak, reduced vowels): on the CCOST interviews this took the per-speaker error from −4% to −2%. Going lower helps conversation a little more but lets background music through: at 0.35, ranking the GigaSpeech 2 YouTube videos by speed fell from ρ = 0.87 to 0.63.

Pauses of 120 ms or more are left out of the speaking time, so the rate is the **articulation rate**: how fast the words come while the person is talking, separate from how much they pause (also reported, as the share of pauses). An **utterance** is speech between pauses of half a second or more; it gets a rate when it holds at least one second of speech.

reo also reports the **median time from one syllable to the next**, which is close to 1 / rate and is what you get from syllable timestamps.

**Speed adaptation.** With one fixed amount of smoothing, a detector over-counts slow speech (a long syllable splits in two) and under-counts fast speech (two short syllables merge). reo smooths each utterance (at most 20 s of speech at a time) in proportion to its own syllable length, found by iterating until the rate settles, and uses a 20 ms analysis window (pat uses 32 ms). When no width settles, as with music under speech, that stretch keeps the normal smoothing. On the speed test (Accuracy) this cut the syllables lost at double speed from 9–25% to 1–6%, and the over-count at 0.6× speed from up to 18% to up to 8%.

**The same speech, recorded differently, should get the same number.** Three things make reo's number stable across recordings:
- *Noise.* Each band's steady noise (its 20th-percentile power over the recording) is subtracted before the levels are used. Without it, noise raised the level floor that separates speech from pauses, quiet parts of speech were counted as pauses, and the rate rose 8–15% at 20 dB SNR.
- *Telephone lines.* Speech and pauses are told apart in the 300–3300 Hz band, which a phone line keeps. With the full band, a phone line removed the voice energy below 300 Hz, more of it for men, and put men 4% faster than women at the same true speed.
- *DC offset.* Each analysis frame has its mean removed; an offset otherwise leaked into the low bands and moved the rate by 12–32%.

All thresholds are relative to the recording (percentiles of its own levels), so the recording level does not matter.

**Where to set "too fast"** is a judgement, not something reo knows. Have a few people rate some clips as too fast or fine, run reo on them, and put the line between the two groups. For orientation: people reading sentences aloud measure about 3.5 per second, read news 4.8, interviews and YouTube talk about 5.

## Accuracy

Checked against syllables marked by hand (LOTUS, TSync-1, CCOST), against riang's alignment of human transcripts (GigaSpeech 2), and against transcripts alone (TVSpeech). The true rate counts one syllable per vowel and leaves out pauses of 120 ms or more, like reo. All numbers come from `validate/run_all.sh` on the code in this repository.

### Recordings and speakers

| Speech | Speakers | Units | True rate | Count error | Rate error | Per speaker |
|---|---|---|---|---|---|---|
| Read sentences, two microphones (LOTUS PD) | 12 women, 12 men | 1,680 sentences | 3.46 /s | 0.0% | +2.4% | r = 0.93 |
| Read news, 8 kHz (TSync-1) | 1 woman | 2,600 sentences | 4.75 /s | −2.0% | +1.2% | |
| Interviews (CCOST), interviewee turns inside the whole interview | 30 women, 19 men | 8,893 turns | 4.73 /s | 0.0% | −2.5% per speaker | r = 0.82, ρ = 0.81 |
| YouTube (GigaSpeech 2 dev) | 21 videos | 6,612 segments | 4.99 /s | 0.0% | +5.0%* | r = 0.90, ρ = 0.90 |
| YouTube talk (TVSpeech) | many | 570 clips | 4.94 /s | 0.0% | (count only) | |

Errors are medians. "Per speaker" pools each speaker's recordings and compares their measured rate with their true rate across speakers (r: correlation, ρ: rank correlation), which is what matters when you compare people. The true rates of these speakers are close together (CCOST: 3.5 to 6.0 /s), so r is a strict test.

\* reo's speaking time is shorter than the alignment's, whose phones likely run over short pauses (the transcripts mark none). The syllable count, which is exact in that truth, shows no bias.

Women and men: on LOTUS the per-speaker error is +1.1% for women and +3.8% for men; on CCOST −2.2% and −2.9%.

### Spans

Long recordings with a known truth (each LOTUS speaker's sentences joined, about 3 minutes, and TSync-1 in 50-sentence blocks), measured in windows with `span()` (`validate/spans.py`):

| Window | Rate error (median) | Typical error | 90% of windows within | In file vs cut out |
|---|---|---|---|---|
| 5 s | +0.8 to +2.9% | 4.9–6.5% | 13–20% | 1.6–1.9% |
| 10 s | +0.5 to +3.4% | 3.8–5.5% | 10–16% | 1.2% |
| 30 s | +0.2 to +3.5% | 3.0–4.5% | 7–13% | 0.7% |
| 60 s | +0.4 to +3.6% | 2.3–4.3% | 6–13% | 0.4–0.5% |
| whole file | +0.1 to +2.8% | 1.5–3.5% | 4–10% | |

(Ranges: TSync-1 first, LOTUS second.) A span of a few seconds holds 15–30 syllables, so one syllable more or less is already 3–7%: judge a person on a minute or more of speech, or on many short spans. "In file vs cut out" compares `span()` on the whole-file analysis with cutting the window out and analysing it alone; they agree within about 1% for spans of 10 s and more, so where a span sits in a file barely matters.

CCOST interviewee turns are short (median 1.9 s, about 9 syllables) and surrounded by the interviewer's unlabelled speech: per turn the error is 0% (median) with a typical 10%, and pooled per speaker it is −2.5% (above).

### Fast and slow

**Speed test** (`validate/tempo.py`): the same recordings played faster and slower with pitch kept, so the words, and the true count, stay the same. Syllables found relative to normal speed, and the measured rate in brackets:

| | 0.6× | 0.8× | 1.0× | 1.25× | 1.5× | 1.75× | 2.0× |
|---|---|---|---|---|---|---|---|
| LOTUS women (read) | 1.06 (2.4) | 1.00 (3.1) | 1.00 (3.7) | 1.00 (4.4) | 1.00 (5.1) | 1.00 (6.0) | 0.95 (6.4) |
| LOTUS men (read) | 1.00 (2.4) | 1.00 (3.1) | 1.00 (3.9) | 1.00 (4.7) | 0.98 (5.2) | 0.94 (5.9) | 0.94 (6.0) |
| TSync-1 (woman, news) | 1.08 (3.4) | 1.10 (4.4) | 1.00 (4.8) | 1.09 (6.5) | 1.01 (7.3) | 1.00 (8.5) | 0.99 (9.2) |
| TVSpeech (YouTube) | 1.08 (3.3) | 1.03 (4.2) | 1.00 (4.9) | 0.98 (5.9) | 0.98 (7.0) | 0.95 (7.9) | 0.94 (8.7) |

From 0.6× to 2× speed the count stays within 10% and the measured rate keeps rising, so faster speech always measures faster. On real speech, TVSpeech's count error by true rate is +1.9% under 4.5 /s, 0% from 4.5 to 5.5, and −3.8% from 5.5 to 6.5 /s.

### Same speech, different recording

Each recording changed in one way and measured again (`validate/robust.py`; 8 LOTUS speakers, 30 TVSpeech clips). Change of the whole-file rate, median (worst):

| Change | LOTUS women | LOTUS men | TVSpeech |
|---|---|---|---|
| Level −20 dB | 0.0% | 0.0% | 0.0% |
| Level +12 dB, clipped | −2.2% (−4.1%) | −2.2% (−2.6%) | −0.1% (−1.5%) |
| Resampled to 8 kHz | +0.1% | +0.4% | +0.2% (−3.7%) |
| Resampled to 44.1 kHz | −0.3% | −0.4% | −0.2% (−4.4%) |
| Telephone line (300–3400 Hz, 8 kHz, μ-law) | +0.4% (+1.8%) | +0.6% (+1.0%) | +0.2% (+11.8%) |
| MP3 at 24 kbit/s | −1.4% | −1.8% | −0.2% (−8.3%) |
| White noise, 20 dB SNR | −0.1% | −0.4% | −0.4% (−12.3%) |
| White noise, 10 dB SNR | −11.8% | −28.0% | −11.7% (−52.4%) |
| Mains hum (50 Hz, −20 dB) | +0.4% | +0.5% | −1.0% (−12.5%) |
| DC offset | 0.0% | 0.0% | 0.0% (−1.5%) |

So level, sample rate, file format, a telephone line, moderate noise, hum and DC offset change the number by about 2% or less, alike for women and men. Heavy noise (10 dB SNR) lowers it a lot, more for men.

### Limits

- **Real call-centre audio is untested.** Telephone lines are covered (simulated above, and TSync-1 is 8 kHz), but line noise, hold music, beeps and crosstalk are not. Music and loud noise add false syllables.
- **Recordings whose loudness swings by more than about 25 dB** (whispering next to normal speech, as in a TV drama) lose the quiet parts, because syllables far below the recording's loudest are taken for noise. Measure such recordings utterance by utterance. In GigaSpeech 2, one drama video measured 3.1 /s as one file and 5.6 /s segment by segment, against a true 4.9 /s; the other 20 videos agree within about 1% either way.
- **Read speech by men** measures about 2 points faster than by women on LOTUS (+3.8% against +1.1% per speaker); interviews do not show it (−2.9% and −2.2%).
- A mono recording of two people gives one rate for both. Children's speech and other languages are untested.

To rerun everything: `validate/run_all.sh` (it says where each corpus comes from). `test/parity.py` checks that Python and JS agree and `test/spans.py` that spans add up. No audio is kept in this repository.

## Licence

MIT (`LICENSE`). The code only: no audio, corpus data or trained model is part of this repository. There are no trained weights. The detector's few settings were chosen by testing against the corpora listed under Accuracy, which are used only for evaluation and are not part of this repository.

## Credit

The detector comes from [korakot/pat](https://github.com/korakot/pat), after de Jong, N. H. & Wempe, T. (2009), *Praat script to detect syllable nuclei and measure speech rate automatically*, Behavior Research Methods 41:385. Test data: LOTUS and TSync-1 (NECTEC), GigaSpeech 2 (SpeechColab, with riang's alignment from korakot/vox), TVSpeech (Typhoon, SCB 10X).
