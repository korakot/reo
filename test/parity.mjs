// JS side of the parity test: analyse each file, every channel, and print the results as JSON.
// Usage: node test/parity.mjs <wav> [<wav> ...]   (run by test/parity.py)
import fs from 'node:fs';
import { analyse, readWav, span, windows } from '../reo.js';
const out = {};
for (const f of process.argv.slice(2)) {
  const { channels, sr } = readWav(fs.readFileSync(f));
  out[f] = channels.map((x) => {
    const r = analyse(x, sr);
    return { syllables: r.syllables, peaks: r.peaks, speech_time: r.speech_time, rate: r.rate, utterances: r.utterances.length,
      windows: windows(r, 5).map((w) => [w.syllables, w.speech, w.rate, w.median_interval]),
      spans: [[0.3, 2.1], [1.0, 7.7], [2.5, 30]].map(([a, b]) => { const q = span(r, a, b); return [q.syllables, q.speech, q.rate]; }) };
  });
}
console.log(JSON.stringify(out));
