// JS side of the parity test: analyse each file, every channel, and print the results as JSON.
// Usage: node test/parity.mjs <wav> [<wav> ...]   (run by test/parity.py)
import fs from 'node:fs';
import { analyse, readWav } from '../reo.js';
const out = {};
for (const f of process.argv.slice(2)) {
  const { channels, sr } = readWav(fs.readFileSync(f));
  out[f] = channels.map((x) => {
    const r = analyse(x, sr);
    return { syllables: r.syllables, peaks: r.peaks, speech_time: r.speech_time, rate: r.rate, utterances: r.utterances.length };
  });
}
console.log(JSON.stringify(out));
