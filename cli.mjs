#!/usr/bin/env node
// reo from the command line (Node 18+): speaking rate of WAV files, one line per file and channel.
//   node cli.mjs call.wav [more.wav ...] [--threshold 6] [--span 12.5:40] [--window 10] [--json]
// WAV only (PCM, float, mu-law, A-law); convert other formats first, e.g. ffmpeg -i in.mp3 out.wav,
// or use the Python version (python reo.py), which calls ffmpeg itself.
import fs from 'node:fs';
import path from 'node:path';
import { analyse, readWav, fastShare, span, windows } from './reo.js';

const args = process.argv.slice(2);
const files = [], opt = { threshold: null, json: false, spans: [], window: null };
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--threshold') opt.threshold = parseFloat(args[++i]);
  else if (args[i] === '--json') opt.json = true;
  else if (args[i] === '--span') opt.spans.push(args[++i].split(':').map(Number));
  else if (args[i] === '--window') opt.window = parseFloat(args[++i]);
  else files.push(args[i]);
}
if (!files.length) { console.error('usage: node cli.mjs <file.wav> [...] [--threshold 6] [--span 12.5:40] [--window 10] [--json]'); process.exit(1); }
const out = [];
for (const f of files) {
  let audio;
  try { audio = readWav(fs.readFileSync(f)); } catch (e) { console.error(`${f}: ${e.message}`); continue; }
  audio.channels.forEach((x, c) => {
    const r = analyse(x, audio.sr);
    const row = {
      file: f, channel: c, duration: r.duration, speech_time: r.speech_time, syllables: r.syllables, rate: r.rate,
      median_interval_ms: r.median_interval && r.median_interval * 1000, pause_share: r.pause_share,
      utterance_rate_median: r.utterance_rate_median, utterance_rate_p90: r.utterance_rate_p90,
      ...(opt.threshold ? { fast_share: fastShare(r, opt.threshold) } : {}),
      utterances: r.utterances,
      spans: opt.spans.map(([a, b]) => span(r, a, b)),
      ...(opt.window ? { windows: windows(r, opt.window) } : {}),
    };
    out.push(row);
    if (!opt.json) {
      const fs_ = opt.threshold ? `; ${Math.round((row.fast_share || 0) * 100)}% of speech above ${opt.threshold}` : '';
      console.log(`${path.basename(f)} ch${c}: ${r.rate ? r.rate.toFixed(2) : '-'} syll/s over ${r.speech_time.toFixed(2)} s of speech ` +
        `(${r.syllables} syllables; median interval ${row.median_interval_ms ? Math.round(row.median_interval_ms) : '-'} ms${fs_})`);
      for (const q of [...row.spans, ...(row.windows || [])])
        console.log(`  ${q.start.toFixed(1)}-${q.end.toFixed(1)} s: ${q.rate === null ? '-' : q.rate.toFixed(2)} syll/s (${q.syllables} syllables in ${q.speech.toFixed(2)} s of speech)`);
    }
  });
}
if (opt.json) console.log(JSON.stringify(out, null, 1));
