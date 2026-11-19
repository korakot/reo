// Runs reo's analysis off the main thread, so the page stays responsive on long recordings.
import { analyse } from './reo.js';

onmessage = (e) => {
  const { id, channel, x, sr } = e.data;
  try {
    const r = analyse(x, sr, {}, (f) => postMessage({ id, channel, progress: f }));
    postMessage({ id, channel, result: r });
  } catch (err) {
    postMessage({ id, channel, error: String(err && err.message || err) });
  }
};
