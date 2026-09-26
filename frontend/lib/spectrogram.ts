/** Minimal in-browser spectrogram: radix-2 FFT + a warm colormap that matches the site palette. */

function fftInPlace(re: Float32Array, im: Float32Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const next = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = next;
      }
    }
  }
}

export interface Spectrogram {
  frames: number;
  bins: number;
  db: Float32Array; // frames x bins, row-major by frame
  seconds: number;
}

export function computeSpectrogram(samples: Float32Array, sampleRate: number, fftSize = 512, hop = 160): Spectrogram {
  const bins = fftSize / 2;
  const frames = Math.max(1, Math.floor((samples.length - fftSize) / hop) + 1);
  const db = new Float32Array(frames * bins);
  const window = new Float32Array(fftSize).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (fftSize - 1)));
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  for (let f = 0; f < frames; f++) {
    const offset = f * hop;
    for (let i = 0; i < fftSize; i++) {
      re[i] = (samples[offset + i] ?? 0) * window[i];
      im[i] = 0;
    }
    fftInPlace(re, im);
    for (let b = 0; b < bins; b++) {
      db[f * bins + b] = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-10);
    }
  }
  return { frames, bins, db, seconds: samples.length / sampleRate };
}

/** Cream -> teal -> deep green: readable on the warm background, never neon. */
const STOPS: [number, [number, number, number]][] = [
  [0.0, [250, 246, 239]],
  [0.35, [191, 214, 201]],
  [0.6, [92, 158, 138]],
  [0.82, [47, 107, 94]],
  [1.0, [28, 52, 46]],
];

export const COLORMAP: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const k = STOPS.findIndex(([s]) => s >= t);
    const [s1, c1] = STOPS[Math.max(0, k - 1)];
    const [s2, c2] = STOPS[Math.max(0, k)];
    const u = s2 === s1 ? 0 : (t - s1) / (s2 - s1);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = c1[c] + (c2[c] - c1[c]) * u;
  }
  return lut;
})();

/** Paint into a canvas sized frames x bins (low frequencies at the bottom), CSS scales it. */
export function paintSpectrogram(canvas: HTMLCanvasElement, spec: Spectrogram, dynamicRangeDb = 70) {
  canvas.width = spec.frames;
  canvas.height = spec.bins;
  const g = canvas.getContext("2d");
  if (!g) return;
  let max = -Infinity;
  for (const v of spec.db) if (v > max) max = v;
  const floor = max - dynamicRangeDb;
  const img = g.createImageData(spec.frames, spec.bins);
  for (let f = 0; f < spec.frames; f++) {
    for (let b = 0; b < spec.bins; b++) {
      const v = Math.max(0, Math.min(1, (spec.db[f * spec.bins + b] - floor) / dynamicRangeDb));
      const idx = Math.round(v * 255) * 3;
      const p = ((spec.bins - 1 - b) * spec.frames + f) * 4;
      img.data[p] = COLORMAP[idx];
      img.data[p + 1] = COLORMAP[idx + 1];
      img.data[p + 2] = COLORMAP[idx + 2];
      img.data[p + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
}
