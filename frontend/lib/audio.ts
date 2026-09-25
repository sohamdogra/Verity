/** Browser audio helpers: resample to 16 kHz mono, cut windows, encode WAV. */

export const TARGET_RATE = 16_000;

/** Mix an AudioBuffer down to mono and resample to 16 kHz with an OfflineAudioContext. */
export async function audioBufferTo16kMono(buffer: AudioBuffer): Promise<Float32Array> {
  const length = Math.max(1, Math.ceil(buffer.duration * TARGET_RATE));
  const offline = new OfflineAudioContext(1, length, TARGET_RATE);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination); // multi-channel is down-mixed to the mono destination
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

export async function resampleTo16k(samples: Float32Array, fromRate: number): Promise<Float32Array> {
  if (fromRate === TARGET_RATE) return samples;
  const buffer = new AudioBuffer({ length: samples.length, sampleRate: fromRate, numberOfChannels: 1 });
  buffer.copyToChannel(new Float32Array(samples), 0);
  return audioBufferTo16kMono(buffer);
}

/** Split into fixed windows; a trailing piece shorter than minSeconds is dropped. */
export function splitWindows(samples: Float32Array, windowSeconds: number, minSeconds = 1): Float32Array[] {
  const size = Math.round(windowSeconds * TARGET_RATE);
  const out: Float32Array[] = [];
  for (let i = 0; i < samples.length; i += size) {
    const piece = samples.subarray(i, i + size);
    if (piece.length >= minSeconds * TARGET_RATE) out.push(piece);
  }
  return out;
}

export function concatChunks(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

/** 16-bit PCM mono WAV. */
export function encodeWav(samples: Float32Array, sampleRate = TARGET_RATE): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
