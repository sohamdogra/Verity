// Forwards raw mic samples (first channel) to the main thread in ~100 ms batches.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.batch = [];
    this.batchLength = 0;
    this.flushAt = Math.round(sampleRate / 10);
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      this.batch.push(new Float32Array(channel));
      this.batchLength += channel.length;
      if (this.batchLength >= this.flushAt) {
        const out = new Float32Array(this.batchLength);
        let offset = 0;
        for (const chunk of this.batch) {
          out.set(chunk, offset);
          offset += chunk.length;
        }
        this.port.postMessage(out, [out.buffer]);
        this.batch = [];
        this.batchLength = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
