"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type AnalyzeResult } from "@/lib/api";
import { audioBufferTo16kMono, concatChunks, encodeWav, resampleTo16k, splitWindows } from "@/lib/audio";

export type ShieldStatus = "idle" | "starting" | "listening" | "unavailable" | "error";
export type ShieldSource = "mic" | "clip";

export interface Reading {
  score: number;
  at: number;
}

const MAX_READINGS = 12;

function micErrorMessage(e: unknown): string {
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError" || name === "SecurityError")
    return "Microphone access was blocked. Allow it from the address bar, or try a sample recording instead.";
  if (name === "NotFoundError") return "No microphone was found. Try a sample recording instead.";
  return e instanceof Error ? e.message : "Could not start the microphone.";
}

/**
 * Live Shield engine. Mic audio and clips are both cut into ~2.5 s windows at 16 kHz mono,
 * encoded as WAV and sent to POST /analyze — results always come from the real backend.
 */
export function useShield({ familyId, windowSeconds }: { familyId: number | null; windowSeconds: number }) {
  const [status, setStatus] = useState<ShieldStatus>("idle");
  const [source, setSource] = useState<ShieldSource | null>(null);
  const [sourceLabel, setSourceLabel] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [readings, setReadings] = useState<Reading[]>([]);
  const [quiet, setQuiet] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodesRef = useRef<AudioNode[]>([]);
  const timersRef = useRef<number[]>([]);
  const sessionRef = useRef("");
  const runRef = useRef(0); // bumps on every start/stop so stale responses are ignored
  const inFlightRef = useRef(0);
  const familyRef = useRef(familyId);

  useEffect(() => {
    familyRef.current = familyId;
  }, [familyId]);

  const teardown = useCallback(() => {
    timersRef.current.forEach((t) => window.clearTimeout(t));
    timersRef.current = [];
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    nodesRef.current.forEach((n) => {
      try {
        if (n instanceof AudioScheduledSourceNode) n.stop();
      } catch {}
      n.disconnect();
    });
    nodesRef.current = [];
    void ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
    setAnalyser(null);
  }, []);

  const stop = useCallback(() => {
    runRef.current++;
    teardown();
    setSource(null);
    setStatus((s) => (s === "unavailable" || s === "error" ? s : "idle"));
  }, [teardown]);

  const send = useCallback(
    async (samples: Float32Array, run: number, allowSkip: boolean) => {
      if (allowSkip && inFlightRef.current >= 2) return; // backend is behind; drop this window
      inFlightRef.current++;
      const sentAt = Date.now(); // the window ended now; used to line verdicts up with the audio
      try {
        const res = await api.analyze(encodeWav(samples), sessionRef.current, familyRef.current);
        if (run !== runRef.current) return;
        setResult(res);
        setQuiet(res.silent);
        const score = res.synthetic_likelihood;
        if (!res.silent && score != null) {
          setReadings((r) => [...r, { score, at: sentAt }].slice(-MAX_READINGS));
        }
      } catch (e) {
        if (run !== runRef.current) return;
        runRef.current++;
        teardown();
        setSource(null);
        if (e instanceof ApiError && e.detectionUnavailable) {
          setStatus("unavailable");
        } else {
          setStatus("error");
          setError(e instanceof Error ? e.message : "Analysis failed.");
        }
      } finally {
        inFlightRef.current--;
      }
    },
    [teardown],
  );

  const begin = useCallback(
    (nextSource: ShieldSource, label: string) => {
      runRef.current++;
      teardown();
      sessionRef.current = crypto.randomUUID();
      setResult(null);
      setReadings([]);
      setQuiet(false);
      setError(null);
      setSource(nextSource);
      setSourceLabel(label);
      setStatus("starting");
      return runRef.current;
    },
    [teardown],
  );

  const fail = useCallback(
    (message: string) => {
      runRef.current++;
      teardown();
      setSource(null);
      setStatus("error");
      setError(message);
    },
    [teardown],
  );

  const startMic = useCallback(async () => {
    const run = begin("mic", "Microphone");
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "This browser only allows the microphone on a secure page. Open http://localhost:3000, or try a sample recording.",
        );
      }
      // Raw audio: echo cancellation / noise suppression would alter the very signal we analyze.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      if (run !== runRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      const ctx = new AudioContext();
      ctxRef.current = ctx;
      await ctx.resume();
      await ctx.audioWorklet.addModule("/worklets/capture-processor.js");
      if (run !== runRef.current) return;

      const input = ctx.createMediaStreamSource(stream);
      const meter = ctx.createAnalyser();
      meter.fftSize = 2048;
      const capture = new AudioWorkletNode(ctx, "capture-processor");
      const mute = ctx.createGain();
      mute.gain.value = 0; // keeps the worklet pulled without echoing the mic to the speakers
      input.connect(meter);
      input.connect(capture);
      capture.connect(mute).connect(ctx.destination);
      nodesRef.current = [input, meter, capture, mute];

      const windowSamples = Math.round(windowSeconds * ctx.sampleRate);
      let chunks: Float32Array[] = [];
      let count = 0;
      capture.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        chunks.push(ev.data);
        count += ev.data.length;
        if (count < windowSamples) return;
        const all = concatChunks(chunks);
        const rest = all.slice(windowSamples);
        chunks = rest.length ? [rest] : [];
        count = rest.length;
        void resampleTo16k(all.slice(0, windowSamples), ctx.sampleRate).then((s) => send(s, run, true));
      };

      setAnalyser(meter);
      setStatus("listening");
    } catch (e) {
      if (run === runRef.current) fail(micErrorMessage(e));
    }
  }, [begin, fail, send, windowSeconds]);

  /** Plays a clip out loud while streaming its windows to the backend in real time. */
  const startClip = useCallback(
    async (input: { url?: string; file?: File; label: string }) => {
      const run = begin("clip", input.label);
      try {
        let data: ArrayBuffer;
        if (input.file) {
          data = await input.file.arrayBuffer();
        } else {
          const res = await fetch(input.url!);
          if (!res.ok) throw new Error(`Couldn't load the sample (${res.status}). See README → Demo clips.`);
          data = await res.arrayBuffer();
        }
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        await ctx.resume();
        let buffer: AudioBuffer;
        try {
          buffer = await ctx.decodeAudioData(data);
        } catch {
          throw new Error("We couldn't read that file. Please use a WAV recording.");
        }
        const windows = splitWindows(await audioBufferTo16kMono(buffer), windowSeconds, 1);
        if (!windows.length) throw new Error("That recording is too short. Use at least a second of speech.");
        if (run !== runRef.current) return;

        const player = ctx.createBufferSource();
        player.buffer = buffer;
        const meter = ctx.createAnalyser();
        meter.fftSize = 2048;
        player.connect(meter);
        meter.connect(ctx.destination);
        nodesRef.current = [player, meter];

        const pending: Promise<void>[] = [];
        windows.forEach((w, i) => {
          const at = Math.min((i + 1) * windowSeconds, buffer.duration) * 1000;
          timersRef.current.push(window.setTimeout(() => pending.push(send(w, run, false)), at));
        });
        timersRef.current.push(
          window.setTimeout(async () => {
            await Promise.allSettled(pending);
            if (run !== runRef.current) return;
            runRef.current++;
            teardown();
            setSource(null);
            setStatus("idle");
          }, buffer.duration * 1000 + 50),
        );

        player.start();
        setAnalyser(meter);
        setStatus("listening");
      } catch (e) {
        if (run === runRef.current) fail(e instanceof Error ? e.message : "Couldn't play that recording.");
      }
    },
    [begin, fail, send, teardown, windowSeconds],
  );

  useEffect(
    () => () => {
      runRef.current++;
      teardown();
    },
    [teardown],
  );

  return { status, source, sourceLabel, result, readings, quiet, error, analyser, startMic, startClip, stop };
}
