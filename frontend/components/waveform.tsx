"use client";

import { useEffect, useRef } from "react";

/** Live waveform from an AnalyserNode; draws a calm flat line when idle. */
export function Waveform({ analyser, color }: { analyser: AnalyserNode | null; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const g = canvas.getContext("2d");
    if (!g) return;
    let frame = 0;
    const data = analyser ? new Float32Array(analyser.fftSize) : null;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const { clientWidth: w, clientHeight: h } = canvas;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      g.lineWidth = 2.5;
      g.lineJoin = "round";
      g.strokeStyle = color;
      g.beginPath();
      if (analyser && data) {
        analyser.getFloatTimeDomainData(data);
        const step = data.length / w;
        for (let x = 0; x < w; x++) {
          const v = data[Math.floor(x * step)];
          const y = h / 2 + Math.max(-1, Math.min(1, v * 2.2)) * (h / 2 - 4);
          if (x === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        }
        frame = requestAnimationFrame(draw);
      } else {
        g.globalAlpha = 0.35;
        g.moveTo(0, h / 2);
        g.lineTo(w, h / 2);
      }
      g.stroke();
      g.globalAlpha = 1;
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [analyser, color]);

  return <canvas ref={canvasRef} className="h-24 w-full" aria-hidden="true" />;
}
