/** Built-in demo clips live in public/demo-clips/. Replace the WAVs to use your own —
 *  see README "Demo clips". They are analyzed by the real backend like any other audio. */
export const DEMO_CLIPS = [
  {
    id: "real",
    title: "Real voice",
    description: "A genuine human recording",
    url: "/demo-clips/real.wav",
  },
  {
    id: "clone",
    title: "AI voice clone",
    description: "The same voice, cloned, asking for money",
    url: "/demo-clips/clone.wav",
  },
  {
    id: "tts-legacy",
    title: "Robotic text-to-speech",
    description: "An older synthetic voice",
    url: "/demo-clips/tts-legacy.wav",
  },
] as const;
