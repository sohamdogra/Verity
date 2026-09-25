# Debugging guide

Start here: **http://localhost:8000/health** reports whether the detector is `loading`, `ready`, `unavailable` (with `detection_error`) or `disabled`. The backend terminal also logs every model load attempt.

## Backend

| Symptom | Cause / fix |
|---|---|
| `/health` stuck on `"detection_state": "loading"` | First run is downloading the model (~360 MB). Watch the backend terminal. On slow wifi, pre-download it at home by running the backend once. |
| `"detection_state": "unavailable"` | Read `detection_error`. Usually no internet on first run, a HuggingFace outage, or a broken torch install. Verification still works. Fix: connect once so the model caches, then set `HF_HUB_OFFLINE=1`. |
| Startup is slow even with the model cached | transformers checks HuggingFace for updates. Set `HF_HUB_OFFLINE=1` in `backend/.env` for a fast, offline start. |
| `ModuleNotFoundError: torch` / `torchaudio` | Install torch first: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`. torch and torchaudio versions must come from the same index. |
| torch downloads a 2+ GB CUDA build on a laptop | You ran plain `pip install torch`. Uninstall it and reinstall from the CPU index above. |
| `OSError: cannot load library 'libsndfile'` (Linux) | `sudo apt install libsndfile1`. |
| `/analyze` → 400 "Could not read audio" | The upload isn't WAV/FLAC/OGG. Convert it: `ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav`. (Browser uploads are converted to WAV automatically.) |
| `/analyze` → 503 | Detection unavailable; see above. The UI switches to "Detection unavailable. Verification remains active." |
| Every clip scores ~0 (all green) or ~1 (all red) | The model doesn't generalize to your audio. Compare models on your clips: `python scripts/calibrate.py --models A,B,C your_real.wav your_clone.wav`, then set `DETECTOR_MODELS` in `.env`. See the README table. |
| Real voice drifts into amber | Speakerphone echo/noise pushes scores up. Raise `BAND_GREEN_MAX`/`BAND_RED_MIN` slightly after running `calibrate.py`, or use the built-in clips for the main demo. |
| First window of a clip scores differently from the rest | Normal: it often has silence or breath at the start. Smoothing over 3 windows, plus "red needs ≥2 samples", keeps it from triggering a false alarm. |
| Verification says "Too many attempts" during rehearsal | 5 failures lock it for `VERIFY_LOCK_SECONDS` (300 s). For rehearsals set `VERIFY_LOCK_SECONDS=30`, or delete `backend/verity.db` to reset everything. |
| `database is locked` | Two backend processes share one `verity.db`. Stop the extra `uvicorn`. |
| Port 8000 already in use | `uvicorn main:app --port 8001`, then set `NEXT_PUBLIC_API_URL=http://localhost:8001` in `frontend/.env.local` and restart `npm run dev`. |
| Webhook alert says "webhook_failed" | Check `ALERT_WEBHOOK_URL`, and look for the warning in the backend logs. The alert is still logged in-app, so the demo still works. |
| `--reload` reloads the model on every save | Expected, since the model loads per process. Run without `--reload` for the demo. |

## Frontend

| Symptom | Cause / fix |
|---|---|
| Red banner "Can't reach the Verity server" | Backend isn't running, or it's on a different port/host than `NEXT_PUBLIC_API_URL`. `NEXT_PUBLIC_*` values are baked in at build time, so restart `npm run dev` or rebuild after changing them. |
| CORS error in the browser console | You opened the app on an origin not in `CORS_ORIGINS` (e.g. `http://192.168.x.x:3000`). Add that origin to `CORS_ORIGINS` in `backend/.env`. |
| "Start listening" does nothing or says the mic is blocked | Mic permission was denied. Click the lock icon in the address bar → allow microphone. Mics only work on `localhost` or HTTPS. For a phone demo use an HTTPS tunnel (e.g. `ngrok http 3000`), and point `NEXT_PUBLIC_API_URL` at a tunnel for port 8000 too. |
| Mic works but it always says "It's quiet" | Input level is below `SILENCE_RMS`. Pick the right input device in OS settings, move closer, or lower `SILENCE_RMS` (e.g. `0.002`). |
| Mic picks up the clip playing from the laptop speakers and scores differently | Speaker → room → mic changes the audio a lot. For a reliable demo, use the built-in clip buttons, which analyze the digital audio directly while it plays out loud. Use the live mic as the "and it works live too" moment. |
| Clip buttons disabled | Detection is loading or unavailable. Check `/health`. |
| "Couldn't load the sample (404)" | `frontend/public/demo-clips/real.wav` or `clone.wav` is missing. Regenerate: `cd backend; .\.venv\Scripts\python scripts\make_demo_clips.py` (needs `pip install pyarrow`), or drop in your own. |
| Clip plays silently in some browsers | Autoplay policies. Clips start from a click, so this is rare; click anywhere on the page once and retry. |
| `next build` fails fetching Google Fonts offline | `next/font` downloads Atkinson Hyperlegible and Fraunces at build time. Build once with internet so the fonts are cached, or remove the two font imports in `frontend/app/layout.tsx` to use system fonts. |
| Family disappeared after clearing the browser | The family id lives in `localStorage`; the app falls back to the latest family on the server (`/family/current`). Nothing is lost unless `verity.db` is deleted. |

## Reset everything

```powershell
# stop the backend first
Remove-Item backend\verity.db
```

Then clear the site's local storage in the browser (DevTools → Application → Local Storage), or just re-run setup.

## Useful commands

```powershell
# Backend unit tests (no model needed)
cd backend; .\.venv\Scripts\python -m pytest tests -q

# Score clips window by window like the app does
.\.venv\Scripts\python scripts\calibrate.py

# Hit the API directly
curl.exe -F file=@..\frontend\public\demo-clips\clone.wav -F session_id=dbg http://localhost:8000/analyze
```
