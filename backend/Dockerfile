# HEARSAY inference image: scores every audio file in /data and writes /out/predictions.tsv
#
#   docker build -t verity-hearsay backend
#   docker run --rm -v "/path/to/HackGTHearsayTesting:/data:ro" -v "/path/to/output:/out" verity-hearsay
#
# The container needs no network at run time: the trained model and every front-end it
# uses are baked in. Extra arguments are passed straight to `hearsay.py predict`, e.g.
#   docker run ... verity-hearsay --threads 8 --out /out/team_predictions.tsv
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    NUMBA_CACHE_DIR=/tmp/numba \
    FORENSICS_ASR_DISABLED=1 \
    FORENSICS_WARMUP_DISABLED=1

# ffmpeg decodes M4A/AAC/OPUS/WEBM; libsndfile handles WAV/FLAC/OGG/MP3.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake in only the front-ends the trained model actually refers to. The model bundle has
# to be present first so the downloader can inspect it — the detector ensemble is ~1.8 GB
# and the current model does not use it.
COPY config.py audio_io.py ./
COPY detection/ detection/
COPY forensics/ forensics/
COPY models/ models/
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

COPY . .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "scripts/hearsay.py", "predict", "--data", "/data", "--out", "/out/predictions.tsv"]
