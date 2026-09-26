# HEARSAY inference image: scores every audio file in /data and writes /out/predictions.tsv
#
#   docker build -t verity-hearsay backend
#   docker run --rm -v "<test folder>:/data:ro" -v "<output folder>:/out" verity-hearsay
#
# Extra arguments are passed to `hearsay.py predict`, e.g. `--threads 8` or `--out /out/team_predictions.tsv`.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    NUMBA_CACHE_DIR=/tmp/numba \
    FORENSICS_ASR_DISABLED=1 \
    FORENSICS_WARMUP_DISABLED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake the pretrained detectors + WavLM into the image so inference needs no internet.
COPY config.py ./
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

COPY . .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "scripts/hearsay.py", "predict", "--data", "/data", "--out", "/out/predictions.tsv"]
