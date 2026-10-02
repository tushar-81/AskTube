FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg      -> audio extraction + yt-dlp merging
# tesseract   -> OCR on video frames
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp needs a JavaScript runtime to solve YouTube's player challenges
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

WORKDIR /app

# Torch first (largest layer, changes least). The cu126 wheel bundles its own CUDA
# libraries, so no nvidia/cuda base image is needed — it falls back to CPU without a GPU.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
