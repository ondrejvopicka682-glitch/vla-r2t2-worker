# RunPod serverless worker: netease-youdao/Confucius4-R2T2 (CUDA).
# Model weights are NOT baked into the image -- they're downloaded once per
# cold start into the container's own ephemeral disk (see handler.py); no
# network volume is used, so idle cost is zero.
#
# torch/torchvision/torchaudio come ONLY from this base image and must not
# be reinstalled by pip (see requirements.txt for why). This base image tag
# is the actual version pin for the whole CUDA/torch stack.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ffmpeg is the actual audio container/codec decoder (see audio_decode.py).
# libsndfile (used by soundfile/librosa directly) cannot decode M4A/AAC at
# all, which is what originally failed with:
# soundfile.LibsndfileError / audioread.exceptions.NoBackendError.
# This is a system package via apt, unrelated to the torch/torchvision pip
# stack above -- installing it does not touch that stack.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --ignore-installed is NOT scoped to a single package in pip -- it applies
# to the whole install command. Run it in its own invocation, just for the
# distutils-installed "blinker" conflict, so the requirements.txt install
# that follows leaves the base image's already-installed torch/torchvision
# (cu124-matched) untouched instead of silently reinstalling a mismatched
# torch build from PyPI's default index. This is the actual fix for:
# RuntimeError: operator torchvision::nms does not exist
RUN pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt

# Build-time smoke check: fails the image build (and therefore the deploy)
# if torch/torchvision no longer import cleanly together, or if qwen_asr's
# model class isn't importable. This only validates imports -- it cannot
# check CUDA runtime behavior, since GitHub Actions build runners have no
# GPU. GPU/model-load correctness is still only provable by a real RunPod
# job.
RUN python3.11 -c "\
import torch; print('torch:', torch.__version__, '| built for cuda:', torch.version.cuda); \
import torchvision; print('torchvision:', torchvision.__version__); \
from torchvision.ops import nms; \
boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]]); \
scores = torch.tensor([0.9, 0.8]); \
kept = nms(boxes, scores, 0.5); \
print('torchvision::nms operator call: OK, kept', kept.tolist()); \
from qwen_asr.inference.qwen3_asr import Qwen3ASRModel; print('qwen_asr Qwen3ASRModel import: OK (same import path as handler.py)'); \
print('SMOKE CHECK PASSED')"

COPY audio_decode.py .

# Build-time audio decoding smoke check: exercises the EXACT same
# decode_audio_to_wav() helper handler.py calls, on an M4A/AAC file (the
# format that originally failed), without needing the model weights or a
# GPU. Fails the build if ffmpeg is missing, if M4A/AAC decode breaks, or
# if the resulting WAV isn't loadable at 16kHz with real samples in it.
RUN ffmpeg -y -v error -f lavfi -i "sine=frequency=440:duration=1" -ar 16000 -ac 1 -c:a aac /tmp/synth_test.m4a && \
    python3.11 -c "\
with open('/tmp/synth_test.m4a', 'rb') as f: \
    audio_bytes = f.read(); \
from audio_decode import decode_audio_to_wav, cleanup; \
src, wav = decode_audio_to_wav(audio_bytes, filename='synth_test.m4a'); \
import os; \
assert os.path.exists(wav), 'normalized WAV was not created'; \
import librosa; \
data, sr = librosa.load(wav, sr=16000, mono=True); \
assert sr == 16000, f'unexpected sample rate: {sr}'; \
assert len(data) > 0, 'decoded audio contains no samples'; \
print('AUDIO DECODE SMOKE TEST: ffmpeg install OK, M4A/AAC decode OK, normalized WAV OK, librosa load OK, samples=', len(data), 'sr=', sr); \
cleanup(src, wav); \
print('AUDIO DECODE SMOKE TEST PASSED')" && \
    rm -f /tmp/synth_test.m4a

COPY handler.py .

CMD ["python3.11", "-u", "handler.py"]
