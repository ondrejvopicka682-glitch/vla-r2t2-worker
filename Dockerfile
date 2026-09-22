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

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

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

COPY handler.py .

CMD ["python3.11", "-u", "handler.py"]
