"""
RunPod serverless handler for netease-youdao/Confucius4-R2T2.

Design:
- Uses the `qwen_asr` package's Transformers backend on CUDA (NOT vLLM). This
  is a deliberate first-deployment choice: the Transformers path is the
  officially documented, simpler, more predictable route, and the project's
  strict-testing rule ("one real test, fix concrete bugs only, no
  benchmarking loop") makes a fragile multi-service vLLM setup the wrong
  choice for a first cloud deployment. vLLM can be swapped in later behind
  the same handler contract if throughput ever requires it.
- No network volume is attached (see MODEL_DIR below for why); the model is
  cached on the container's own ephemeral disk, so it survives multiple jobs
  on the SAME warm worker but is re-downloaded on every cold start. This is
  a deliberate cost trade-off, not an oversight -- see handler.py comments.
- Input audio arrives as base64 (appropriate for the short lecture clips this
  app sends; a presigned-URL path can be added later without changing the
  response contract).
- Never fabricates fields R2T2 doesn't produce (e.g. no invented timestamps).
"""
import base64
import os
import tempfile
import time
import traceback

import runpod

MODEL_REPO = "netease-youdao/Confucius4-R2T2"
# No persistent network volume is attached (deliberate cost trade-off: zero
# ongoing storage billing when idle, at the cost of re-downloading the ~4GB
# checkpoint into the container's ephemeral disk on every cold start). If a
# network volume is reattached later, point MODEL_DIR at /runpod-volume/...
# instead and the rest of this file needs no changes.
MODEL_DIR = "/app/model_cache/Confucius4-R2T2"

_model = None
_load_time_seconds = None
_device_name = None


def _ensure_model_downloaded():
    """Downloads the checkpoint into the container's ephemeral disk once per
    worker lifetime (not once globally -- see module docstring). A
    marker file (not just directory existence) guards against a partially
    completed prior download being treated as valid."""
    marker = os.path.join(MODEL_DIR, ".download_complete")
    if os.path.exists(marker):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
    with open(marker, "w") as f:
        f.write("ok")


def _load_model():
    global _model, _load_time_seconds, _device_name
    if _model is not None:
        return

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available on this worker -- Confucius4-R2T2 requires an NVIDIA GPU."
        )

    _ensure_model_downloaded()

    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    t0 = time.time()
    _model = Qwen3ASRModel.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    _load_time_seconds = time.time() - t0
    _device_name = torch.cuda.get_device_name(0)


def _gpu_memory_mb():
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:
        return None


def handler(job):
    job_input = job.get("input", {}) or {}
    audio_b64 = job_input.get("audio_base64")
    language = job_input.get("language")  # None/omitted -> Auto
    context_terms = job_input.get("context") or []

    if not audio_b64:
        return {"status": "failed", "error": "missing 'audio_base64' in job input"}

    try:
        _load_model()
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "error": f"model load failed: {type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=6),
        }

    tmp_path = None
    try:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        import librosa
        wav, _sr = librosa.load(tmp_path, sr=16000, mono=True)
        audio_duration = float(len(wav)) / 16000.0

        context_str = ", ".join(context_terms) if context_terms else ""
        kwargs = {}
        if context_str:
            kwargs["context"] = [context_str]

        t0 = time.time()
        results = _model.transcribe(
            audio=[(wav, 16000)],
            language=[language] if language else None,
            **kwargs,
        )
        processing_seconds = time.time() - t0

        text = results[0].text if results else ""
        detected_language = getattr(results[0], "language", None) if results else None

        return {
            "status": "completed",
            "model": MODEL_REPO,
            "language": detected_language,
            "text": text,
            "segments": [],  # R2T2's one-shot transcribe does not return reliable segment timing
            "audio_duration_seconds": audio_duration,
            "processing_seconds": processing_seconds,
            "rtf": (processing_seconds / audio_duration) if audio_duration > 0 else None,
            "gpu": _device_name,
            "gpu_memory_peak_mb": _gpu_memory_mb(),
            "load_time_seconds": _load_time_seconds,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=6),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


runpod.serverless.start({"handler": handler})
