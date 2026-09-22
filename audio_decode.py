"""ffmpeg-based audio decoding for the R2T2 worker.

Turns arbitrary uploaded audio bytes (M4A/AAC, MP3, WAV, FLAC, OGG, ...)
into a normalized mono/16kHz/PCM16 WAV file. This is the ONE decode path
used by both handler.py and the build-time smoke test (see Dockerfile) --
kept in its own module (no torch/runpod/qwen_asr imports) so the smoke
test can exercise the exact same code without pulling in the ML stack.

Previously handler.py wrote incoming bytes to a generically-named temp
file and fed it straight into soundfile/librosa. libsndfile cannot decode
the M4A/AAC container at all, and the audioread fallback had no usable
backend (no ffmpeg was installed in the image), which produced:
soundfile.LibsndfileError / audioread.exceptions.NoBackendError.
"""
import os
import subprocess
import tempfile

# Whitelist only. A client-supplied filename is never trusted beyond its
# extension -- decode_audio_to_wav() only ever reads os.path.basename() of
# it, and any extension outside this list falls back to a generic suffix
# and lets ffmpeg probe the actual content instead of guessing.
ALLOWED_EXTENSIONS = {
    ".m4a", ".mp4", ".aac", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".webm",
}


class AudioDecodeError(Exception):
    """Raised when ffmpeg cannot decode the supplied audio bytes. Message
    text is always a short, sanitized summary -- never a full stderr dump
    and never the audio bytes themselves."""


def _safe_extension(filename):
    if not filename:
        return ".input"
    base = os.path.basename(str(filename))
    _, ext = os.path.splitext(base)
    ext = ext.lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".input"


def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def decode_audio_to_wav(audio_bytes, filename=None):
    """Writes audio_bytes to a temp source file and uses ffmpeg to produce
    a normalized mono/16kHz/PCM16 WAV file next to it.

    Returns (source_path, wav_path) on success -- the caller owns both and
    must remove them (cleanup() above, in a finally block) once done.

    On failure, this function cleans up any temp files it created itself
    (the caller never receives paths it didn't ask for) and raises
    AudioDecodeError with a short, sanitized message.
    """
    if not audio_bytes:
        raise AudioDecodeError("empty audio payload")

    suffix = _safe_extension(filename)
    src_fd, src_path = tempfile.mkstemp(suffix=suffix)
    wav_path = src_path + ".norm.wav"

    with os.fdopen(src_fd, "wb") as f:
        f.write(audio_bytes)

    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", src_path,
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        cleanup(src_path, wav_path)
        raise AudioDecodeError("ffmpeg decode timed out")

    if proc.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        stderr_lines = (proc.stderr or "").strip().splitlines()
        summary = stderr_lines[-1] if stderr_lines else "unknown ffmpeg error"
        cleanup(src_path, wav_path)
        raise AudioDecodeError(f"ffmpeg could not decode audio: {summary}")

    return src_path, wav_path
