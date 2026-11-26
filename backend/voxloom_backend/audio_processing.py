"""Audio analysis, waveform extraction, silence trimming, atempo fitting, and loudnorm."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from .config import get_ffmpeg_path, get_ffprobe_path


@dataclass
class AudioMetadata:
    duration: float
    sample_rate: int
    channels: int
    codec: str
    size_bytes: int
    mean_volume_db: float | None = None
    max_volume_db: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuggestedSegment:
    start_sec: float
    end_sec: float
    duration_sec: float
    score: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PostProcessResult:
    output_path: str
    raw_duration: float
    trimmed_duration: float
    final_duration: float
    atempo_applied: float
    status: str  # "PASS" | "PASS_ADJUSTED" | "FLAGGED"
    needs_regeneration: bool

    def to_dict(self) -> dict:
        return asdict(self)


def probe_audio(filepath: str | Path) -> AudioMetadata:
    """Inspect duration, sample rate, channels, codec, and loudness stats."""
    ffprobe = get_ffprobe_path()
    ffmpeg = get_ffmpeg_path()
    path = Path(filepath)

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-select_streams",
        "a:0",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    stream = data["streams"][0]
    fmt = data["format"]

    mean_db: float | None = None
    max_db: float | None = None
    vol = subprocess.run(
        [ffmpeg, "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    for line in vol.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                mean_db = float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except ValueError:
                pass
        elif "max_volume:" in line:
            try:
                max_db = float(line.split("max_volume:")[1].split("dB")[0].strip())
            except ValueError:
                pass

    return AudioMetadata(
        duration=float(fmt.get("duration", 0.0)),
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 1)),
        codec=str(stream.get("codec_name", "unknown")),
        size_bytes=int(fmt.get("size", path.stat().st_size)),
        mean_volume_db=mean_db,
        max_volume_db=max_db,
    )


def get_duration(filepath: str | Path) -> float:
    ffprobe = get_ffprobe_path()
    out = subprocess.check_output(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(filepath),
        ],
        text=True,
    ).strip()
    return float(out)


def _decode_mono_float(filepath: str | Path, sample_rate: int = 16000) -> tuple[int, np.ndarray]:
    """Decode any audio/video file to mono float32 numpy array via ffmpeg."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(filepath),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    samples = np.frombuffer(raw, dtype=np.float32)
    return sample_rate, samples


def extract_waveform_peaks(filepath: str | Path, num_bars: int = 240) -> dict:
    """Compute normalized waveform peak bars and active-speech flags for UI rendering."""
    sr, samples = _decode_mono_float(filepath, sample_rate=16000)
    duration = len(samples) / sr if sr > 0 else 0.0
    if len(samples) == 0:
        return {"duration": 0.0, "peaks": [], "silence_mask": []}

    chunk_size = max(1, len(samples) // num_bars)
    trimmed_len = chunk_size * num_bars
    if len(samples) >= trimmed_len:
        blocks = samples[:trimmed_len].reshape(num_bars, chunk_size)
    else:
        padded = np.pad(samples, (0, trimmed_len - len(samples)))
        blocks = padded.reshape(num_bars, chunk_size)

    peaks = np.max(np.abs(blocks), axis=1)
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    max_peak = float(np.max(peaks)) if np.max(peaks) > 0 else 1.0
    norm_peaks = [round(float(p / max_peak), 4) for p in peaks]
    silence_thresh = max(0.008, float(np.max(rms)) * 0.12)
    silence_mask = [bool(r < silence_thresh) for r in rms]

    return {
        "duration": round(duration, 3),
        "peaks": norm_peaks,
        "silence_mask": silence_mask,
    }


def auto_pick_cleanest_segment(
    filepath: str | Path,
    min_sec: float = 10.0,
    max_sec: float = 14.0,
) -> SuggestedSegment:
    """Find the cleanest continuous 10-14s speech window starting/ending in silence."""
    sr, samples = _decode_mono_float(filepath, sample_rate=16000)
    total_dur = len(samples) / sr
    if total_dur <= min_sec:
        return SuggestedSegment(
            start_sec=0.0,
            end_sec=round(total_dur, 2),
            duration_sec=round(total_dur, 2),
            score=1.0,
            reason="Recording is shorter than target window; using full clip.",
        )

    # Frame RMS at 50ms hop
    hop = int(sr * 0.05)
    win = int(sr * 0.05)
    n_frames = max(1, (len(samples) - win) // hop)
    rms = np.array(
        [np.sqrt(np.mean(samples[i * hop : i * hop + win] ** 2)) for i in range(n_frames)]
    )
    peak_rms = float(np.max(rms)) if np.max(rms) > 0 else 1.0
    silence_thresh = max(0.006, peak_rms * 0.1)
    is_silent = rms < silence_thresh

    # Candidate silence midpoints (at least 0.20s of silence = 4 consecutive frames)
    pause_points: list[float] = [0.0]
    i = 0
    while i < len(is_silent):
        if is_silent[i]:
            j = i
            while j < len(is_silent) and is_silent[j]:
                j += 1
            if (j - i) >= 4:
                mid_sec = ((i + j) / 2.0) * 0.05
                pause_points.append(mid_sec)
            i = j
        else:
            i += 1
    pause_points.append(total_dur)

    best_score = -1.0
    best_pair = (0.0, min(total_dur, 11.0))

    for idx_a, start_s in enumerate(pause_points):
        for end_s in pause_points[idx_a + 1 :]:
            dur = end_s - start_s
            if dur < min_sec:
                continue
            if dur > max_sec + 1.5:
                break

            fa = int(start_s / 0.05)
            fb = min(len(rms), int(end_s / 0.05))
            if fb <= fa:
                continue

            seg_rms = rms[fa:fb]
            seg_silent = is_silent[fa:fb]
            speech_ratio = float(np.mean(~seg_silent))

            # Penalize long internal pauses (>0.6s)
            max_pause = 0
            cur = 0
            for flag in seg_silent:
                if flag:
                    cur += 1
                    max_pause = max(max_pause, cur)
                else:
                    cur = 0
            pause_penalty = max(0.0, (max_pause * 0.05) - 0.5) * 0.5

            # Prefer steady energy close to target 11.5s duration
            energy_score = float(np.mean(seg_rms)) / peak_rms
            dur_closeness = 1.0 - abs(dur - 11.5) / 10.0
            score = (speech_ratio * 0.55) + (energy_score * 0.25) + (dur_closeness * 0.2) - pause_penalty

            if score > best_score:
                best_score = score
                best_pair = (start_s, end_s)

    start_s, end_s = best_pair
    return SuggestedSegment(
        start_sec=round(start_s, 2),
        end_sec=round(end_s, 2),
        duration_sec=round(end_s - start_s, 2),
        score=round(max(0.0, min(1.0, best_score)), 3),
        reason="Selected continuous high-clarity speech bounded by natural pauses.",
    )


def slice_reference_wav(
    input_path: str | Path,
    start_sec: float,
    duration_sec: float,
    output_path: str | Path,
) -> Path:
    """Slice a clean 24kHz mono PCM WAV suitable for Qwen3-TTS voice cloning."""
    ffmpeg = get_ffmpeg_path()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{max(1.0, duration_sec):.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def trim_silence_array(
    wav_data: np.ndarray,
    sr: int,
    pad_sec: float = 0.05,
    threshold_ratio: float = 0.012,
) -> np.ndarray:
    """Trim leading and trailing silence while preserving pad_sec buffer."""
    mono = np.mean(wav_data, axis=1) if wav_data.ndim > 1 else wav_data
    if mono.dtype == np.int16:
        float_mono = mono.astype(np.float32) / 32768.0
    else:
        float_mono = mono.astype(np.float32)

    peak = float(np.max(np.abs(float_mono))) if len(float_mono) else 0.0
    if peak == 0.0:
        return wav_data

    win_size = max(1, int(sr * 0.02))
    hop = max(1, int(sr * 0.005))
    thresh = max(1e-4, peak * threshold_ratio)

    num_frames = max(1, (len(float_mono) - win_size) // hop + 1)
    rms = np.array(
        [np.sqrt(np.mean(float_mono[i * hop : i * hop + win_size] ** 2)) for i in range(num_frames)]
    )
    active = np.where(rms > thresh)[0]
    if len(active) == 0:
        return wav_data

    pad_samples = int(pad_sec * sr)
    start_sample = max(0, int(active[0]) * hop - pad_samples)
    end_sample = min(len(wav_data), (int(active[-1]) * hop + win_size) + pad_samples)
    return wav_data[start_sample:end_sample]


def postprocess_clip(
    raw_wav: str | Path,
    output_wav: str | Path,
    *,
    trim_silence: bool = True,
    pad_sec: float = 0.05,
    max_duration: float | None = None,
    max_atempo: float = 1.08,
    loudnorm: bool = True,
    target_lufs: float = -20.0,
    sample_rate: int = 48000,
    channels: int = 1,
) -> PostProcessResult:
    """Apply silence trimming, duration-fit atempo (<=max_atempo), and EBU R128 loudnorm."""
    ffmpeg = get_ffmpeg_path()
    raw_path = Path(raw_wav)
    out_path = Path(output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dur = get_duration(raw_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        trimmed_tmp = Path(tmp.name)

    try:
        if trim_silence:
            sr, data = wavfile.read(str(raw_path))
            trimmed = trim_silence_array(data, sr, pad_sec=pad_sec)
            wavfile.write(str(trimmed_tmp), sr, trimmed)
        else:
            trimmed_tmp.write_bytes(raw_path.read_bytes())

        trimmed_dur = get_duration(trimmed_tmp)
        atempo_factor = 1.0
        needs_regen = False
        status = "PASS"

        if max_duration is not None and trimmed_dur > max_duration:
            needed = trimmed_dur / max_duration
            if needed <= max_atempo:
                atempo_factor = needed
                status = "PASS_ADJUSTED"
            else:
                atempo_factor = max_atempo
                needs_regen = True
                status = "FLAGGED"

        filters: list[str] = []
        if atempo_factor > 1.001:
            filters.append(f"atempo={atempo_factor:.4f}")
        if loudnorm:
            filters.append(f"loudnorm=I={target_lufs:.1f}:TP=-1.5:LRA=7")

        cmd = [ffmpeg, "-y", "-i", str(trimmed_tmp)]
        if filters:
            cmd.extend(["-af", ",".join(filters)])
        cmd.extend(["-ar", str(sample_rate), "-ac", str(channels), str(out_path)])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        final_dur = get_duration(out_path)
        return PostProcessResult(
            output_path=str(out_path),
            raw_duration=round(raw_dur, 3),
            trimmed_duration=round(trimmed_dur, 3),
            final_duration=round(final_dur, 3),
            atempo_applied=round(atempo_factor, 4),
            status=status,
            needs_regeneration=needs_regen,
        )
    finally:
        if trimmed_tmp.exists():
            trimmed_tmp.unlink()
