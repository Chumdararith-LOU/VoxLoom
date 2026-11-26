"""Unit tests for ffmpeg/ffprobe audio analysis, silence trimming, atempo, and loudnorm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from voxloom_backend.audio_processing import (
    auto_pick_cleanest_segment,
    extract_waveform_peaks,
    get_duration,
    postprocess_clip,
    probe_audio,
    slice_reference_wav,
)


def _make_tone_with_silence(path: Path, lead_s: float, tone_s: float, trail_s: float, sr: int = 24000) -> Path:
    t = np.linspace(0, tone_s, int(sr * tone_s), endpoint=False)
    tone = 0.35 * np.sin(2 * np.pi * 220.0 * t)
    lead = np.zeros(int(sr * lead_s), dtype=np.float32)
    trail = np.zeros(int(sr * trail_s), dtype=np.float32)
    full = np.concatenate([lead, tone.astype(np.float32), trail])
    pcm = (full * 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm)
    return path


def test_probe_waveform_and_auto_pick(tmp_path: Path) -> None:
    # Build a 20s clip: 2s silence, 12s tone, 6s silence
    src = _make_tone_with_silence(tmp_path / "source.wav", lead_s=2.0, tone_s=12.0, trail_s=6.0)

    meta = probe_audio(src)
    assert 19.8 <= meta.duration <= 20.2
    assert meta.sample_rate == 24000
    assert meta.channels == 1

    wf = extract_waveform_peaks(src, num_bars=100)
    assert len(wf["peaks"]) == 100
    assert len(wf["silence_mask"]) == 100
    assert wf["silence_mask"][0] is True  # leading silence
    assert wf["silence_mask"][50] is False  # active middle speech

    suggestion = auto_pick_cleanest_segment(src, min_sec=10.0, max_sec=14.0)
    assert 10.0 <= suggestion.duration_sec <= 15.5
    assert suggestion.start_sec <= 2.5

    ref_wav = slice_reference_wav(src, suggestion.start_sec, suggestion.duration_sec, tmp_path / "ref.wav")
    assert ref_wav.exists()
    ref_meta = probe_audio(ref_wav)
    assert ref_meta.sample_rate == 24000
    assert ref_meta.channels == 1


def test_postprocess_trim_and_pass(tmp_path: Path) -> None:
    raw = _make_tone_with_silence(tmp_path / "raw.wav", lead_s=1.0, tone_s=3.0, trail_s=1.0)
    out = tmp_path / "final.wav"

    res = postprocess_clip(
        raw,
        out,
        trim_silence=True,
        pad_sec=0.05,
        max_duration=4.0,
        loudnorm=True,
        target_lufs=-20.0,
        sample_rate=48000,
        channels=1,
    )
    assert out.exists()
    assert res.status == "PASS"
    assert res.needs_regeneration is False
    assert 2.9 <= res.final_duration <= 3.3
    assert probe_audio(out).sample_rate == 48000


def test_postprocess_atempo_fit_and_flag(tmp_path: Path) -> None:
    # 5.2s speech into 5.0s max -> needs ~1.04x speedup (<= 1.08x) -> PASS_ADJUSTED
    raw_fit = _make_tone_with_silence(tmp_path / "raw_fit.wav", lead_s=0.2, tone_s=5.2, trail_s=0.2)
    res_fit = postprocess_clip(
        raw_fit,
        tmp_path / "out_fit.wav",
        max_duration=5.0,
        max_atempo=1.08,
        loudnorm=False,
    )
    assert res_fit.status == "PASS_ADJUSTED"
    assert res_fit.needs_regeneration is False
    assert 1.02 <= res_fit.atempo_applied <= 1.08
    assert get_duration(tmp_path / "out_fit.wav") <= 5.1

    # 6.5s speech into 4.0s max -> exceeds 1.08x -> FLAGGED + needs_regeneration
    raw_long = _make_tone_with_silence(tmp_path / "raw_long.wav", lead_s=0.1, tone_s=6.5, trail_s=0.1)
    res_flag = postprocess_clip(
        raw_long,
        tmp_path / "out_flag.wav",
        max_duration=4.0,
        max_atempo=1.08,
        loudnorm=False,
    )
    assert res_flag.status == "FLAGGED"
    assert res_flag.needs_regeneration is True
    assert res_flag.atempo_applied == 1.08
