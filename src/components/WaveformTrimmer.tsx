import { useEffect, useRef, useState } from "react";
import { Pause, Play, Wand2 } from "lucide-react";
import type { WaveformData } from "../types/api";

interface WaveformTrimmerProps {
  waveform: WaveformData;
  audioUrl: string;
  startSec: number;
  endSec: number;
  onChangeRange: (start: number, end: number) => void;
  onAutoPick?: () => void;
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, "0")}`;
}

export function WaveformTrimmer({
  waveform,
  audioUrl,
  startSec,
  endSec,
  onChangeRange,
  onAutoPick,
}: WaveformTrimmerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const dragModeRef = useRef<"none" | "start" | "end" | "move">("none");
  const dragOffsetRef = useRef<number>(0);

  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadSec, setPlayheadSec] = useState(startSec);

  const duration = Math.max(waveform.duration, 0.1);
  const selDur = Math.max(0, endSec - startSec);

  // Render canvas waveform with silence shading, selection window, handles, and playhead
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim() || "#f59e0b";
    const teal = styles.getPropertyValue("--signal-teal").trim() || "#22d3ee";
    const muted = styles.getPropertyValue("--ink-muted").trim() || "#758699";
    const borderStrong = styles.getPropertyValue("--border-strong").trim() || "#435060";

    ctx.clearRect(0, 0, w, h);

    // Center zero line
    ctx.strokeStyle = borderStrong;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const xStart = (startSec / duration) * w;
    const xEnd = (endSec / duration) * w;

    // Selection window highlight
    ctx.fillStyle = "rgba(245, 158, 11, 0.14)";
    ctx.fillRect(xStart, 0, Math.max(2, xEnd - xStart), h);

    // Bars
    const peaks = waveform.peaks;
    const silenceMask = waveform.silence_mask;
    const barCount = peaks.length;
    if (barCount > 0) {
      const step = w / barCount;
      const barW = Math.max(1.5, step * 0.68);
      for (let i = 0; i < barCount; i++) {
        const x = i * step;
        const t = (i / barCount) * duration;
        const inSelection = t >= startSec && t <= endSec;
        const isSilent = silenceMask[i];
        const amp = Math.max(0.04, peaks[i]);
        const barH = amp * (h * 0.82);

        if (inSelection) {
          ctx.fillStyle = isSilent ? muted : accent;
        } else {
          ctx.fillStyle = isSilent ? "rgba(117, 134, 153, 0.3)" : "rgba(117, 134, 153, 0.65)";
        }
        ctx.fillRect(x, (h - barH) / 2, barW, barH);
      }
    }

    // Start & End Handles
    ctx.fillStyle = accent;
    ctx.fillRect(xStart - 2, 0, 4, h);
    ctx.fillRect(xEnd - 2, 0, 4, h);

    // Handle grip caps
    ctx.fillRect(xStart - 6, 0, 12, 16);
    ctx.fillRect(xEnd - 6, 0, 12, 16);

    // Playhead
    if (playheadSec >= 0 && playheadSec <= duration) {
      const xPlay = (playheadSec / duration) * w;
      ctx.strokeStyle = teal;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xPlay, 0);
      ctx.lineTo(xPlay, h);
      ctx.stroke();
    }
  }, [waveform, startSec, endSec, duration, playheadSec]);

  // Playback loop bounded to [startSec, endSec]
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    let rafId = 0;

    const tick = () => {
      if (!audio.paused) {
        setPlayheadSec(audio.currentTime);
        if (audio.currentTime >= endSec) {
          audio.pause();
          audio.currentTime = startSec;
          setIsPlaying(false);
          setPlayheadSec(startSec);
          return;
        }
        rafId = requestAnimationFrame(tick);
      }
    };

    const onPlay = () => {
      setIsPlaying(true);
      rafId = requestAnimationFrame(tick);
    };
    const onPause = () => {
      setIsPlaying(false);
      cancelAnimationFrame(rafId);
    };

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    return () => {
      cancelAnimationFrame(rafId);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
    };
  }, [startSec, endSec]);

  const clientXToSec = (clientX: number): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return ratio * duration;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.setPointerCapture(e.pointerId);
    const sec = clientXToSec(e.clientX);
    const tol = Math.max(0.4, duration * 0.025);

    if (Math.abs(sec - startSec) <= tol) {
      dragModeRef.current = "start";
    } else if (Math.abs(sec - endSec) <= tol) {
      dragModeRef.current = "end";
    } else if (sec > startSec && sec < endSec) {
      dragModeRef.current = "move";
      dragOffsetRef.current = sec - startSec;
    } else {
      // Jump selection center here
      const half = selDur / 2;
      const nextStart = Math.max(0, Math.min(duration - selDur, sec - half));
      onChangeRange(Number(nextStart.toFixed(2)), Number((nextStart + selDur).toFixed(2)));
      dragModeRef.current = "move";
      dragOffsetRef.current = sec - nextStart;
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragModeRef.current === "none") return;
    const sec = clientXToSec(e.clientX);
    const minWindow = 2.0;

    if (dragModeRef.current === "start") {
      const next = Math.max(0, Math.min(endSec - minWindow, sec));
      onChangeRange(Number(next.toFixed(2)), endSec);
    } else if (dragModeRef.current === "end") {
      const next = Math.min(duration, Math.max(startSec + minWindow, sec));
      onChangeRange(startSec, Number(next.toFixed(2)));
    } else if (dragModeRef.current === "move") {
      const nextStart = Math.max(0, Math.min(duration - selDur, sec - dragOffsetRef.current));
      onChangeRange(Number(nextStart.toFixed(2)), Number((nextStart + selDur).toFixed(2)));
    }
  };

  const onPointerUp = () => {
    dragModeRef.current = "none";
  };

  const togglePlaySlice = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.currentTime = startSec;
      void audio.play();
    }
  };

  const durationHealthy = selDur >= 8 && selDur <= 15;

  return (
    <div className="waveform-deck">
      <div className="waveform-canvas-wrap">
        <canvas
          ref={canvasRef}
          className="waveform-canvas"
          role="img"
          aria-label={`Audio waveform from ${fmtTime(startSec)} to ${fmtTime(endSec)}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        />
      </div>

      <audio ref={audioRef} src={audioUrl} preload="auto" />

      <div className="waveform-readout">
        <div className="btn-row">
          <button type="button" className="btn" onClick={togglePlaySlice}>
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            {isPlaying ? "Stop slice" : "Play selected slice"}
          </button>
          {onAutoPick && (
            <button type="button" className="btn" onClick={onAutoPick}>
              <Wand2 size={14} />
              Auto-pick cleanest 10–14s
            </button>
          )}
        </div>

        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center" }}>
          <span>
            IN <strong>{fmtTime(startSec)}</strong>
          </span>
          <span>
            OUT <strong>{fmtTime(endSec)}</strong>
          </span>
          <span className={`status-pill ${durationHealthy ? "pass" : "warn"}`}>
            SLICE {selDur.toFixed(2)}s {durationHealthy ? "(ideal 8–15s)" : "(aim for 8–15s)"}
          </span>
        </div>
      </div>
    </div>
  );
}
