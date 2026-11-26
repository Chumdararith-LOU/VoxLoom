import { useState } from "react";
import { Download, Loader2, Sparkles, Volume2 } from "lucide-react";
import { api, resolveAssetUrl } from "../lib/api";
import type { PostProcessOptions, SingleGenerateResponse, VoiceProfile } from "../types/api";

interface SingleTakeViewProps {
  activeProfile: VoiceProfile | null;
  onGoToProfiles: () => void;
  onNotify: (text: string, tone?: "info" | "success" | "warning" | "error") => void;
}

export function SingleTakeView({
  activeProfile,
  onGoToProfiles,
  onNotify,
}: SingleTakeViewProps) {
  const [text, setText] = useState(
    "After walking past the bright riverside stalls, we stopped to grab some ice cream and enjoy the cool evening breeze."
  );
  const [speed, setSpeed] = useState(1.0);
  const [temperature, setTemperature] = useState(0.7);
  const [trimSilence, setTrimSilence] = useState(true);
  const [loudnorm, setLoudnorm] = useState(true);
  const [targetLufs, setTargetLufs] = useState(-20.0);

  const [generating, setGenerating] = useState(false);
  const [output, setOutput] = useState<{
    res: SingleGenerateResponse;
    resolvedUrl: string;
    refUrl: string;
  } | null>(null);

  if (!activeProfile) {
    return (
      <section className="deck">
        <h2 className="deck-title">Select or create a Voice Profile first</h2>
        <p className="deck-sub">
          To generate cloned speech, slice an 8–15s reference clip and verify its transcript in
          Voice Profiles.
        </p>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={onGoToProfiles}>
            Open Voice Profiles
          </button>
        </div>
      </section>
    );
  }

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setGenerating(true);
    try {
      const pp: PostProcessOptions = {
        trim_silence: trimSilence,
        pad_sec: 0.05,
        loudnorm,
        target_lufs: targetLufs,
        sample_rate: 48000,
        channels: 1,
        max_atempo: 1.08,
      };
      const res = await api.generateSingle({
        profile_id: activeProfile.id,
        text,
        speed,
        temperature,
        language: activeProfile.language,
        postprocess: pp,
      });
      const resolvedUrl = await resolveAssetUrl(res.audio_url);
      const refUrl = await resolveAssetUrl(`/api/profiles/${activeProfile.id}/audio`);
      setOutput({ res, resolvedUrl, refUrl });
      onNotify(`Generated cloned take (${res.result.final_duration.toFixed(2)}s).`, "success");
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Voice generation failed", "error");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <>
      <section className="deck">
        <div className="deck-header">
          <div>
            <h2 className="deck-title">
              <Sparkles size={17} />
              Single Take Clone — {activeProfile.name}
            </h2>
            <p className="deck-sub">
              Model stays warm in memory after the first run so repeat takes generate in seconds.
            </p>
          </div>
          <span className="mono-badge">
            REF {activeProfile.duration_sec.toFixed(1)}s · {activeProfile.language.toUpperCase()}
          </span>
        </div>

        <form onSubmit={handleGenerate} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="field">
            <label htmlFor="single-script">Script line to synthesize</label>
            <textarea
              id="single-script"
              className="textarea"
              required
              minLength={2}
              value={text}
              onChange={(e) => setText(e.target.value)}
              aria-invalid="false"
              data-error="single-script-err"
            />
            <div id="single-script-err" className="error-msg">
              Error: Enter at least 2 characters to synthesize.
            </div>
          </div>

          <div className="field-grid">
            <div className="field">
              <label htmlFor="single-speed">
                Speaking speed: <code style={{ fontFamily: "var(--font-mono)" }}>{speed.toFixed(2)}x</code>
              </label>
              <input
                id="single-speed"
                type="range"
                min={0.75}
                max={1.25}
                step={0.05}
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
              />
            </div>

            <div className="field">
              <label htmlFor="single-temp">
                Expressiveness (temperature):{" "}
                <code style={{ fontFamily: "var(--font-mono)" }}>{temperature.toFixed(2)}</code>
              </label>
              <input
                id="single-temp"
                type="range"
                min={0.2}
                max={1.1}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </div>

            <div className="field">
              <label htmlFor="single-lufs">Loudness target</label>
              <select
                id="single-lufs"
                className="select"
                value={targetLufs}
                onChange={(e) => setTargetLufs(Number(e.target.value))}
              >
                <option value={-20}>-20 LUFS (Video editor / ChatCut)</option>
                <option value={-16}>-16 LUFS (Podcast / YouTube)</option>
                <option value={-14}>-14 LUFS (Loud streaming)</option>
              </select>
            </div>
          </div>

          <div className="btn-row">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={trimSilence}
                onChange={(e) => setTrimSilence(e.target.checked)}
              />
              <span>Trim leading/trailing silence (50ms pad)</span>
            </label>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={loudnorm}
                onChange={(e) => setLoudnorm(e.target.checked)}
              />
              <span>Normalize loudness (48kHz mono WAV)</span>
            </label>
          </div>

          <div className="btn-row">
            <button type="submit" className="btn btn-primary" disabled={generating}>
              {generating ? <Loader2 size={15} /> : <Sparkles size={15} />}
              {generating ? "Synthesizing locally…" : "Generate take"}
            </button>
          </div>
        </form>
      </section>

      {output && (
        <section className="deck">
          <div className="deck-header">
            <div>
              <h2 className="deck-title">
                <Volume2 size={17} />
                A/B Output Preview
              </h2>
              <p className="deck-sub">
                Raw {output.res.result.raw_duration.toFixed(2)}s → Final{" "}
                <strong>{output.res.result.final_duration.toFixed(2)}s</strong> (48 kHz mono)
              </p>
            </div>
            <a href={output.resolvedUrl} download={output.res.filename} className="btn">
              <Download size={14} />
              Download WAV
            </a>
          </div>

          <div className="field-grid">
            <div className="field">
              <label>Generated clone</label>
              <audio controls className="audio-player" src={output.resolvedUrl} />
            </div>
            <div className="field">
              <label>Original reference ({activeProfile.name})</label>
              <audio controls className="audio-player" src={output.refUrl} />
            </div>
          </div>
        </section>
      )}
    </>
  );
}
