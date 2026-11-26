import { useRef, useState } from "react";
import { FileAudio, Loader2, Mic, Save, Subtitles, Upload } from "lucide-react";
import { api, resolveAssetUrl } from "../lib/api";
import type { UploadAnalysisResponse, VoiceProfile } from "../types/api";
import { WaveformTrimmer } from "./WaveformTrimmer";

interface VoiceStudioViewProps {
  onProfileCreated: (profile: VoiceProfile) => void;
  onNotify: (text: string, tone?: "info" | "success" | "warning" | "error") => void;
}

export function VoiceStudioView({ onProfileCreated, onNotify }: VoiceStudioViewProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [saving, setSaving] = useState(false);

  const [analysis, setAnalysis] = useState<UploadAnalysisResponse | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>("");
  const [startSec, setStartSec] = useState(0);
  const [endSec, setEndSec] = useState(11);

  const [profileName, setProfileName] = useState("");
  const [refText, setRefText] = useState("");
  const [language, setLanguage] = useState("en");
  const [notes, setNotes] = useState("");

  // Optional direct microphone recording
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const handleFile = async (file: File) => {
    setUploading(true);
    try {
      const res = await api.uploadSourceAudio(file);
      const resolvedUrl = await resolveAssetUrl(`/api/profiles/upload/${res.upload_id}/audio`);
      setAnalysis(res);
      setAudioUrl(resolvedUrl);
      setStartSec(res.suggestion.start_sec);
      setEndSec(res.suggestion.end_sec);
      if (!profileName) {
        setProfileName(file.name.replace(/\.[^.]+$/, ""));
      }
      onNotify(
        `Analyzed ${file.name} (${res.metadata.duration.toFixed(1)}s). Suggested cleanest ${res.suggestion.duration_sec.toFixed(1)}s slice.`,
        "success"
      );
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Failed to analyze audio file", "error");
    } finally {
      setUploading(false);
    }
  };

  const toggleMicRecording = async () => {
    if (recording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const file = new File([blob], "mic_recording.webm", { type: "audio/webm" });
        void handleFile(file);
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      onNotify("Recording microphone... speak clearly for 12–15 seconds, then stop.", "info");
    } catch {
      onNotify("Microphone permission denied or unavailable on this device.", "error");
    }
  };

  const handleAutoTranscribe = async () => {
    if (!analysis) return;
    setTranscribing(true);
    try {
      const res = await api.transcribeSlice({
        upload_id: analysis.upload_id,
        start_sec: startSec,
        duration_sec: Math.max(1, endSec - startSec),
        language,
      });
      setRefText(res.text);
      onNotify("Reference slice transcribed locally. Verify every word matches your audio.", "success");
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Local transcription failed", "error");
    } finally {
      setTranscribing(false);
    }
  };

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!analysis) return;
    setSaving(true);
    try {
      const res = await api.createProfile({
        name: profileName,
        upload_id: analysis.upload_id,
        start_sec: startSec,
        duration_sec: Math.max(1, endSec - startSec),
        ref_text: refText,
        language,
        notes,
      });
      onProfileCreated(res.profile);
      onNotify(`Saved voice profile "${res.profile.name}" (24kHz mono ref.wav ready).`, "success");
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Failed to save profile", "error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <section className="deck">
        <div className="deck-header">
          <div>
            <h2 className="deck-title">
              <FileAudio size={17} />
              1. Load or Record Source Voice
            </h2>
            <p className="deck-sub">
              Drop any voiceover recording (.m4a, .wav, .mp3, .mp4) or record from your microphone.
              VoxLoom converts your chosen slice to 24 kHz mono PCM WAV automatically.
            </p>
          </div>
          <div className="btn-row">
            <button type="button" className="btn" onClick={() => fileInputRef.current?.click()}>
              <Upload size={14} />
              Browse file
            </button>
            <button
              type="button"
              className={`btn ${recording ? "btn-danger" : ""}`}
              onClick={toggleMicRecording}
            >
              <Mic size={14} />
              {recording ? "Stop recording" : "Record mic"}
            </button>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,video/*"
          className="visually-hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
          }}
        />

        <div
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) void handleFile(file);
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          {uploading ? (
            <>
              <Loader2 size={24} className="spin" />
              <strong>Analyzing audio levels & silence boundaries…</strong>
            </>
          ) : analysis ? (
            <>
              <strong>{analysis.filename}</strong>
              <span className="deck-sub" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                {analysis.metadata.duration.toFixed(2)}s · {analysis.metadata.sample_rate} Hz ·{" "}
                {analysis.metadata.channels}ch · mean {analysis.metadata.mean_volume_db ?? "--"} dB
              </span>
            </>
          ) : (
            <>
              <Upload size={22} style={{ color: "var(--accent)" }} />
              <strong>Drop your voiceover recording here</strong>
              <span className="deck-sub">Supports M4A, WAV, MP3, FLAC, MOV, MP4</span>
            </>
          )}
        </div>
      </section>

      {analysis && (
        <section className="deck">
          <div className="deck-header">
            <div>
              <h2 className="deck-title">2. Slice Cleanest 8–15s Reference Clip</h2>
              <p className="deck-sub">
                Drag the amber handles on the waveform so your slice starts and ends in natural
                silence without cutting off any word.
              </p>
            </div>
          </div>

          <WaveformTrimmer
            waveform={analysis.waveform}
            audioUrl={audioUrl}
            startSec={startSec}
            endSec={endSec}
            onChangeRange={(s, e) => {
              setStartSec(s);
              setEndSec(e);
            }}
            onAutoPick={() => {
              setStartSec(analysis.suggestion.start_sec);
              setEndSec(analysis.suggestion.end_sec);
              onNotify(analysis.suggestion.reason, "info");
            }}
          />

          <form onSubmit={handleSaveProfile} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div className="field-grid">
              <div className="field">
                <label htmlFor="vp-name">Profile name</label>
                <span id="vp-name-hint" className="hint">
                  Example: My Vlog Voice (Close Mic)
                </span>
                <input
                  id="vp-name"
                  className="input"
                  required
                  minLength={2}
                  value={profileName}
                  onChange={(e) => setProfileName(e.target.value)}
                  aria-describedby="vp-name-hint"
                  aria-invalid="false"
                  data-error="vp-name-err"
                />
                <div id="vp-name-err" className="error-msg">
                  Error: Enter a profile name with at least 2 characters.
                </div>
              </div>

              <div className="field">
                <label htmlFor="vp-lang">Spoken language</label>
                <span id="vp-lang-hint" className="hint">
                  Used by both Whisper transcription and Qwen3-TTS
                </span>
                <select
                  id="vp-lang"
                  className="select"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  aria-describedby="vp-lang-hint"
                >
                  <option value="en">English (en)</option>
                  <option value="zh">Chinese (zh)</option>
                  <option value="ja">Japanese (ja)</option>
                  <option value="ko">Korean (ko)</option>
                  <option value="es">Spanish (es)</option>
                  <option value="fr">French (fr)</option>
                  <option value="de">German (de)</option>
                </select>
              </div>
            </div>

            <div className="field">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                <label htmlFor="vp-ref-text">Exact words spoken in selected slice (--ref_text)</label>
                <button
                  type="button"
                  className="btn"
                  disabled={transcribing}
                  onClick={handleAutoTranscribe}
                >
                  {transcribing ? <Loader2 size={14} /> : <Subtitles size={14} />}
                  {transcribing ? "Transcribing locally…" : "Auto-transcribe slice"}
                </button>
              </div>
              <span id="vp-text-hint" className="hint">
                Must match the audio slice verbatim for clean voice timbre & accent alignment.
              </span>
              <textarea
                id="vp-ref-text"
                className="textarea"
                required
                minLength={5}
                value={refText}
                onChange={(e) => setRefText(e.target.value)}
                placeholder="Click 'Auto-transcribe slice' or type the exact words spoken between IN and OUT..."
                aria-describedby="vp-text-hint"
                aria-invalid="false"
                data-error="vp-text-err"
              />
              <div id="vp-text-err" className="error-msg">
                Error: Provide the exact transcript spoken inside the reference audio slice.
              </div>
            </div>

            <div className="field">
              <label htmlFor="vp-notes">Optional notes</label>
              <input
                id="vp-notes"
                className="input"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Mic setup, room acoustic notes, project name..."
              />
            </div>

            <div className="btn-row">
              <button type="submit" className="btn btn-primary" disabled={saving}>
                {saving ? <Loader2 size={15} /> : <Save size={15} />}
                {saving ? "Slicing ref.wav & saving…" : "Save voice profile"}
              </button>
            </div>
          </form>
        </section>
      )}
    </>
  );
}
