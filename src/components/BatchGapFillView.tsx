import { useRef, useState } from "react";
import {
  Archive,
  FileUp,
  Layers,
  Loader2,
  Play,
  Plus,
  Trash2,
} from "lucide-react";
import { api, resolveAssetUrl } from "../lib/api";
import type { BatchGapInput, BatchGapResult, VoiceProfile } from "../types/api";

interface BatchGapFillViewProps {
  activeProfile: VoiceProfile | null;
  onGoToProfiles: () => void;
  onNotify: (text: string, tone?: "info" | "success" | "warning" | "error") => void;
}

const SAMPLE_GAPS: BatchGapInput[] = [
  {
    id: "s1",
    filename: "gap_01_0m16.9s.wav",
    timeline_start: "0m16.9s",
    max_duration: 8.8,
    text: "The road was full of people, so we just followed the crowd toward the river.",
  },
  {
    id: "s2",
    filename: "gap_02_0m33.0s.wav",
    timeline_start: "0m33.0s",
    max_duration: 17.5,
    text: "Everywhere we looked, there were families, kids and people carrying bags. The lights along the riverside were so bright and pretty.",
  },
  {
    id: "s3",
    filename: "gap_03_2m25.1s.wav",
    timeline_start: "2m25.1s",
    max_duration: 2.2,
    text: "What a fun night!",
  },
];

function parseImportedScript(content: string, filename: string): BatchGapInput[] {
  if (filename.toLowerCase().endsWith(".json")) {
    const parsed = JSON.parse(content) as Array<Record<string, unknown>>;
    return parsed.map((row, idx) => ({
      id: `imp_${idx}_${Date.now()}`,
      filename: String(row.filename || `clip_${String(idx + 1).padStart(2, "0")}.wav`),
      timeline_start: String(row.timeline_start || row.start || ""),
      max_duration:
        row.max_duration !== undefined && row.max_duration !== null && row.max_duration !== ""
          ? Number(row.max_duration)
          : null,
      text: String(row.text || ""),
    }));
  }

  // Simple CSV parser: filename,timeline_start,max_duration,text
  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
  const startIdx = lines[0]?.toLowerCase().includes("filename") ? 1 : 0;
  return lines.slice(startIdx).map((line, idx) => {
    const parts = line.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/).map((s) =>
      s.trim().replace(/^"|"$/g, "")
    );
    return {
      id: `csv_${idx}_${Date.now()}`,
      filename: parts[0] || `clip_${String(idx + 1).padStart(2, "0")}.wav`,
      timeline_start: parts[1] || "",
      max_duration: parts[2] ? Number(parts[2]) : null,
      text: parts[3] || "",
    };
  });
}

export function BatchGapFillView({
  activeProfile,
  onGoToProfiles,
  onNotify,
}: BatchGapFillViewProps) {
  const importRef = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<BatchGapInput[]>(SAMPLE_GAPS);
  const [targetLufs, setTargetLufs] = useState(-20.0);
  const [maxAtempo, setMaxAtempo] = useState(1.08);
  const [running, setRunning] = useState(false);
  const [progressMsg, setProgressMsg] = useState("");
  const [completedCount, setCompletedCount] = useState(0);
  const [results, setResults] = useState<Record<string, BatchGapResult>>({});
  const [zipUrl, setZipUrl] = useState<string | null>(null);
  const [resolvedAudioUrls, setResolvedAudioUrls] = useState<Record<string, string>>({});

  if (!activeProfile) {
    return (
      <section className="deck">
        <h2 className="deck-title">Select a Voice Profile for Batch Gap-Fill</h2>
        <p className="deck-sub">
          Batch Gap-Fill loads Qwen3-TTS once in memory, generates every timeline gap clip, trims
          silence, fits duration with gentle atempo, and normalizes loudness.
        </p>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={onGoToProfiles}>
            Open Voice Profiles
          </button>
        </div>
      </section>
    );
  }

  const addRow = () => {
    const n = items.length + 1;
    setItems((prev) => [
      ...prev,
      {
        id: `row_${Date.now()}`,
        filename: `gap_${String(n).padStart(2, "0")}.wav`,
        timeline_start: "",
        max_duration: 10.0,
        text: "",
      },
    ]);
  };

  const updateRow = (id: string, patch: Partial<BatchGapInput>) => {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));
  };

  const removeRow = (id: string) => {
    setItems((prev) => prev.filter((it) => it.id !== id));
  };

  const handleImportFile = async (file: File) => {
    try {
      const content = await file.text();
      const parsed = parseImportedScript(content, file.name);
      if (parsed.length === 0) throw new Error("No valid rows found in imported file.");
      setItems(parsed);
      setResults({});
      setZipUrl(null);
      onNotify(`Imported ${parsed.length} gap items from ${file.name}.`, "success");
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Import failed", "error");
    }
  };

  const handleRunBatch = async () => {
    const valid = items.filter((it) => it.text.trim().length > 0);
    if (valid.length === 0) {
      onNotify("Add at least one script row with text before running batch.", "warning");
      return;
    }

    setRunning(true);
    setCompletedCount(0);
    setResults({});
    setZipUrl(null);
    setProgressMsg("Preparing local TTS model…");

    try {
      await api.generateBatchStream(
        {
          profile_id: activeProfile.id,
          items: valid.map(({ filename, timeline_start, max_duration, text }) => ({
            filename,
            timeline_start,
            max_duration,
            text,
          })),
          speed: 1.0,
          temperature: 0.7,
          language: activeProfile.language,
          postprocess: {
            trim_silence: true,
            pad_sec: 0.05,
            loudnorm: true,
            target_lufs: targetLufs,
            sample_rate: 48000,
            channels: 1,
            max_atempo: maxAtempo,
          },
        },
        (evt) => {
          const type = String(evt.type || "");
          if (type === "status" || type === "item_retry") {
            setProgressMsg(String(evt.message || ""));
          } else if (type === "item_start") {
            setProgressMsg(
              `Generating [${evt.index}/${evt.total}] ${evt.filename}…`
            );
          } else if (type === "item_done") {
            const row = evt.item as BatchGapResult;
            setCompletedCount((c) => c + 1);
            setResults((prev) => ({ ...prev, [row.filename]: row }));
            void resolveAssetUrl(row.audio_url).then((url) =>
              setResolvedAudioUrls((prev) => ({ ...prev, [row.filename]: url }))
            );
          } else if (type === "complete") {
            const z = String(evt.zip_url || "");
            if (z) {
              void resolveAssetUrl(z).then(setZipUrl);
            }
            setProgressMsg(`Finished ${valid.length} clips.`);
            onNotify(`Batch complete: ${valid.length} normalized WAV clips ready.`, "success");
          } else if (type === "error") {
            throw new Error(String(evt.message || "Batch failed"));
          }
        }
      );
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Batch run failed", "error");
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="deck">
      <div className="deck-header">
        <div>
          <h2 className="deck-title">
            <Layers size={17} />
            Batch Timeline Gap-Fill — {activeProfile.name}
          </h2>
          <p className="deck-sub">
            Loads the model once, trims silence (50ms pad), fits over-length clips up to{" "}
            {maxAtempo.toFixed(2)}x atempo, and normalizes loudness to {targetLufs} LUFS.
          </p>
        </div>

        <div className="btn-row">
          <input
            ref={importRef}
            type="file"
            accept=".json,.csv"
            className="visually-hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleImportFile(f);
            }}
          />
          <button type="button" className="btn" onClick={() => importRef.current?.click()}>
            <FileUp size={14} />
            Import JSON / CSV
          </button>
          <button type="button" className="btn" onClick={addRow}>
            <Plus size={14} />
            Add row
          </button>
          {zipUrl && (
            <a href={zipUrl} download className="btn">
              <Archive size={14} />
              Download ZIP
            </a>
          )}
          <button
            type="button"
            className="btn btn-primary"
            disabled={running}
            onClick={handleRunBatch}
          >
            {running ? <Loader2 size={14} /> : <Play size={14} />}
            {running ? "Generating batch…" : `Generate all (${items.length})`}
          </button>
        </div>
      </div>

      <div className="field-grid">
        <div className="field">
          <label htmlFor="batch-lufs">Target loudness</label>
          <select
            id="batch-lufs"
            className="select"
            value={targetLufs}
            onChange={(e) => setTargetLufs(Number(e.target.value))}
          >
            <option value={-20}>-20 LUFS (ChatCut / Video Timeline)</option>
            <option value={-16}>-16 LUFS (Podcast / Narration)</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="batch-atempo">
            Max auto-fit speedup:{" "}
            <code style={{ fontFamily: "var(--font-mono)" }}>{maxAtempo.toFixed(2)}x</code>
          </label>
          <input
            id="batch-atempo"
            type="range"
            min={1.0}
            max={1.15}
            step={0.01}
            value={maxAtempo}
            onChange={(e) => setMaxAtempo(Number(e.target.value))}
          />
        </div>
      </div>

      {(running || completedCount > 0) && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div className="waveform-readout">
            <span>{progressMsg}</span>
            <span>
              {completedCount} / {items.length} clips
            </span>
          </div>
          <progress className="studio-progress" value={completedCount} max={Math.max(1, items.length)} />
        </div>
      )}

      <div
        tabIndex={0}
        role="region"
        aria-labelledby="batch-table-caption"
        className="table-wrapper"
      >
        <table className="studio-table">
          <caption id="batch-table-caption">
            Timeline gap queue — edit filenames, max durations, and scripts inline before running
          </caption>
          <thead>
            <tr>
              <th>Filename</th>
              <th>Start</th>
              <th>Max (s)</th>
              <th>Script Text</th>
              <th>Actual / Fit</th>
              <th>Status</th>
              <th>Preview</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => {
              const res = results[row.filename];
              const pct =
                res && row.max_duration
                  ? Math.min(100, (res.actual_duration / row.max_duration) * 100)
                  : 0;
              const statusClass =
                res?.status === "FLAGGED"
                  ? "flag"
                  : res?.status === "PASS_ADJUSTED"
                  ? "warn"
                  : "pass";

              return (
                <tr key={row.id}>
                  <th>
                    <input
                      aria-label="Output filename"
                      className="input"
                      style={{ minWidth: 165, fontFamily: "var(--font-mono)", fontSize: 12 }}
                      value={row.filename}
                      onChange={(e) => updateRow(row.id, { filename: e.target.value })}
                    />
                  </th>
                  <td data-label="Start">
                    <input
                      aria-label="Timeline start"
                      className="input"
                      style={{ width: 84, fontFamily: "var(--font-mono)", fontSize: 12 }}
                      value={row.timeline_start}
                      placeholder="0m16.9s"
                      onChange={(e) => updateRow(row.id, { timeline_start: e.target.value })}
                    />
                  </td>
                  <td data-label="Max (s)">
                    <input
                      aria-label="Max duration seconds"
                      type="number"
                      step="0.1"
                      min="0.5"
                      className="input"
                      style={{ width: 76, fontFamily: "var(--font-mono)", fontSize: 12 }}
                      value={row.max_duration ?? ""}
                      onChange={(e) =>
                        updateRow(row.id, {
                          max_duration: e.target.value ? Number(e.target.value) : null,
                        })
                      }
                    />
                  </td>
                  <td data-label="Script Text" style={{ minWidth: 260 }}>
                    <textarea
                      aria-label="Script line"
                      className="textarea"
                      style={{ minHeight: 54, fontSize: 12.5 }}
                      value={row.text}
                      onChange={(e) => updateRow(row.id, { text: e.target.value })}
                    />
                  </td>
                  <td data-label="Actual / Fit">
                    {res ? (
                      <div className="duration-meter">
                        <span>
                          {res.actual_duration.toFixed(2)}s / {row.max_duration ?? "--"}s
                        </span>
                        {row.max_duration && (
                          <div className="duration-bar">
                            <div
                              className={`duration-bar-fill ${statusClass}`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="deck-sub">—</span>
                    )}
                  </td>
                  <td data-label="Status">
                    {res ? (
                      <span className={`status-pill ${statusClass}`}>
                        {res.status === "PASS_ADJUSTED"
                          ? `PASS (${res.atempo_applied.toFixed(2)}x)`
                          : res.status}
                      </span>
                    ) : (
                      <span className="deck-sub">Queued</span>
                    )}
                  </td>
                  <td data-label="Preview">
                    <div className="btn-row">
                      {resolvedAudioUrls[row.filename] && (
                        <audio
                          controls
                          src={resolvedAudioUrls[row.filename]}
                          style={{ height: 30, width: 150 }}
                        />
                      )}
                      <button
                        type="button"
                        className="btn-icon"
                        aria-label={`Remove ${row.filename}`}
                        onClick={() => removeRow(row.id)}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
