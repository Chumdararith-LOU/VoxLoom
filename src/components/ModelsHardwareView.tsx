import { useState } from "react";
import { CheckCircle2, Cpu, Download, HardDrive, Loader2, Power, Zap } from "lucide-react";
import { api } from "../lib/api";
import type { EngineStatus, HardwareProfile, ModelItem } from "../types/api";

interface ModelsHardwareViewProps {
  hardware: HardwareProfile | null;
  models: ModelItem[];
  engineStatus: EngineStatus | null;
  onRefresh: () => Promise<void>;
  onNotify: (text: string, tone?: "info" | "success" | "warning" | "error") => void;
}

interface DownloadProgress {
  modelId: string;
  percent: number;
  downloadedMb: number;
  totalMb: number;
  speedMbS: number;
  etaSec: number | null;
  currentFile: string;
  message: string;
}

function formatSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(0)} MB`;
}

function formatEta(sec: number | null): string {
  if (sec === null || sec < 0 || !Number.isFinite(sec)) return "calculating…";
  if (sec < 60) return `~${sec}s left`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `~${m}m ${String(s).padStart(2, "0")}s left`;
}

export function ModelsHardwareView({
  hardware,
  models,
  engineStatus,
  onRefresh,
  onNotify,
}: ModelsHardwareViewProps) {
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [showOtherPlatforms, setShowOtherPlatforms] = useState(false);

  const downloadingId = downloadProgress?.modelId ?? null;

  const visibleModels = [...models]
    .filter((m) => showOtherPlatforms || m.compatible)
    .sort(
      (a, b) =>
        Number(b.compatible) - Number(a.compatible) ||
        Number(b.recommended) - Number(a.recommended)
    );

  const handleDownload = async (model: ModelItem) => {
    setDownloadProgress({
      modelId: model.id,
      percent: 0,
      downloadedMb: 0,
      totalMb: model.size_gb * 1024,
      speedMbS: 0,
      etaSec: null,
      currentFile: "",
      message: `Connecting to Hugging Face Hub for ${model.name}…`,
    });

    try {
      await api.downloadModelStream(model.id, (evt) => {
        if (evt.stage === "error") {
          throw new Error(String(evt.message || "Download failed"));
        }
        if (evt.stage === "starting" || evt.stage === "downloading" || evt.stage === "complete") {
          setDownloadProgress((prev) => ({
            modelId: model.id,
            percent: typeof evt.percent === "number" ? evt.percent : prev?.percent ?? 0,
            downloadedMb:
              typeof evt.downloaded_mb === "number"
                ? evt.downloaded_mb
                : prev?.downloadedMb ?? 0,
            totalMb:
              typeof evt.total_mb === "number"
                ? evt.total_mb
                : prev?.totalMb ?? model.size_gb * 1024,
            speedMbS:
              typeof evt.speed_mb_s === "number" ? evt.speed_mb_s : prev?.speedMbS ?? 0,
            etaSec:
              typeof evt.eta_sec === "number"
                ? evt.eta_sec
                : evt.eta_sec === null
                ? null
                : prev?.etaSec ?? null,
            currentFile:
              typeof evt.current_file === "string"
                ? evt.current_file
                : prev?.currentFile ?? "",
            message:
              typeof evt.message === "string"
                ? evt.message
                : prev?.message ?? `Downloading ${model.name}…`,
          }));
        }
      });
      await onRefresh();
      onNotify(`Downloaded and cached ${model.name}.`, "success");
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Download failed", "error");
    } finally {
      setDownloadProgress(null);
    }
  };

  const handleLoad = async (model: ModelItem) => {
    setLoadingId(model.id);
    try {
      await api.loadModel(model.id);
      await onRefresh();
      onNotify(`${model.name} is now loaded hot in memory.`, "success");
    } catch (err) {
      onNotify(err instanceof Error ? err.message : "Failed to load model", "error");
    } finally {
      setLoadingId(null);
    }
  };

  const handleUnload = async () => {
    await api.unloadModel();
    await onRefresh();
    onNotify("Unloaded active TTS model and freed memory.", "info");
  };

  return (
    <>
      <section className="deck">
        <div className="deck-header">
          <div>
            <h2 className="deck-title">
              <Cpu size={17} />
              Detected Local Hardware
            </h2>
            <p className="deck-sub">
              VoxLoom automatically selects Apple Silicon MLX on macOS or PyTorch CUDA/CPU on
              Windows and Linux.
            </p>
          </div>
          {engineStatus?.loaded && (
            <button type="button" className="btn" onClick={handleUnload}>
              <Power size={14} />
              Unload active model
            </button>
          )}
        </div>

        {hardware ? (
          <div className="field-grid">
            <div className="model-card">
              <span className="nav-label" style={{ padding: 0 }}>
                Accelerator
              </span>
              <strong>{hardware.accelerator_name}</strong>
              <span className="deck-sub" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                {hardware.os_name} ({hardware.arch}) · {hardware.ram_gb} GB RAM
              </span>
            </div>
            <div className="model-card">
              <span className="nav-label" style={{ padding: 0 }}>
                Recommended Backend
              </span>
              <strong style={{ textTransform: "uppercase" }}>{hardware.recommended_backend}</strong>
              <span className="deck-sub" style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>
                Installed: {hardware.available_backends.join(", ") || "none"}
              </span>
            </div>
            <div className="model-card">
              <span className="nav-label" style={{ padding: 0 }}>
                Media Engine (FFmpeg)
              </span>
              <strong>{hardware.ffmpeg_available ? "Ready" : "Missing"}</strong>
              <span className="deck-sub">
                {hardware.ffmpeg_available
                  ? "Silence trim, atempo & loudnorm enabled"
                  : "Install ffmpeg to enable audio post-processing"}
              </span>
            </div>
          </div>
        ) : (
          <p className="deck-sub">Waiting for local backend telemetry…</p>
        )}
      </section>

      <section className="deck">
        <div className="deck-header">
          <div>
            <h2 className="deck-title">
              <HardDrive size={17} />
              Local Model Manager (Hugging Face Hub)
            </h2>
            <p className="deck-sub">
              Weights download directly from Hugging Face into your local cache (~1.5–3.6 GB). Once
              cached, VoxLoom runs 100% offline.
            </p>
          </div>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={showOtherPlatforms}
              onChange={(e) => setShowOtherPlatforms(e.target.checked)}
            />
            <span>Show other OS backends</span>
          </label>
        </div>

        <div className="model-grid">
          {visibleModels.map((m) => {
            const isDownloading = downloadProgress?.modelId === m.id;
            return (
              <article
                key={m.id}
                className={`model-card ${m.recommended ? "recommended" : ""}`}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                  <strong>{m.name}</strong>
                  {m.loaded && <span className="status-pill pass">HOT IN RAM</span>}
                  {!m.loaded && m.recommended && (
                    <span className="status-pill warn">RECOMMENDED</span>
                  )}
                </div>

                <p className="deck-sub" style={{ fontSize: 12.5 }}>
                  {m.description}
                </p>

                <div className="model-card-meta">
                  <span className="mono-badge">{m.kind.toUpperCase()}</span>
                  <span className="mono-badge">{m.backend.toUpperCase()}</span>
                  <span className="mono-badge">~{m.size_gb} GB</span>
                  <span className="mono-badge">{m.min_ram_gb}GB+ RAM</span>
                  <span className="mono-badge">{m.license}</span>
                </div>

                {isDownloading && downloadProgress && (
                  <div className="download-telemetry" aria-live="polite">
                    <div className="download-telemetry-row">
                      <span>
                        <strong>{downloadProgress.percent.toFixed(1)}%</strong> ·{" "}
                        {formatSize(downloadProgress.downloadedMb)} /{" "}
                        {formatSize(downloadProgress.totalMb)}
                      </span>
                      <span>
                        {downloadProgress.speedMbS > 0
                          ? `${downloadProgress.speedMbS.toFixed(1)} MB/s`
                          : "connecting…"}{" "}
                        · {formatEta(downloadProgress.etaSec)}
                      </span>
                    </div>
                    <progress
                      className="studio-progress"
                      value={downloadProgress.percent}
                      max={100}
                    />
                    <span className="download-telemetry-file">
                      {downloadProgress.currentFile || downloadProgress.message}
                    </span>
                  </div>
                )}

                <div className="btn-row" style={{ marginTop: "auto", paddingTop: 4 }}>
                  {m.downloaded ? (
                    <span className="status-pill pass">
                      <CheckCircle2 size={13} />
                      Cached ({m.cached_size_gb || m.size_gb} GB)
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="btn"
                      disabled={!m.compatible || downloadingId !== null}
                      onClick={() => handleDownload(m)}
                    >
                      {isDownloading ? <Loader2 size={14} /> : <Download size={14} />}
                      {isDownloading
                        ? `Downloading (${downloadProgress?.percent.toFixed(0)}%)`
                        : "Download"}
                    </button>
                  )}

                  {m.kind === "tts" && m.compatible && !m.loaded && (
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={loadingId !== null || downloadingId !== null}
                      onClick={() => handleLoad(m)}
                    >
                      {loadingId === m.id ? <Loader2 size={14} /> : <Zap size={14} />}
                      {loadingId === m.id ? "Loading…" : "Load into memory"}
                    </button>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </>
  );
}
