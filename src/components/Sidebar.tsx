import {
  AudioWaveform,
  Cpu,
  Layers,
  Mic,
  Plus,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { EngineStatus, HardwareProfile, VoiceProfile } from "../types/api";

export type StudioTab = "voice-studio" | "single-take" | "batch-gaps" | "models";

interface SidebarProps {
  activeTab: StudioTab;
  onSelectTab: (tab: StudioTab) => void;
  backendOnline: boolean;
  hardware: HardwareProfile | null;
  engineStatus: EngineStatus | null;
  profiles: VoiceProfile[];
  activeProfileId: string | null;
  onSelectProfile: (id: string) => void;
  onRequestDeleteProfile: (profile: VoiceProfile) => void;
}

export function Sidebar({
  activeTab,
  onSelectTab,
  backendOnline,
  hardware,
  engineStatus,
  profiles,
  activeProfileId,
  onSelectProfile,
  onRequestDeleteProfile,
}: SidebarProps) {
  const backendLabel = !backendOnline
    ? "OFFLINE"
    : engineStatus?.loaded
    ? (engineStatus.backend || hardware?.recommended_backend || "HOT").toUpperCase()
    : (hardware?.recommended_backend || "IDLE").toUpperCase();

  const ledClass = !backendOnline
    ? "offline"
    : engineStatus?.loaded
    ? "hot"
    : "online";

  return (
    <aside className="sidebar" aria-label="Studio navigation">
      <div className="brand-header">
        <div className="brand-mark">
          <div className="brand-glyph" aria-hidden="true">
            <AudioWaveform size={17} />
          </div>
          <h2 className="brand-title">VoxLoom</h2>
        </div>
        <span
          className="hardware-pill"
          title={hardware?.accelerator_name || "Connecting to local engine"}
        >
          <span className={`led ${ledClass}`} aria-hidden="true" />
          {backendLabel}
        </span>
      </div>

      <nav className="nav-group" aria-label="Workspace modes">
        <span className="nav-label">Studio Desk</span>
        <button
          type="button"
          className="nav-btn"
          aria-current={activeTab === "voice-studio" ? "page" : undefined}
          onClick={() => onSelectTab("voice-studio")}
        >
          <Mic size={16} />
          Voice Profiles
        </button>
        <button
          type="button"
          className="nav-btn"
          aria-current={activeTab === "single-take" ? "page" : undefined}
          onClick={() => onSelectTab("single-take")}
        >
          <Sparkles size={16} />
          Single Take
        </button>
        <button
          type="button"
          className="nav-btn"
          aria-current={activeTab === "batch-gaps" ? "page" : undefined}
          onClick={() => onSelectTab("batch-gaps")}
        >
          <Layers size={16} />
          Batch Gap-Fill
        </button>
        <button
          type="button"
          className="nav-btn"
          aria-current={activeTab === "models" ? "page" : undefined}
          onClick={() => onSelectTab("models")}
        >
          <Cpu size={16} />
          Models & Hardware
        </button>
      </nav>

      <div className="nav-group" style={{ flex: 1, minHeight: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            paddingInline: 8,
            paddingBottom: 4,
          }}
        >
          <span className="nav-label" style={{ padding: 0 }}>
            Saved Voices ({profiles.length})
          </span>
          <button
            type="button"
            className="btn-icon"
            title="Create new voice profile"
            aria-label="Create new voice profile"
            onClick={() => onSelectTab("voice-studio")}
          >
            <Plus size={14} />
          </button>
        </div>

        <div className="profile-list">
          {profiles.length === 0 ? (
            <p className="deck-sub" style={{ padding: "6px 8px", fontSize: 12 }}>
              No voice profiles yet. Upload a recording in Voice Profiles to slice your first reference clip.
            </p>
          ) : (
            profiles.map((p) => {
              const isActive = p.id === activeProfileId;
              return (
                <div
                  key={p.id}
                  className={`profile-chip ${isActive ? "active" : ""}`}
                >
                  <button
                    type="button"
                    className="profile-chip-main"
                    onClick={() => onSelectProfile(p.id)}
                  >
                    <span className="profile-chip-name">{p.name}</span>
                    <span className="profile-chip-meta">
                      {p.duration_sec.toFixed(1)}s · {p.language.toUpperCase()}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="btn-icon"
                    aria-label={`Delete ${p.name}`}
                    onClick={() => onRequestDeleteProfile(p)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}
