import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "./lib/api";
import { announceStatus, initAriaValidationSync } from "./lib/domUtils";
import type {
  EngineStatus,
  HardwareProfile,
  ModelItem,
  ToastMessage,
  VoiceProfile,
} from "./types/api";
import { BatchGapFillView } from "./components/BatchGapFillView";
import { ConfirmDeleteDialog, ConsentDialog } from "./components/ConsentDialog";
import { ModelsHardwareView } from "./components/ModelsHardwareView";
import { Sidebar, type StudioTab } from "./components/Sidebar";
import { SingleTakeView } from "./components/SingleTakeView";
import { ToastRegion } from "./components/ToastRegion";
import { VoiceStudioView } from "./components/VoiceStudioView";

const CONSENT_KEY = "voxloom_consent_accepted_v1";

const TAB_TITLES: Record<StudioTab, { title: string; subtitle: string }> = {
  "voice-studio": {
    title: "Voice Profile Studio",
    subtitle: "Slice a clean 8–15s reference clip, auto-transcribe locally, and save reusable voices",
  },
  "single-take": {
    title: "Single Take Cloner",
    subtitle: "Synthesize and A/B compare single lines against your reference voice",
  },
  "batch-gaps": {
    title: "Batch Timeline Gap-Fill",
    subtitle: "Generate full voiceover scripts with silence trim, atempo duration fit, and loudnorm",
  },
  models: {
    title: "Models & Hardware",
    subtitle: "Manage local Hugging Face weights and inspect Apple Silicon MLX / CUDA / CPU status",
  },
};

export default function App() {
  const [activeTab, setActiveTab] = useState<StudioTab>("voice-studio");
  const [consentOpen, setConsentOpen] = useState(() => {
    try {
      return localStorage.getItem(CONSENT_KEY) !== "true";
    } catch {
      return true;
    }
  });

  const [backendOnline, setBackendOnline] = useState(false);
  const [startingUp, setStartingUp] = useState(true);
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [profiles, setProfiles] = useState<VoiceProfile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
  const [deleteCandidate, setDeleteCandidate] = useState<VoiceProfile | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const pushToast = useCallback(
    (text: string, tone: ToastMessage["tone"] = "info") => {
      const id = `${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      setToasts((prev) => [...prev.slice(-3), { id, text, tone }]);
      announceStatus(text);
    },
    []
  );

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const refreshAll = useCallback(async () => {
    try {
      const [health, hw, modRes, profRes] = await Promise.all([
        api.health(),
        api.hardware(),
        api.listModels(),
        api.listProfiles(),
      ]);
      setBackendOnline(health.status === "ok");
      if (health.status === "ok") setStartingUp(false);
      setEngineStatus(health.engine);
      setHardware(hw);
      setModels(modRes.models);
      setProfiles(profRes.profiles);
      setActiveProfileId((prev) => {
        if (prev && profRes.profiles.some((p) => p.id === prev)) return prev;
        return profRes.profiles[0]?.id ?? null;
      });
    } catch {
      setBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    const cleanupAria = initAriaValidationSync(document);
    void refreshAll();
    const startupGrace = window.setTimeout(() => setStartingUp(false), 12000);
    return () => {
      cleanupAria();
      window.clearTimeout(startupGrace);
    };
  }, [refreshAll]);

  useEffect(() => {
    const intervalMs = backendOnline ? 8000 : 1200;
    const timer = window.setInterval(() => {
      void refreshAll();
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [backendOnline, refreshAll]);

  const activeProfile = profiles.find((p) => p.id === activeProfileId) ?? null;
  const meta = TAB_TITLES[activeTab];

  const handleAcceptConsent = () => {
    try {
      localStorage.setItem(CONSENT_KEY, "true");
    } catch {
      // ignore storage errors
    }
    setConsentOpen(false);
  };

  const handleConfirmDelete = async () => {
    if (!deleteCandidate) return;
    const target = deleteCandidate;
    setDeleteCandidate(null);
    try {
      await api.deleteProfile(target.id);
      await refreshAll();
      pushToast(`Deleted voice profile "${target.name}".`, "info");
    } catch (err) {
      pushToast(err instanceof Error ? err.message : "Delete failed", "error");
    }
  };

  return (
    <div className="studio-shell">
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        backendOnline={backendOnline}
        hardware={hardware}
        engineStatus={engineStatus}
        profiles={profiles}
        activeProfileId={activeProfileId}
        onSelectProfile={(id) => setActiveProfileId(id)}
        onRequestDeleteProfile={(p) => setDeleteCandidate(p)}
      />

      <main className="workspace">
        <header className="topbar">
          <div className="topbar-title-group">
            <h1>{meta.title}</h1>
            <p className="topbar-subtitle">{meta.subtitle}</p>
          </div>

          <div className="telemetry-strip">
            {activeProfile && (
              <span className="mono-badge">
                VOICE: <strong>{activeProfile.name}</strong>
              </span>
            )}
            <span className="mono-badge">
              ENGINE: {engineStatus?.loaded ? engineStatus.model_id : "standby"}
            </span>
            <button
              type="button"
              className="btn-icon"
              title="Refresh hardware and profile status"
              aria-label="Refresh status"
              onClick={() => void refreshAll()}
            >
              <RefreshCw size={15} />
            </button>
          </div>
        </header>

        <div className="workspace-body">
          {!backendOnline && (
            <div className="banner-warn" role="alert">
              <span>
                {startingUp ? (
                  <>Starting local AI engine… connecting automatically in a few seconds.</>
                ) : (
                  <>
                    Local AI engine is offline. In dev mode run <code>pnpm backend:dev</code>, or
                    restart the desktop app.
                  </>
                )}
              </span>
              <button type="button" className="btn" onClick={() => void refreshAll()}>
                Retry connection
              </button>
            </div>
          )}

          {activeTab === "voice-studio" && (
            <VoiceStudioView
              onProfileCreated={(profile) => {
                setProfiles((prev) => [profile, ...prev]);
                setActiveProfileId(profile.id);
                setActiveTab("single-take");
              }}
              onNotify={pushToast}
            />
          )}

          {activeTab === "single-take" && (
            <SingleTakeView
              activeProfile={activeProfile}
              onGoToProfiles={() => setActiveTab("voice-studio")}
              onNotify={pushToast}
            />
          )}

          {activeTab === "batch-gaps" && (
            <BatchGapFillView
              activeProfile={activeProfile}
              onGoToProfiles={() => setActiveTab("voice-studio")}
              onNotify={pushToast}
            />
          )}

          {activeTab === "models" && (
            <ModelsHardwareView
              hardware={hardware}
              models={models}
              engineStatus={engineStatus}
              onRefresh={refreshAll}
              onNotify={pushToast}
            />
          )}
        </div>
      </main>

      <ConsentDialog open={consentOpen} onAccept={handleAcceptConsent} />
      <ConfirmDeleteDialog
        profileName={deleteCandidate?.name ?? null}
        onConfirm={() => void handleConfirmDelete()}
        onCancel={() => setDeleteCandidate(null)}
      />
      <ToastRegion toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
