import type {
  BatchGapInput,
  HardwareProfile,
  HealthResponse,
  ModelItem,
  PostProcessOptions,
  SingleGenerateResponse,
  UploadAnalysisResponse,
  VoiceProfile,
} from "../types/api";

let cachedBaseUrl: string | null = null;

export async function getApiBaseUrl(): Promise<string> {
  if (cachedBaseUrl !== null) return cachedBaseUrl;

  // In Tauri desktop runtime, query the Rust sidecar host for the dynamic localhost port.
  if (typeof window !== "undefined" && "__TAURI_INTERNALS__" in window) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const url = await invoke<string>("get_backend_url");
      if (url) {
        cachedBaseUrl = url.replace(/\/$/, "");
        return cachedBaseUrl;
      }
    } catch {
      // Fallback to Vite proxy / default localhost port
    }
  }

  cachedBaseUrl = "";
  return cachedBaseUrl;
}

export async function resolveAssetUrl(path: string): Promise<string> {
  const base = await getApiBaseUrl();
  return `${base}${path}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const base = await getApiBaseUrl();
  const res = await fetch(`${base}${path}`, init);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore non-JSON error
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => requestJson<HealthResponse>("/api/system/health"),
  hardware: () => requestJson<HardwareProfile>("/api/system/hardware"),

  listModels: () =>
    requestJson<{ models: ModelItem[]; active: HealthResponse["engine"] }>("/api/models"),

  loadModel: (model_id?: string) =>
    requestJson<HealthResponse["engine"]>("/api/models/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: model_id ?? null }),
    }),

  unloadModel: () =>
    requestJson<HealthResponse["engine"]>("/api/models/unload", {
      method: "POST",
    }),

  async downloadModelStream(
    model_id: string,
    onEvent: (evt: Record<string, unknown>) => void
  ): Promise<void> {
    await streamPostSse("/api/models/download", { model_id }, onEvent);
  },

  listProfiles: () => requestJson<{ profiles: VoiceProfile[] }>("/api/profiles"),

  uploadSourceAudio: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return requestJson<UploadAnalysisResponse>("/api/profiles/upload", {
      method: "POST",
      body: form,
    });
  },

  transcribeSlice: (payload: {
    upload_id: string;
    start_sec: number;
    duration_sec: number;
    language?: string;
  }) =>
    requestJson<{ text: string }>("/api/profiles/transcribe-slice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  createProfile: (payload: {
    name: string;
    upload_id: string;
    start_sec: number;
    duration_sec: number;
    ref_text: string;
    language: string;
    notes: string;
  }) =>
    requestJson<{ profile: VoiceProfile }>("/api/profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  deleteProfile: (profileId: string) =>
    requestJson<{ deleted: string }>(`/api/profiles/${encodeURIComponent(profileId)}`, {
      method: "DELETE",
    }),

  generateSingle: (payload: {
    profile_id: string;
    text: string;
    model_id?: string;
    speed: number;
    temperature: number;
    language: string;
    postprocess: PostProcessOptions;
  }) =>
    requestJson<SingleGenerateResponse>("/api/generate/single", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  async generateBatchStream(
    payload: {
      profile_id: string;
      items: Omit<BatchGapInput, "id">[];
      model_id?: string;
      speed: number;
      temperature: number;
      language: string;
      postprocess: PostProcessOptions;
    },
    onEvent: (evt: Record<string, unknown>) => void
  ): Promise<void> {
    await streamPostSse("/api/generate/batch", payload, onEvent);
  },
};

async function streamPostSse(
  path: string,
  payload: unknown,
  onEvent: (evt: Record<string, unknown>) => void
): Promise<void> {
  const base = await getApiBaseUrl();
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    let detail = `Stream request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (line.startsWith("data: ")) {
          let parsed: Record<string, unknown> | null = null;
          try {
            parsed = JSON.parse(line.slice(6)) as Record<string, unknown>;
          } catch {
            // ignore malformed keepalive
          }
          if (parsed !== null) {
            onEvent(parsed);
          }
        }
      }
    }
  }
}
