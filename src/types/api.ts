export interface HardwareProfile {
  os_name: string;
  os_version: string;
  arch: string;
  ram_gb: number;
  recommended_backend: "mlx" | "cuda" | "cpu";
  available_backends: string[];
  accelerator_name: string;
  vram_gb: number | null;
  mlx_audio_installed: boolean;
  torch_installed: boolean;
  cuda_available: boolean;
  ffmpeg_available: boolean;
  recommended_tts_model: string;
  recommended_stt_model: string;
}

export interface EngineStatus {
  loaded: boolean;
  model_id: string | null;
  repo_id: string | null;
  backend: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  data_dir: string;
  engine: EngineStatus;
}

export interface ModelItem {
  id: string;
  name: string;
  kind: "tts" | "stt";
  backend: "mlx" | "torch";
  repo_id: string;
  size_gb: number;
  min_ram_gb: number;
  description: string;
  license: string;
  compatible: boolean;
  recommended: boolean;
  downloaded: boolean;
  cached_size_gb: number;
  loaded: boolean;
}

export interface AudioMetadata {
  duration: number;
  sample_rate: number;
  channels: number;
  codec: string;
  size_bytes: number;
  mean_volume_db: number | null;
  max_volume_db: number | null;
}

export interface WaveformData {
  duration: number;
  peaks: number[];
  silence_mask: boolean[];
}

export interface SuggestedSegment {
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  score: number;
  reason: string;
}

export interface UploadAnalysisResponse {
  upload_id: string;
  filename: string;
  metadata: AudioMetadata;
  waveform: WaveformData;
  suggestion: SuggestedSegment;
}

export interface VoiceProfile {
  id: string;
  name: string;
  ref_text: string;
  ref_audio_path: string;
  duration_sec: number;
  language: string;
  created_at: string;
  notes: string;
}

export interface PostProcessOptions {
  trim_silence: boolean;
  pad_sec: number;
  loudnorm: boolean;
  target_lufs: number;
  sample_rate: number;
  channels: number;
  max_atempo: number;
}

export interface PostProcessResult {
  output_path: string;
  raw_duration: number;
  trimmed_duration: number;
  final_duration: number;
  atempo_applied: number;
  status: "PASS" | "PASS_ADJUSTED" | "FLAGGED";
  needs_regeneration: boolean;
}

export interface SingleGenerateResponse {
  job_id: string;
  filename: string;
  audio_url: string;
  result: PostProcessResult;
}

export interface BatchGapInput {
  id: string;
  filename: string;
  timeline_start: string;
  max_duration: number | null;
  text: string;
}

export interface BatchGapResult {
  index: number;
  filename: string;
  timeline_start: string;
  text: string;
  max_duration: number | null;
  actual_duration: number;
  trimmed_duration: number;
  atempo_applied: number;
  status: "PASS" | "PASS_ADJUSTED" | "FLAGGED";
  regenerated: boolean;
  audio_url: string;
}

export interface ToastMessage {
  id: string;
  text: string;
  tone: "info" | "success" | "warning" | "error";
}
