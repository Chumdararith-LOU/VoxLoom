import { X } from "lucide-react";
import type { ToastMessage } from "../types/api";

interface ToastRegionProps {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

export function ToastRegion({ toasts, onDismiss }: ToastRegionProps) {
  return (
    <div role="region" aria-label="Notifications" className="toast-region">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          id={`toast-${toast.id}`}
          className={`toast-item ${toast.tone}`}
          role="status"
        >
          <span>{toast.text}</span>
          <button
            type="button"
            className="btn-icon"
            aria-label="Dismiss notification"
            onClick={() => onDismiss(toast.id)}
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
