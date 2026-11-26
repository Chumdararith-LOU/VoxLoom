import { useEffect, useRef } from "react";
import { ShieldCheck } from "lucide-react";
import { attachLightDismissFallback } from "../lib/domUtils";

interface ConsentDialogProps {
  open: boolean;
  onAccept: () => void;
}

export function ConsentDialog({ open, onAccept }: ConsentDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  return (
    <dialog
      ref={dialogRef}
      className="studio-dialog"
      aria-labelledby="consent-dialog-title"
      {...({ closedby: "none" } as Record<string, string>)}
      onCancel={(e) => e.preventDefault()}
    >
      <div className="dialog-body">
        <h1 id="consent-dialog-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <ShieldCheck size={20} style={{ color: "var(--accent)" }} />
          Responsible Local Voice Cloning
        </h1>
        <p className="deck-sub">
          VoxLoom runs 100% locally on your hardware — your voice recordings and generated audio
          never leave your computer.
        </p>
        <p style={{ margin: 0, fontSize: 13 }}>
          Before continuing, confirm that you will only clone your own voice or voices for which you
          have explicit permission, and will not impersonate others without consent.
        </p>
      </div>
      <div className="dialog-actions">
        <button
          type="button"
          className="btn btn-primary"
          autoFocus
          onClick={() => {
            dialogRef.current?.close("accept");
            onAccept();
          }}
        >
          I understand, open studio
        </button>
      </div>
    </dialog>
  );
}

interface ConfirmDeleteDialogProps {
  profileName: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDeleteDialog({
  profileName,
  onConfirm,
  onCancel,
}: ConfirmDeleteDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const detach = attachLightDismissFallback(dialog);
    const onClose = () => {
      const value = dialog.returnValue;
      dialog.returnValue = "";
      if (value === "delete") {
        onConfirm();
      } else {
        onCancel();
      }
    };
    dialog.addEventListener("close", onClose);
    return () => {
      detach();
      dialog.removeEventListener("close", onClose);
    };
  }, [onConfirm, onCancel]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (profileName && !dialog.open) {
      dialog.showModal();
    } else if (!profileName && dialog.open) {
      dialog.close();
    }
  }, [profileName]);

  return (
    <dialog
      ref={dialogRef}
      className="studio-dialog"
      aria-labelledby="delete-profile-title"
      {...({ closedby: "any" } as Record<string, string>)}
    >
      <form method="dialog">
        <div className="dialog-body">
          <h1 id="delete-profile-title">Delete voice profile?</h1>
          <p className="deck-sub">
            Remove <strong>{profileName}</strong> and its sliced reference audio (<code>ref.wav</code>)
            from local storage. This action cannot be undone.
          </p>
        </div>
        <div className="dialog-actions">
          <button type="submit" value="cancel" className="btn" autoFocus>
            Cancel
          </button>
          <button type="submit" value="delete" className="btn btn-danger">
            Delete profile
          </button>
        </div>
      </form>
    </dialog>
  );
}
