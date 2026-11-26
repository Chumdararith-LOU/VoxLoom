/**
 * Progressive-enhancement DOM helpers guided by modern web platform standards:
 * - Light-dismiss fallback for <dialog closedby="any"> (required for macOS WKWebView / Safari)
 * - aria-invalid + aria-describedby synchronization with :user-invalid
 * - ariaNotify() with polite ARIA live-region fallback
 */

export function attachLightDismissFallback(dialog: HTMLDialogElement): () => void {
  if ("closedBy" in HTMLDialogElement.prototype) {
    return () => {};
  }

  const isOutside = (e: PointerEvent) => {
    const r = dialog.getBoundingClientRect();
    return (
      e.clientX < r.left ||
      e.clientX > r.right ||
      e.clientY < r.top ||
      e.clientY > r.bottom
    );
  };

  let pressedOutside = false;
  const onDown = (e: PointerEvent) => {
    pressedOutside = isOutside(e);
  };
  const onUp = (e: PointerEvent) => {
    if (pressedOutside && isOutside(e)) {
      dialog.close();
    }
    pressedOutside = false;
  };

  dialog.addEventListener("pointerdown", onDown);
  dialog.addEventListener("pointerup", onUp);

  return () => {
    dialog.removeEventListener("pointerdown", onDown);
    dialog.removeEventListener("pointerup", onUp);
  };
}

let liveRegionEl: HTMLDivElement | null = null;

export function announceStatus(message: string): void {
  if (typeof document === "undefined") return;

  // Use native ariaNotify when supported
  const docWithNotify = document as Document & {
    ariaNotify?: (text: string, opts?: { priority?: "normal" | "high" }) => void;
  };
  if (typeof docWithNotify.ariaNotify === "function") {
    docWithNotify.ariaNotify(message, { priority: "normal" });
    return;
  }

  // Fallback: persistent polite ARIA live region present in DOM
  if (!liveRegionEl) {
    liveRegionEl = document.createElement("div");
    liveRegionEl.id = "voxloom-sr-announcer";
    liveRegionEl.setAttribute("aria-live", "polite");
    liveRegionEl.setAttribute("aria-atomic", "true");
    liveRegionEl.className = "visually-hidden";
    document.body.append(liveRegionEl);
  }
  liveRegionEl.textContent = "";
  window.setTimeout(() => {
    if (liveRegionEl) liveRegionEl.textContent = message;
  }, 60);
}

/**
 * Syncs aria-invalid and data-error -> aria-describedby with CSS :user-invalid
 * so assistive technologies announce errors only after user interaction.
 */
export function initAriaValidationSync(root: Document | HTMLElement = document): () => void {
  const linkError = (el: HTMLElement, shown: boolean) => {
    const errorId = el.dataset.error;
    if (!errorId) return;
    const ids = (el.getAttribute("aria-describedby") || "")
      .split(/\s+/)
      .filter((id) => id && id !== errorId);
    if (shown) ids.push(errorId);
    if (ids.length) el.setAttribute("aria-describedby", ids.join(" "));
    else el.removeAttribute("aria-describedby");
  };

  const supportsUserInvalid =
    typeof CSS !== "undefined" && CSS.supports("selector(:user-invalid)");

  const sync = (target: EventTarget | null, forceCheck = false) => {
    if (!(target instanceof HTMLElement)) return;
    if (!target.matches("input, select, textarea")) return;
    const field = target as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;

    const invalid = supportsUserInvalid
      ? field.matches(":user-invalid")
      : forceCheck && !field.validity.valid;

    if (!supportsUserInvalid && forceCheck) {
      field.classList.toggle("user-invalid-fallback", invalid);
    }

    field.setAttribute("aria-invalid", invalid ? "true" : "false");
    linkError(field, invalid);
  };

  const onBlur = (e: Event) => sync(e.target, true);
  const onInput = (e: Event) => sync(e.target, false);
  const onInvalid = (e: Event) => sync(e.target, true);

  root.addEventListener("blur", onBlur, true);
  root.addEventListener("input", onInput, true);
  root.addEventListener("invalid", onInvalid, true);

  return () => {
    root.removeEventListener("blur", onBlur, true);
    root.removeEventListener("input", onInput, true);
    root.removeEventListener("invalid", onInvalid, true);
  };
}
