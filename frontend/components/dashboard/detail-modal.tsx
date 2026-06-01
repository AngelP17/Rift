"use client";

import { AnimatePresence, motion } from "framer-motion";
import { X } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";

type DetailModalProps = {
  title: string;
  open: boolean;
  onClose: () => void;
  payload: unknown;
};

export function DetailModal({ title, open, onClose, payload }: DetailModalProps) {
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();
    document.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = "";
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open ? (
        <div role="dialog" aria-modal="true" aria-label={title}>
          <motion.button
            aria-label="Close detail modal"
            className="fixed inset-0 z-40 bg-slate-950/72 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            type="button"
          />
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 18, scale: 0.96 }}
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
            className="surface fixed left-1/2 top-1/2 z-50 flex w-[min(92vw,760px)] -translate-x-1/2 -translate-y-1/2 flex-col rounded-[32px] border border-[color:var(--color-line)] p-6 shadow-glass"
          >
            <div className="mb-5 flex items-center justify-between gap-4">
              <div>
                <h3 className="font-display text-3xl tracking-[-0.05em] text-ink">{title}</h3>
                <p className="mt-1 text-sm text-muted">Drill-down payload from the current Rift snapshot.</p>
              </div>
              <button
                aria-label="Close"
                className="rounded-full border border-[color:var(--color-line)] bg-white/[0.03] p-3 text-muted transition hover:border-[color:var(--color-line-strong)] hover:text-ink"
                onClick={onClose}
                ref={closeButtonRef}
                type="button"
              >
                <X className="h-4 w-4" weight="bold" />
              </button>
            </div>
            <pre className="thin-scrollbar max-h-[60vh] overflow-auto rounded-[24px] bg-slate-950/80 p-5 font-mono text-xs leading-6 text-slate-100">
              {JSON.stringify(payload, null, 2)}
            </pre>
          </motion.div>
        </div>
      ) : null}
    </AnimatePresence>
  );
}
