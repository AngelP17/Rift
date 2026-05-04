"use client";

import { useRef } from "react";
import Link from "next/link";
import { useGSAP } from "@gsap/react";
import {
  ArrowRight,
  ChartLineUp,
  Database,
  GitBranch,
  Graph,
  LockKey,
  ShieldCheck,
  Sparkle,
  WarningDiamond
} from "@phosphor-icons/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger, useGSAP);

const bentoCards = [
  {
    title: "Device farms become neighborhoods",
    body: "Shared devices, account reuse, merchant clusters, and temporal links stay visible as connected evidence instead of disappearing into flat tables.",
    className: "md:col-span-5",
    image: "https://picsum.photos/seed/rift-graph-topology/1200/800",
    icon: Graph
  },
  {
    title: "Every score leaves a receipt",
    body: "Feature inputs, calibrated probabilities, model run metadata, and durable hashes travel together through the review path.",
    className: "md:col-span-4",
    image: "https://picsum.photos/seed/rift-audit-ledger/1200/800",
    icon: LockKey
  },
  {
    title: "Uncertainty gets routed",
    body: "Conformal bands separate low-risk approvals from cases that deserve a human analyst before money moves.",
    className: "md:col-span-3",
    image: "https://picsum.photos/seed/rift-analyst-desk/1200/800",
    icon: WarningDiamond
  },
  {
    title: "Governance is not a PDF afterthought",
    body: "Model cards, fairness audits, drift reports, and markdown exports are generated from the same local evidence store.",
    className: "md:col-span-7",
    image: "https://picsum.photos/seed/rift-governance-review/1400/900",
    icon: ShieldCheck
  },
  {
    title: "Local stack, serious posture",
    body: "DuckDB, Polars, FastAPI, and Next.js run without managed services, while still feeling like an investigation-grade console.",
    className: "md:col-span-5",
    image: "https://picsum.photos/seed/rift-local-control-room/1200/900",
    icon: Database
  }
];

const scrollItems = [
  {
    title: "Detect coordinated fraud rings",
    body: "GraphSAGE-style embeddings capture the shared infrastructure behind mule accounts, device farms, and merchant collusion.",
    metric: "47.2% recall at 1% FPR"
  },
  {
    title: "Calibrate before decisions",
    body: "Platt and isotonic calibration keep probabilities aligned with observed outcomes before anything reaches an analyst.",
    metric: "3.1% expected calibration error"
  },
  {
    title: "Prove the audit trail",
    body: "Decision payloads, model lineage, and SHA-256 hashes make every intervention replayable from local artifacts.",
    metric: "64-char deterministic hashes"
  }
];

const partners = ["DuckDB", "Polars", "FastAPI", "Next.js", "XGBoost", "PyTorch", "Recharts", "TanStack Table"];

export function LandingExperience() {
  const root = useRef<HTMLElement>(null);
  const pinSection = useRef<HTMLElement>(null);
  const revealText = useRef<HTMLParagraphElement>(null);

  useGSAP(
    () => {
      gsap.from("[data-hero-copy]", {
        opacity: 0,
        y: 28,
        duration: 0.9,
        stagger: 0.12,
        ease: "power3.out"
      });

      gsap.from("[data-bento-card]", {
        opacity: 0,
        scale: 0.92,
        y: 40,
        duration: 0.8,
        stagger: 0.08,
        ease: "power3.out",
        scrollTrigger: {
          trigger: "[data-bento-grid]",
          start: "top 72%"
        }
      });

      if (pinSection.current) {
        ScrollTrigger.create({
          trigger: pinSection.current,
          start: "top top",
          end: "bottom bottom",
          pin: "[data-pinned-story]",
          pinSpacing: false
        });
      }

      if (revealText.current) {
        const words = revealText.current.querySelectorAll("span");
        gsap.to(words, {
          opacity: 1,
          y: 0,
          stagger: 0.08,
          ease: "none",
          scrollTrigger: {
            trigger: revealText.current,
            start: "top 78%",
            end: "bottom 42%",
            scrub: true
          }
        });
      }

      gsap.to("[data-scroll-card]", {
        opacity: 0.28,
        scale: 0.94,
        stagger: 0.08,
        scrollTrigger: {
          trigger: pinSection.current,
          start: "top 20%",
          end: "bottom 70%",
          scrub: true
        }
      });
    },
    { scope: root }
  );

  const revealWords =
    "Rift turns fraud modeling into an accountable operating system: graph intelligence, calibrated uncertainty, and governance records move together from raw transaction to final decision.".split(
      " "
    );

  return (
    <main ref={root} className="w-full max-w-full overflow-x-hidden bg-[#030712] text-ink">
      <nav className="fixed left-1/2 top-5 z-20 w-[min(94vw,1080px)] -translate-x-1/2 rounded-full border border-white/10 bg-slate-950/76 px-4 py-3 shadow-glass backdrop-blur-2xl">
        <div className="flex items-center justify-between gap-4">
          <Link className="flex items-center gap-3 text-sm font-semibold tracking-tight" href="/">
            <span className="grid h-9 w-9 place-items-center rounded-full bg-white text-slate-950">
              <GitBranch size={18} weight="bold" />
            </span>
            <span>Rift</span>
          </Link>
          <div className="hidden items-center gap-6 text-sm text-muted md:flex">
            <a className="transition hover:text-ink" href="#intelligence">
              Intelligence
            </a>
            <a className="transition hover:text-ink" href="#proof">
              Proof
            </a>
            <Link className="transition hover:text-ink" href="/dashboard">
              Dashboard
            </Link>
          </div>
          <Link
            className="rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-slate-950 transition hover:scale-[1.02] active:scale-[0.98]"
            href="/dashboard"
          >
            Open evidence console
          </Link>
        </div>
      </nav>

      <section className="relative min-h-[100dvh] px-4 pb-24 pt-36 md:px-8 md:pb-32 md:pt-44">
        <div className="absolute inset-0 -z-0">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_14%_18%,rgba(110,168,254,0.22),transparent_30%),radial-gradient(circle_at_86%_12%,rgba(47,191,113,0.14),transparent_26%),linear-gradient(135deg,rgba(8,13,27,0.94),rgba(3,7,18,0.98))]" />
          <div className="absolute inset-x-0 top-28 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
          <div className="absolute left-[8vw] top-36 hidden h-[62vh] w-px bg-gradient-to-b from-transparent via-white/15 to-transparent lg:block" />
          <div className="absolute inset-0 bg-[linear-gradient(rgba(110,168,254,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(110,168,254,0.06)_1px,transparent_1px)] bg-[size:96px_96px] opacity-30" />
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#030712]/68 to-[#030712]" />
        </div>

        <div className="relative mx-auto max-w-[1400px]">
          <div className="grid gap-12 lg:grid-cols-[minmax(0,1.02fr)_minmax(430px,0.98fr)] lg:items-end">
            <div className="min-w-0 max-w-6xl">
              <h1
                data-hero-copy
                className="max-w-6xl font-display text-[clamp(3rem,5vw,5.5rem)] font-semibold leading-[0.92] tracking-[-0.07em]"
              >
                Follow the fraud graph until the evidence holds{" "}
                <span
                  className="mx-2 inline-block h-10 w-28 rounded-full bg-cover bg-center align-middle grayscale contrast-125 md:h-12 md:w-36"
                  style={{ backgroundImage: "url(https://picsum.photos/seed/rift-inline-forensic-map/600/300)" }}
                />{" "}
                in replay.
              </h1>
              <p data-hero-copy className="mt-7 max-w-3xl text-lg leading-8 text-muted md:text-xl">
                Rift is a local-first fraud lab where graph signals, calibrated uncertainty, and audit receipts converge into one reviewer-grade surface.
              </p>
              <div data-hero-copy className="mt-9 flex flex-col gap-3 sm:flex-row">
                <Link
                  className="group inline-flex items-center justify-center gap-3 rounded-full bg-white px-7 py-4 text-sm font-semibold text-slate-950 transition hover:scale-[1.02] active:scale-[0.98]"
                  href="/dashboard"
                >
                  Open evidence console
                  <ArrowRight className="transition group-hover:translate-x-1" size={18} weight="bold" />
                </Link>
                <a
                  className="inline-flex items-center justify-center rounded-full border border-white/25 bg-white/[0.08] px-7 py-4 text-sm font-semibold text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.12)] transition hover:border-white/45 hover:bg-white/[0.12] active:scale-[0.98]"
                  href="#intelligence"
                >
                  Inspect the atlas
                </a>
              </div>
            </div>

            <aside data-hero-copy className="glass-edge relative overflow-hidden rounded-[2rem] bg-slate-950/70 p-5 backdrop-blur-2xl">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_15%,rgba(110,168,254,0.2),transparent_30%)]" />
              <div className="relative mb-5 flex items-center justify-between gap-4">
                <div>
                  <h2 className="font-display text-3xl font-semibold tracking-[-0.05em]">DEC_84AF31</h2>
                  <p className="mt-2 text-sm leading-6 text-muted">A replayable fraud path with model lineage attached.</p>
                </div>
                <LockKey className="h-5 w-5 text-emerald-300" weight="duotone" />
              </div>

              <div className="relative h-[390px] overflow-hidden rounded-[1.5rem] border border-white/10 bg-[radial-gradient(circle_at_50%_42%,rgba(110,168,254,0.15),transparent_34%),linear-gradient(135deg,rgba(15,23,42,0.9),rgba(2,6,23,0.96))]">
                <svg className="absolute inset-0 h-full w-full" role="img" viewBox="0 0 640 420">
                  <title>Fraud graph evidence map</title>
                  <defs>
                    <linearGradient id="edge-gradient" x1="0" x2="1" y1="0" y2="1">
                      <stop offset="0%" stopColor="#6ea8fe" stopOpacity="0.7" />
                      <stop offset="100%" stopColor="#2fbf71" stopOpacity="0.28" />
                    </linearGradient>
                  </defs>
                  {[
                    [118, 112, 298, 186],
                    [298, 186, 470, 96],
                    [298, 186, 506, 278],
                    [118, 112, 190, 312],
                    [190, 312, 506, 278],
                    [298, 186, 190, 312],
                    [470, 96, 506, 278]
                  ].map(([x1, y1, x2, y2], index) => (
                    <line
                      key={`${x1}-${y1}-${index}`}
                      stroke="url(#edge-gradient)"
                      strokeDasharray={index % 2 ? "6 10" : undefined}
                      strokeWidth="1.4"
                      x1={x1}
                      x2={x2}
                      y1={y1}
                      y2={y2}
                    />
                  ))}
                  {[
                    [118, 112, "device"],
                    [298, 186, "txn"],
                    [470, 96, "merchant"],
                    [506, 278, "account"],
                    [190, 312, "user"]
                  ].map(([x, y, label]) => (
                    <g key={label}>
                      <circle cx={x as number} cy={y as number} fill="rgba(255,255,255,0.08)" r="34" stroke="rgba(255,255,255,0.16)" />
                      <circle cx={x as number} cy={y as number} fill={label === "txn" ? "#ffffff" : "#6ea8fe"} r={label === "txn" ? "8" : "6"} />
                      <text fill="rgba(243,246,255,0.72)" fontFamily="JetBrains Mono" fontSize="11" x={(x as number) - 24} y={(y as number) + 52}>
                        {String(label).toUpperCase()}
                      </text>
                    </g>
                  ))}
                </svg>
                <div className="absolute bottom-4 left-4 right-4 grid gap-3 sm:grid-cols-3">
                  {[
                    ["Risk", "0.782"],
                    ["Band", "Review"],
                    ["Trail", "6 hops"]
                  ].map(([label, value]) => (
                    <div className="border-l border-white/15 bg-slate-950/64 px-3 py-2" key={label}>
                      <p className="text-xs text-muted">{label}</p>
                      <p className="mt-1 font-mono text-lg font-semibold">{value}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="relative mt-5 grid grid-cols-2 gap-3">
                {[
                  ["PR-AUC", "91.2%"],
                  ["ECE", "0.031"],
                  ["Reviewer load", "6.8%"],
                  ["Audit rows", "18,472"]
                ].map(([label, value]) => (
                  <div className="border-t border-white/10 bg-white/[0.025] p-4" key={label}>
                    <p className="text-xs text-muted">{label}</p>
                    <p className="mt-2 font-mono text-2xl font-semibold">{value}</p>
                  </div>
                ))}
              </div>
            </aside>
          </div>
        </div>
      </section>

      <section id="intelligence" className="px-4 py-32 md:px-8 md:py-48">
        <div className="mx-auto max-w-[1400px]">
          <div className="mb-14 max-w-4xl">
            <h2 className="font-display text-[clamp(2.5rem,4vw,4.5rem)] font-semibold leading-none tracking-[-0.06em]">
              Built for messy fraud evidence, not tidy dashboard theater.
            </h2>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-muted">
              The interface keeps the operational story intact: where the data came from, why the model reacted, and how reviewers can replay the decision.
            </p>
          </div>

          <div data-bento-grid className="grid-flow-dense grid gap-5 md:grid-cols-12">
            {bentoCards.map((card) => (
              <article
                className={`group glass-edge overflow-hidden rounded-[2.5rem] bg-slate-950/72 p-5 ${card.className}`}
                data-bento-card
                key={card.title}
              >
                <div className="relative h-56 overflow-hidden rounded-[2rem]">
                  <div
                    aria-hidden="true"
                    className="h-full w-full bg-cover bg-center opacity-90 grayscale contrast-125 transition-transform duration-700 ease-out group-hover:scale-105"
                    style={{ backgroundImage: `url(${card.image})` }}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950/82 to-transparent" />
                  <card.icon className="absolute bottom-5 left-5 text-white" size={28} weight="duotone" />
                </div>
                <h3 className="mt-6 font-display text-3xl font-semibold tracking-[-0.05em]">{card.title}</h3>
                <p className="mt-3 max-w-xl text-sm leading-7 text-muted">{card.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="overflow-hidden border-y border-white/10 py-8">
        <div className="marquee-track flex w-max gap-5 text-5xl font-semibold tracking-[-0.06em] text-white/16">
          {[...partners, ...partners].map((item, index) => (
            <span className="px-5" key={`${item}-${index}`}>
              {item}
            </span>
          ))}
        </div>
        <div className="marquee-track-reverse mt-3 flex w-max gap-5 text-5xl font-semibold tracking-[-0.06em] text-white/10">
          {[...partners, ...partners].map((item, index) => (
            <span className="px-5" key={`${item}-reverse-${index}`}>
              {item}
            </span>
          ))}
        </div>
      </section>

      <section ref={pinSection} id="proof" className="relative px-4 py-32 md:px-8 md:py-48">
        <div className="mx-auto grid max-w-[1400px] gap-10 lg:grid-cols-[0.85fr_1.15fr]">
          <div data-pinned-story className="h-fit lg:sticky lg:top-32">
            <p className="mb-6 flex items-center gap-3 text-sm uppercase tracking-[0.3em] text-muted">
              <Sparkle size={18} weight="duotone" />
              Evidence in motion
            </p>
            <h2 className="font-display text-[clamp(2.4rem,4vw,4.6rem)] font-semibold leading-none tracking-[-0.06em]">
              Scroll through the decision chain.
            </h2>
            <p ref={revealText} className="mt-8 max-w-xl text-xl leading-9 text-muted">
              {revealWords.map((word, index) => (
                <span className="inline-block translate-y-2 pr-1 opacity-10" key={`${word}-${index}`}>
                  {word}
                </span>
              ))}
            </p>
          </div>

          <div className="space-y-6">
            {scrollItems.map((item, index) => (
              <article
                className="glass-edge min-h-[360px] rounded-[2.5rem] bg-slate-950/76 p-8 md:p-10"
                data-scroll-card
                key={item.title}
                style={{ marginTop: index ? "-1.5rem" : 0 }}
              >
                <div className="mb-12 flex h-14 w-14 items-center justify-center rounded-2xl bg-white text-slate-950">
                  <ChartLineUp size={24} weight="bold" />
                </div>
                <h3 className="max-w-2xl font-display text-4xl font-semibold tracking-[-0.05em]">{item.title}</h3>
                <p className="mt-5 max-w-2xl text-lg leading-8 text-muted">{item.body}</p>
                <p className="mt-10 font-mono text-3xl font-semibold text-white">{item.metric}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="px-4 pb-12 pt-20 md:px-8 md:pb-16">
        <div className="mx-auto max-w-[1400px] overflow-hidden rounded-[3rem] bg-white p-8 text-slate-950 md:p-12">
          <div className="grid gap-10 lg:grid-cols-[1fr_0.55fr] lg:items-end">
            <div>
              <h2 className="max-w-5xl font-display text-[clamp(2.5rem,5vw,5.2rem)] font-semibold leading-[0.93] tracking-[-0.07em]">
                Put the evidence console in front of reviewers.
              </h2>
              <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600">
                Run the local stack, generate realistic transactions, train the hybrid model, and open a dashboard that explains itself.
              </p>
            </div>
            <div className="flex flex-col gap-3">
              <Link
                className="inline-flex items-center justify-center gap-3 rounded-full bg-slate-950 px-7 py-4 text-sm font-semibold text-white transition hover:scale-[1.02] active:scale-[0.98]"
                href="/dashboard"
              >
                Launch console
                <ArrowRight size={18} weight="bold" />
              </Link>
              <a
                className="inline-flex items-center justify-center rounded-full border border-slate-200 px-7 py-4 text-sm font-semibold text-slate-950 transition hover:bg-slate-100 active:scale-[0.98]"
                href="https://github.com/AngelP17/Rift"
              >
                View repository
              </a>
            </div>
          </div>
          <footer className="mt-16 flex flex-col gap-4 border-t border-slate-200 pt-6 text-sm text-slate-500 md:flex-row md:items-center md:justify-between">
            <span>Rift fraud intelligence platform</span>
            <span>Local-first ML, governance, replay, and audit reporting</span>
          </footer>
        </div>
      </section>
    </main>
  );
}
