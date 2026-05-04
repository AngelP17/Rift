# Rift Agent Guide

Use this file as the first stop for future Codex sessions. Keep changes grounded in the repo; do not invent commands, services, or conventions that are not present in code, scripts, manifests, or docs.

## Project Shape

- Python package: `src/rift/`
- Tests: `tests/`
- Optional React frontend: `frontend/`
- Documentation: `README.md`, `docs/`, `AUDIT_GUIDE.md`, `CONTRIBUTING.md`
- Local orchestration: `docker-compose.yml`, `docker/`, `dags/`, `observability/`
- Utility scripts: `scripts/`
- Sector configs: `configs/sectors/`
- Generated runtime artifacts: `.rift/`

## Setup

Backend/local CLI:

```bash
python3 -m pip install -e ".[dev]"
```

Optional local stack dependencies:

```bash
python3 -m pip install -e ".[dev,local-stack]"
python3 -m pip install -e ".[dev,local-stack,advanced]"
```

Frontend:

```bash
cd frontend
npm install
```

## Run Commands

CLI demo:

```bash
rift generate --txns 5000 --users 500 --merchants 120 --fraud-rate 0.03
rift train --model graphsage_xgb --time-split
rift predict --tx demo/sample_transaction.json
```

FastAPI dashboard:

```bash
rift dashboard --host 127.0.0.1 --port 8000
```

Next.js frontend:

```bash
cd frontend
npm run dev
```

Local orchestration stack:

```bash
./scripts/init_local_stack.sh
```

## Verification Commands

Python tests:

```bash
python3 -m pytest -q
```

Python lint, matching CI:

```bash
ruff check src/ tests/
```

Full repo validation gate:

```bash
./scripts/ci_validate.sh
```

Local stack validation:

```bash
./scripts/validate_local_stack.sh
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Release verification, when checking published artifacts:

```bash
./scripts/verify_release.sh <version>
./scripts/verify_release.sh <version> --bundle <path>
```

## CI

GitHub Actions are defined in `.github/workflows/ci.yml`, `.github/workflows/validate.yml`, and `.github/workflows/release.yml`.

- CI installs Python 3.12.
- Python lint uses `ruff check src/ tests/`.
- Python tests run with `pytest tests/ -v --tb=short`.
- The validation workflow also runs `./scripts/ci_validate.sh`.

## Frontend Quality Gates

The frontend is a Next.js 14 App Router app using Tailwind CSS, Recharts, Framer Motion, GSAP + ScrollTrigger, Phosphor icons, and SWR. The current landing page in `frontend/components/landing/landing-experience.tsx` already follows several high-end visual gates: Outfit typography, image-led sections, GSAP scroll choreography, `grid-flow-dense` bento layout, inline typography imagery, marquee motion, and generous section spacing.

For future visual frontend changes, apply these hard QA gates before considering the task complete:

- Keep the first viewport clean on small laptops: short hero headline, readable supporting copy, obvious primary CTA, no overcrowded cards above the fold.
- Preserve `main` overflow protection for animated pages: `overflow-x-hidden w-full max-w-full`.
- Prefer large, readable, section-specific imagery over tiny decorative thumbnails.
- For visually important redesigns, generate or obtain section-level design references first, analyze typography, spacing, color, imagery, and controls, then implement.
- Avoid fake technical pills, cheap meta labels, and decorative microcopy unless it directly helps the product story.
- Avoid nested cards and giant wrapper panels unless they clarify a real tool surface.
- Bento grids must use dense flow and span math that leaves no intentional empty cells.
- Clickable cards and image modules should have visible hover motion inside stable, overflow-hidden frames.
- If adding GSAP, use real `@gsap/react`/`ScrollTrigger` behavior and verify reduced-motion handling remains acceptable.

## Conventions and Constraints

- Do not change application behavior during documentation-only tasks.
- Derive docs from existing source, tests, scripts, manifests, and CI.
- Keep the local-first, open-source, zero-cost architecture intact unless the user explicitly asks otherwise.
- Keep optional integrations optional; do not make paid cloud services required for core workflows.
- Update docs when CLI commands, API routes, generated artifacts, audit output, dashboard behavior, or workflow steps change.
- Use Mermaid for diagrams in Markdown; do not add ASCII art diagrams.
- Use `python3 -m rift.cli.main ...` in scripts when matching existing scripts; user-facing docs may use the installed `rift` command after editable install.
- Treat `.rift/`, `frontend/.next/`, `frontend/node_modules/`, `.pytest_cache/`, Docker volumes, logs, and generated screenshots/artifacts as generated output. Do not hand-edit them.
- Avoid committing secrets or local service credentials. Docker Compose contains local demo credentials only.

## Done When

A task is done when:

- The requested behavior or documentation change is complete and scoped.
- Relevant tests, lint, type checks, builds, or scripts have been run where feasible.
- Any skipped verification is explicitly explained.
- Generated files and unrelated worktree changes are left alone.
- Documentation is updated for any changed command, workflow, artifact, API route, or frontend quality expectation.
- The final response lists files changed, verification run, results, blockers, and useful follow-up.

## Known Friction

- `docker-compose.yml` maps Grafana to port `3000`, which conflicts with the default Next.js dev server. If both are needed, run one on a different port.
- The frontend can display realistic demo telemetry when the FastAPI backend is offline; this is intentional for visual QA and screenshots.
