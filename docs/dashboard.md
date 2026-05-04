# Operations Dashboard

The Rift Operations Dashboard is the governance-focused UI layer that surfaces platform health, model performance, audit lineage, and compliance signals at a glance.

## Access

| Method | URL | Description |
|---|---|---|
| Server-rendered HTML | `GET /dashboard` | Full operations dashboard |
| Landing page | `GET /` | Hero landing with key metrics |
| JSON API | `GET /dashboard/summary` | Raw snapshot data as JSON |
| Next.js frontend | `http://localhost:3000` | React landing page and dashboard |

Start the server:

```bash
rift dashboard --port 8000
# or
python -m rift.cli.main dashboard --host 127.0.0.1 --port 8000
```

Then visit `http://localhost:8000/dashboard`.

For the React frontend:

```bash
cd frontend
npm install
npm run dev
```

Then visit `http://localhost:3000` for the GSAP-powered landing page or `http://localhost:3000/dashboard` for the React operations console.

If the local Docker stack is running, Grafana also uses `localhost:3000`. In that case, start the frontend on a different port:

```bash
npm run dev -- --port 3001
```

## Architecture

```mermaid
flowchart TD
    subgraph backend ["FastAPI Backend"]
        SNAP["dashboard_snapshot"] --> KPIS["KPI card builder"]
        SNAP --> TABLES["Data table payloads"]
        SNAP --> EXPORTS["Model card and audit exports"]
        KPIS --> RENDER["build_dashboard_html"]
        TABLES --> RENDER
        RENDER --> HTML["Server-rendered HTML"]
    end

    subgraph static ["Static Assets"]
        CSS["dashboard.css"] --> HTML
    end

    subgraph routes ["Dashboard Routes"]
        HTML --> DASH["GET /dashboard"]
        HTML --> LAND["GET /"]
        SNAP --> API["GET /dashboard/summary"]
        EXPORTS --> DL["GET /dashboard/export/*"]
    end

    subgraph frontend ["Next.js Frontend - Optional"]
        REACT["React components"] --> FETCH["fetch /dashboard/summary"]
        REACT --> METRICS["fetch /metrics/latest"]
        FETCH --> API
        METRICS --> MAPI["GET /metrics/latest"]
    end

    CLIENT["Browser"] --> DASH
    CLIENT --> REACT
    CLIENT --> DL

    classDef backend fill:#10243f,stroke:#6ea8fe,color:#f3f6ff
    classDef route fill:#12351f,stroke:#2fbf71,color:#f3f6ff
    classDef frontend fill:#0f172a,stroke:#6ea8fe,color:#f3f6ff
    classDef export fill:#3a1f12,stroke:#f39c12,color:#f3f6ff
    class SNAP,KPIS,TABLES,RENDER,HTML,CSS backend
    class DASH,LAND,API,MAPI route
    class REACT,FETCH,METRICS,CLIENT frontend
    class EXPORTS,DL export
```

The dashboard is built with a hybrid rendering approach:

- **Server-rendered HTML** in `src/rift/dashboard/views.py` generates the full dashboard and landing page using Python f-strings with `html.escape()` for XSS safety.
- **Static CSS** in `src/rift/dashboard/static/dashboard.css` provides the enterprise dark theme with CSS custom properties.
- **Next.js frontend** in `frontend/` provides a React-based landing page and dashboard with Tailwind CSS, charts (Recharts), GSAP scroll interactions, Phosphor icons, and animated components.
- **Fallback demo telemetry** in `frontend/lib/mock-dashboard.ts` keeps the React dashboard populated for demos and screenshots when the FastAPI API is unavailable locally. Production deployments should connect the frontend to the FastAPI service.

### Key Files

| File | Purpose |
|---|---|
| `src/rift/dashboard/views.py` | Dashboard snapshot collection, HTML rendering, landing page |
| `src/rift/dashboard/kpis.py` | Centralized KPI threshold logic with color-coded status bands |
| `src/rift/dashboard/__init__.py` | Package exports |
| `src/rift/dashboard/static/dashboard.css` | Dark theme CSS with responsive breakpoints |
| `src/rift/api/server.py` | FastAPI routes for dashboard, exports, predictions |
| `frontend/` | Optional Next.js/React dashboard |

## KPI Cards

```mermaid
flowchart LR
    RAW["Raw metric value"] --> EVAL["Threshold evaluator"]
    EVAL -->|"meets green band"| G["Good"]
    EVAL -->|"inside warning band"| Y["Warning"]
    EVAL -->|"outside tolerance"| R["Critical"]
    G --> CARD["KPI card component"]
    Y --> CARD
    R --> CARD
    CARD --> DASH["Dashboard render"]
    CARD --> SUMMARY["JSON summary consumer"]

    classDef input fill:#0f172a,stroke:#6ea8fe,color:#f3f6ff
    classDef good fill:#12351f,stroke:#2fbf71,color:#f3f6ff
    classDef warn fill:#3a1f12,stroke:#f39c12,color:#f3f6ff
    classDef bad fill:#3a1212,stroke:#e74c3c,color:#f3f6ff
    class RAW,EVAL,CARD,SUMMARY,DASH input
    class G good
    class Y warn
    class R bad
```

KPI cards display governance and model health metrics with color-coded status:

| Metric | Green | Yellow | Red | Direction |
|---|---|---|---|---|
| PR-AUC | >= 0.85 | >= 0.70 | < 0.70 | Higher is better |
| ECE | <= 0.05 | <= 0.10 | > 0.10 | Lower is better |
| Brier Score | <= 0.12 | <= 0.20 | > 0.20 | Lower is better |
| Recall@1%FPR | >= 0.60 | >= 0.40 | < 0.40 | Higher is better |

Count-based KPIs (ETL runs, fairness audits, drift reports, federated runs, recorded audits) use the accent color.

Thresholds are centrally defined in `src/rift/dashboard/kpis.py` and can be modified without touching templates.

## Data Tables

The dashboard shows six data tables:

1. **Latest ETL Runs** -- source system, valid/invalid rows, duplicates removed
2. **Recent Fairness Audits** -- sensitive column, demographic parity, disparity ratio
3. **Recent Drift Reports** -- drift score, is_drift flag, retrain trigger
4. **Federated Training Runs** -- client column, client count, rounds
5. **Prepared Public Datasets** -- adapter, rows prepared, ETL run ID
6. **Recent Audit Decisions** -- decision ID, model run, outcome, calibrated probability, confidence

Each table shows guided empty states with CLI commands when no records exist.

## API Routes

### Dashboard and Landing Routes

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/` | Landing page (HTML) |
| GET | `/dashboard` | Full operations dashboard (HTML) |
| GET | `/dashboard/summary` | JSON snapshot |

### Export Routes

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/dashboard/export/model-card` | Download latest model card as markdown |
| GET | `/dashboard/export/audit` | Download latest audit report as markdown |

### Prediction and Audit Routes

| Method | Endpoint | Returns |
|---|---|---|
| POST | `/predict` | Score a transaction and record decision |
| GET | `/replay/{decision_id}` | Replay a past decision |
| GET | `/audit/{decision_id}` | Get audit report for a decision |
| GET | `/metrics/latest` | Latest model metrics |
| GET | `/models/current` | Current model info |

### Governance and Monitoring Routes

| Method | Endpoint | Returns |
|---|---|---|
| POST | `/governance/model-card/{run_id}` | Generate model card for a run |
| GET | `/fairness/status` | Recent fairness audits |
| GET | `/monitor/drift-status` | Recent drift reports |
| GET | `/query?natural=...` | Natural language query |
| GET | `/etl/status` | Recent ETL runs |
| GET | `/datasets/status` | Prepared datasets |
| GET | `/federated/status` | Federated training runs |
| GET | `/storage/status` | Storage backend info |
| GET | `/lakehouse/status` | Lakehouse DB path |
| GET | `/lakehouse/query?sql=...` | Run SQL against lakehouse |
| GET | `/health` | Health check |

## Next.js Frontend (Optional)

The `frontend/` directory contains an optional React-based dashboard built with:

- **Next.js 14** with App Router
- **Tailwind CSS** for styling
- **Recharts** for performance trend and operations breakdown charts
- **GSAP + ScrollTrigger** for landing page scroll choreography
- **Phosphor Icons** for consistent iconography
- **Animated number components** for KPI transitions

To run:

```bash
cd frontend
npm install
npm run dev
```

The frontend connects to the same FastAPI backend at `http://localhost:8000`. When the backend is offline, realistic demo telemetry is used as fallback data so the dashboard remains populated for screenshots and demo walkthroughs. This fallback is a preview aid; production use should connect the API and treat `/dashboard/summary` and `/metrics/latest` as the source of truth.

To verify:

```bash
cd frontend
npm run lint
npm run build
```

### Visual QA Gates

Use these gates for future landing-page or dashboard presentation work:

- Keep the first viewport readable on a small laptop: short hero copy, clear CTA, and no crowded card stacks above the fold.
- Preserve the current image-led landing direction with large section-specific imagery and stable media frames.
- Keep `GSAP` and `ScrollTrigger` interactions real and purposeful; avoid decorative motion that hides content or creates horizontal scroll.
- Bento grids should keep `grid-flow-dense` and span math that avoids intentional empty cells.
- Avoid cheap meta labels, fake technical pills, excessive microcopy, nested cards, and giant wrapper panels unless they materially clarify the workflow.
- Verify button contrast in both light and dark CTA areas.
- Keep fallback demo telemetry realistic and populated when the FastAPI backend is offline, but do not show a large warning banner in screenshots.

Cheap/meta patterns to remove:

- Decorative section markers, fake runtime tags, ornamental status chips, and brand-style labels that do not map to real product state.
- Unsupported README badges or claims that are not backed by current CI output.
- Repeated pill rows where plain text, table labels, or icon buttons would be clearer.

Acceptable operational labels:

- Labels that identify actual data, routes, filters, chart series, refresh state, API availability, audit lineage, or model/governance metadata.

## Customization

### Changing Thresholds

Edit `src/rift/dashboard/kpis.py`:

```python
THRESHOLDS = {
    "pr_auc": {"green": 0.85, "yellow": 0.70, "lower_is_better": False},
    "ece": {"green": 0.05, "yellow": 0.10, "lower_is_better": True},
}
```

### Adding Quick Actions

Edit `QUICK_ACTIONS` in `src/rift/dashboard/kpis.py`:

```python
QUICK_ACTIONS = [
    ActionLink("Run Prediction", "/predict"),
    ActionLink("Latest Model Card", "/governance/model-card/latest"),
    ...
]
```
