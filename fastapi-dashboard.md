# Pattern: FastAPI Local Dashboard + Dark UI

A single-user local web dashboard: FastAPI serves a thin JSON API plus a
server-rendered HTML shell, and the page does its own fetching with vanilla JS
and Chart.js. No frontend build step, no SPA framework, no npm — the entire UI
is a handful of `.html` files with an inline `<style>` block and a `<script>`
tag pulling Chart.js from a CDN.

Use this when you want a polished, chart-heavy dashboard over a local database
(personal data, an internal tool, a homelab service) and a React/Vite toolchain
would be more ceremony than the project is worth. It scales comfortably to a few
dozen routes and a handful of pages. It is *not* the right pattern for
multi-user apps with auth, shared state, or a large interactive surface — reach
for a real frontend framework there.

**Why server-rendered shell + client-fetched data (over templating the data in):**
- The HTML page is static and cacheable; only the data is dynamic. You can open
  the JSON endpoints directly to debug them, and the same endpoints feed both
  the dashboard and anything else (an MCP server, a CLI, a notebook).
- No template/data coupling — changing a chart never means touching Python.
- No build step means no `node_modules`, no bundler config, no stale-dist bugs.
  Edit the `.html`, reload the browser.

**The tradeoff you're accepting:** vanilla JS string-templating gets unwieldy
past a certain size, and there's no component reuse across pages beyond
copy-paste. That's the deliberate ceiling — if you're fighting it, you've
outgrown the pattern.

---

## Stack

- **FastAPI** — two routers: one returns `HTMLResponse` (page shells), one
  returns JSON (data), mounted under `/api`.
- **Jinja2** (`Jinja2Templates`) — renders the shell once per page load, only to
  inject server-side config (user name, IDs from the URL). Not used for data.
- **uvicorn** — ASGI server, launched from a CLI command.
- **Chart.js** (CDN) — all charts. Themed once globally via `Chart.defaults`.
- **Vanilla JS `fetch`** — the page pulls its own data after load.
- No frontend dependencies installed; Chart.js and (optionally) a webfont load
  from a CDN.

## Dependencies

```bash
pip install fastapi "uvicorn[standard]" jinja2
```

---

## 1. App wiring — two routers, one prefix

The whole architecture is in how the routers split. Page routes return HTML;
data routes return JSON under `/api`. Keep them in separate modules.

```python
# api/app.py
from fastapi import FastAPI
from myapp.api.routes import dashboard, data   # placeholder: your package root

app = FastAPI(title="My Dashboard", docs_url=None, redoc_url=None)  # docs off for a local tool

app.include_router(dashboard.router)             # HTML pages at /
app.include_router(data.router, prefix="/api")   # JSON at /api/*
```

```python
# api/routes/dashboard.py — the HTML shell routes
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from myapp import config as cfg_mod   # placeholder: your config accessor

router = APIRouter(tags=["dashboard"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    cfg = cfg_mod.get()
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"user_name": cfg.user.name},   # only server-side config goes in context
    )


# Detail pages take an ID/date from the path and pass it to the template,
# which then fetches /api/<thing>/<id> on load. The Python side stays trivial.
@router.get("/item/{item_id}", response_class=HTMLResponse)
def item_page(request: Request, item_id: int):
    cfg = cfg_mod.get()
    return templates.TemplateResponse(
        request=request, name="item.html",
        context={"user_name": cfg.user.name, "item_id": item_id},
    )
```

## 2. JSON data routes — open a session, return a dict, always close

Every data route follows the same shape: open a session, build a plain dict,
close in `finally`. FastAPI serializes the dict to JSON. Use `HTTPException` for
404/400 so the frontend can branch on `response.ok`.

```python
# api/routes/data.py
from datetime import date
from fastapi import APIRouter, HTTPException, Query

from myapp.db.session import get_session, init_db   # placeholder
from myapp.db.models import DailyRow                 # placeholder

router = APIRouter(tags=["data"])


def _session():
    return get_session(init_db())   # idempotent init; cheap for SQLite


@router.get("/day/{day}")
def day_detail(day: str):
    try:
        d = date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=400, detail="Expected YYYY-MM-DD")
    session = _session()
    try:
        row = session.get(DailyRow, d)
        if not row:
            raise HTTPException(status_code=404, detail="No data for that day")
        return {"date": str(d), "score": row.score, "detail": row.detail}
    finally:
        session.close()


# Time-series endpoints return parallel arrays keyed by metric — Chart.js
# consumes labels[] + one array per dataset directly, no reshaping client-side.
@router.get("/trend")
def trend(days: int = Query(default=30, ge=7, le=365)):
    session = _session()
    try:
        rows = ...  # query ordered by date
        return {
            "labels": [str(r.date) for r in rows],
            "score":  [r.score for r in rows],
            "stress": [r.stress for r in rows],
        }
    finally:
        session.close()
```

**Shape data for the consumer.** Time-series endpoints return parallel arrays
(`labels`, then one array per series) because that's exactly what a Chart.js
dataset wants. Detail endpoints return a flat dict. Do rounding and unit
conversion server-side so the frontend just displays values.

## 3. Launch from a CLI command

Guard the port so a second launch fails loudly instead of silently colliding.

```python
import socket, uvicorn, click

@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8080, type=int)
def serve(host, port):
    """Start the web dashboard."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise click.ClickException(f"Port {port} already in use. Already running?")
    init_db()
    uvicorn.run("myapp.api.app:app", host=host, port=port, reload=False)
```

Note `0.0.0.0` binds all interfaces but you still browse to `localhost:<port>` —
a common confusion when the startup log prints the bind address.

---

## 4. The UI: one design system in CSS variables

Every page opens with the same `:root` token block and the same base styles. The
look is a dark, slightly-glassy dashboard: near-black background with faint
radial color glows, translucent borders, soft shadows, a gradient accent used
sparingly for the active state and section markers. Copy this block verbatim
into each template's `<style>`; it's the contract that keeps pages consistent.

```css
:root {
  --bg: #0a0c13;
  --surface: #12151f;
  --surface-2: #1a1e2c;
  --border: rgba(148, 163, 184, 0.13);
  --border-strong: rgba(148, 163, 184, 0.28);
  --text: #e9edf5;
  --muted: #8b94a9;
  --excellent: #34d399;   /* semantic status colors — rename to your domain */
  --good: #a3e635;
  --moderate: #fbbf24;
  --poor: #f87171;
  --accent: #818cf8;
  --accent-2: #22d3ee;
  --grad: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  /* ambient corner glows — the single biggest "this looks designed" win */
  background-image:
    radial-gradient(1100px 520px at 85% -10%, rgba(99, 102, 241, 0.12), transparent 60%),
    radial-gradient(900px 480px at -10% 110%, rgba(34, 211, 238, 0.06), transparent 60%);
  background-attachment: fixed;
  color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px;
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
}
```

Load Inter from Google Fonts (`<link>` in `<head>`); the `-apple-system` stack
is the fallback. Numbers use `font-variant-numeric: tabular-nums` everywhere
they appear in cards and tables so digits don't jitter as values change.

**Recurring design moves** (each is small but they compound):

- **Cards** share one treatment — a faint top-down white gradient over
  `--surface`, a translucent border, a soft shadow, 14px radius:
  ```css
  .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0) 55%), var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
  }
  .metric-card:hover { border-color: var(--border-strong); transform: translateY(-1px); }
  ```
- **Section headings** get a small gradient bar via `::before`:
  ```css
  h2::before { content: ""; width: 4px; height: 14px; border-radius: 2px; background: var(--grad); }
  ```
- **Sticky frosted header** — `position: sticky` + `backdrop-filter: blur(14px)`
  over a semi-transparent bg.
- **Pill nav** — tabs in a rounded container; the active tab gets `--grad` and a
  glow `box-shadow`. Status badges are pills with a pulsing dot
  (`box-shadow: 0 0 10px currentColor` + a keyframe opacity pulse).
- **Custom select arrow** — strip native chrome with `appearance: none` and set
  a chevron via an inline-SVG `background-image`, so dropdowns match the theme.
- **Tab fade-in** — `.tab-content.active { animation: fadeUp 0.28s ease; }` with
  a 6px translateY. One keyframe, applied on every tab switch.

## 5. Tabs without a router

A single page hides/shows `.tab-content` divs and lazy-loads each tab's data on
first reveal. No client router, no hash handling.

```html
<nav>
  <button class="active" onclick="showTab('today', this)">Today</button>
  <button onclick="showTab('trends', this)">Trends</button>
</nav>
<div id="today" class="tab-content active">…</div>
<div id="trends" class="tab-content">…</div>
```

```javascript
function showTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  btn.classList.add('active');
  if (name === 'trends') loadTrends();   // fetch on first/every reveal
}
```

## 6. Theme Chart.js once, globally

Set `Chart.defaults` at the top of the script so every chart inherits the font,
muted tick color, and dark tooltip. Then a tiny `mkChart` helper destroys the
prior instance before redraw (so re-fetching a tab doesn't leak canvases).

```javascript
Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.color = '#8b94a9';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(26,30,44,0.95)';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(148,163,184,0.28)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.cornerRadius = 8;

const chartInstances = {};
function mkChart(id, config) {
  if (chartInstances[id]) chartInstances[id].destroy();   // prevent canvas leak on re-render
  chartInstances[id] = new Chart(document.getElementById(id), config);
}

// A shared baseOpts object with grid color + tick styling, spread into each
// chart's options, keeps every chart visually consistent:
const gridColor = 'rgba(255,255,255,0.06)';
const baseOpts = {
  responsive: true, maintainAspectRatio: true,
  plugins: { legend: { labels: { color: '#8b94a9', font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: '#8b94a9', maxTicksLimit: 8 }, grid: { color: gridColor } },
    y: { ticks: { color: '#8b94a9' }, grid: { color: gridColor } },
  },
};
```

## 7. Fetch-and-render with explicit error handling

Each loader fetches, throws on non-`ok`, and writes an error message into the
DOM on failure so a dead endpoint shows a red message instead of a blank panel.

```javascript
async function loadToday() {
  const el = document.getElementById('today');
  let data;
  try {
    data = await fetch('/api/today').then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
  } catch (e) {
    el.innerHTML = `<div style="color:var(--poor)">Failed to load: ${e}</div>`;
    return;
  }
  // build HTML by string-templating data into card markup, then assign innerHTML
  el.innerHTML = data.metrics.map(m => `
    <div class="card metric-card">
      <div class="label">${m.label}</div>
      <div class="value">${m.value}</div>
    </div>`).join('');
}
```

**Clickable table rows → detail pages.** A row navigates by setting
`window.location`; the detail page reads its ID from the Jinja context and
fetches `/api/<thing>/<id>`. This is the whole "drill-down" mechanism:

```javascript
`<tr style="cursor:pointer" onclick="window.location.href='/day/${r.date}'">…</tr>`
```

```javascript
// in the detail template, the only server-injected value:
const ITEM_ID = "{{ item_id }}";
fetch(`/api/day/${ITEM_ID}`).then(/* render */);
```

## 8. Comparison-aware stat cards (optional but high-value)

A detail value is more useful next to its baseline. Have the detail endpoint
return a trailing-window average alongside the point value, and render a
direction-aware arrow — green when the change is *good*, which depends on the
metric (higher is better for some, lower for others):

```javascript
function cmp(value, avg, higherIsBetter, unit = '') {
  if (value == null || avg == null) return '';
  const diff = value - avg;
  if (Math.abs(diff) < 0.05 * Math.abs(avg || 1)) return `≈ avg (${avg}${unit})`;
  const better = higherIsBetter ? diff > 0 : diff < 0;
  return `<span class="${better ? 'up' : 'down'}">${diff > 0 ? '▲' : '▼'}</span> avg ${avg}${unit}`;
}
```

---

## Known limitations

- **String-templated HTML doesn't scale.** Past ~5 pages or heavy interactivity,
  the lack of components and the manual `innerHTML` assembly become the
  bottleneck. That's the signal to migrate to a real frontend framework — the
  JSON API you already have stays unchanged underneath it.
- **No client-side caching or state.** Every tab reveal re-fetches. Fine for a
  local single-user tool over SQLite; add caching if endpoints get expensive.
- **CDN dependency.** Chart.js and the webfont load from a CDN, so the dashboard
  needs network on first load (and won't render charts fully offline). Vendor
  them locally into a `static/` dir if you need true offline use.
- **Design tokens are copy-pasted per template, not shared.** Each `.html` has
  its own `:root` block. With more than a couple of pages, factor the `<style>`
  into a single served CSS file (`StaticFiles`) and `<link>` it, so the design
  system has one source of truth.
- **Single-user assumptions.** No auth, no per-request user scoping, no CSRF.
  Bind to `localhost` or put it behind a trusted network/reverse proxy; don't
  expose it to the open internet as-is.
- **Slow endpoints block the panel.** If a route does live external I/O (not
  just a DB read), that panel hangs until it returns. Either keep routes
  DB-only and sync external data separately, or render the page from cached data
  first and load the slow piece into its own panel.
```
