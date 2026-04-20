"""
theme.py
All visual constants, Plotly chart helpers, the CSS string,
and small reusable UI components (kcard, sec).
"""

import plotly.graph_objects as go
from shiny import ui

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════
CARD_BG   = "#ffffff"
BG_PLOT   = "#f8fafc"
ACCENT    = "#2563eb"
TEXT      = "#1e293b"
MUTED     = "#64748b"
GRID      = "#e2e8f0"
BAT_GRAD  = [[0, "#93c5fd"], [1, "#1d4ed8"]]
BOWL_GRAD = [[0, "#c4b5fd"], [1, "#6d28d9"]]

# ══════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════
def apply_theme(fig, title="", height=400):
    """Apply the standard white dashboard theme to any Plotly figure."""
    fig.update_layout(
        height=height,
        title=title,
        title_font=dict(size=13, color=ACCENT,
                        family="Segoe UI, system-ui, sans-serif"),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT, family="Segoe UI, system-ui, sans-serif"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font_color=MUTED,
            orientation="h", yanchor="bottom", y=-0.28,
            xanchor="center", x=0.5,
        ),
        margin=dict(l=16, r=16, t=48, b=48),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID,
                     tickcolor=MUTED, tickfont_color=MUTED, tickfont_size=11)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID,
                     tickcolor=MUTED, tickfont_color=MUTED, tickfont_size=11)
    return fig


def empty_fig(msg="No data available"):
    """Return a blank themed figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, text=msg, showarrow=False,
        font=dict(size=13, color=MUTED), xref="paper", yref="paper")
    return apply_theme(fig)


def hbar(data, x_col, y_col, title, colorscale=None):
    """Horizontal bar chart with gradient colouring."""
    cs = colorscale or BAT_GRAD
    fig = go.Figure(go.Bar(
        x=data[x_col], y=data[y_col],
        orientation="h",
        marker=dict(color=data[x_col], colorscale=cs,
                    showscale=False, line_width=0),
        text=data[x_col].round(0).astype(int),
        textposition="outside",
        textfont=dict(color=MUTED, size=10),
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return apply_theme(fig, title, height=380)

# ══════════════════════════════════════════════════════════════
#  REUSABLE UI COMPONENTS
# ══════════════════════════════════════════════════════════════
def kcard(label, out_id, sub_id=None, icon="", extra_cls=""):
    """Stat card with coloured top-bar, big value, and subtitle."""
    return ui.div(
        {"class": f"kcard {extra_cls}"},
        ui.div({"class": "klabel"}, label),
        ui.div({"class": "kvalue"}, ui.output_text(out_id)),
        ui.div({"class": "ksub"},   ui.output_text(sub_id) if sub_id else ""),
        ui.div({"class": "kicon"},  icon),
    )


def sec(title):
    """Section divider with an uppercase label."""
    return ui.div({"class": "sec-title"}, title)

# ══════════════════════════════════════════════════════════════
#  CSS  (loaded once, injected via ui.tags.style in app.py)
# ══════════════════════════════════════════════════════════════
CSS = """
:root {
  --bg:    #f8fafc;
  --bg2:   #f1f5f9;
  --bg3:   #ffffff;
  --brd:   #e2e8f0;
  --acc:   #2563eb;
  --acc-l: #dbeafe;
  --txt:   #1e293b;
  --mut:   #64748b;
  --shad:  0 1px 6px rgba(15,23,42,.07);
}

html, body {
  background: var(--bg) !important;
  color: var(--txt) !important;
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
}
* { box-sizing: border-box; }
.container-fluid { padding: 0 !important; }

/* ── navset_pill_list layout ── */
.bslib-navs-pill-list {
  min-height: 100vh;
  background: var(--bg) !important;
  display: flex !important;
}
.bslib-navs-pill-list > .row > div:first-child,
.bslib-navs-pill-list > div:first-child {
  background: #ffffff !important;
  border-right: 1px solid var(--acc-l) !important;
  min-height: 100vh;
  padding: 0 8px 24px !important;
  box-shadow: 2px 0 10px rgba(37,99,235,.05);
}
.bslib-navs-pill-list > .row > div:last-child,
.bslib-navs-pill-list > div:last-child {
  background: var(--bg) !important;
  padding: 28px 32px !important;
}

/* ── Sidebar brand ── */
.sb-brand { padding: 22px 14px 18px; border-bottom: 1px solid var(--brd); margin-bottom: 10px; }
.sb-title { font-size: 1.05rem; font-weight: 900; color: var(--acc); letter-spacing: -.2px; display: block; }
.sb-sub   { font-size: .68rem; color: var(--mut); margin-top: 3px; display: block; font-weight: 500; }

/* ── Nav links ── */
.nav-stacked .nav-link,
.nav-pills   .nav-link {
  color: var(--mut) !important;
  border-radius: 9px !important;
  padding: 9px 13px !important;
  font-size: .84rem; font-weight: 500;
  transition: background .13s, color .13s;
  border-left: 3px solid transparent !important;
  margin: 2px 0 !important;
  white-space: nowrap;
  display: flex; align-items: center; gap: 6px;
}
.nav-stacked .nav-link:hover,
.nav-pills   .nav-link:hover { background: #eff6ff !important; color: var(--acc) !important; }
.nav-stacked .nav-link.active,
.nav-pills   .nav-link.active {
  background: var(--acc-l) !important;
  color: var(--acc) !important;
  border-left-color: var(--acc) !important;
  font-weight: 700 !important;
}

/* ── Stat cards ── */
.kcard {
  background: #fff; border: 1px solid var(--brd); border-radius: 12px;
  padding: 16px 18px 14px; position: relative; overflow: hidden;
  height: 100%; box-shadow: var(--shad); transition: box-shadow .15s;
}
.kcard:hover { box-shadow: 0 4px 16px rgba(37,99,235,.1); }
.kcard::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #2563eb, #60a5fa);
  border-radius: 12px 12px 0 0;
}
.kcard-orange::after { background: linear-gradient(90deg,#ea580c,#fb923c); }
.kcard-purple::after { background: linear-gradient(90deg,#7c3aed,#a78bfa); }
.kcard-green::after  { background: linear-gradient(90deg,#059669,#34d399); }
.kcard-blue::after   { background: linear-gradient(90deg,#1d4ed8,#60a5fa); }
.kcard-red::after    { background: linear-gradient(90deg,#b91c1c,#f87171); }
.kcard-gold::after   { background: linear-gradient(90deg,#b45309,#fcd34d); }
.klabel { font-size: .64rem; text-transform: uppercase; letter-spacing: 1px; color: var(--mut); margin-bottom: 6px; }
.kvalue { font-size: 1.5rem; font-weight: 800; color: var(--acc); line-height: 1.2; word-break: break-word; }
.ksub   { font-size: .74rem; color: var(--mut); margin-top: 4px; line-height: 1.4; }
.kicon  { position: absolute; top: 12px; right: 13px; font-size: 1.5rem; opacity: .1; }

/* ── Section header ── */
.sec-title {
  font-size: .7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.2px; color: var(--acc);
  border-bottom: 2px solid var(--brd); padding-bottom: 7px; margin: 24px 0 14px;
}

/* ── Wrappers ── */
.tbl-wrap {
  background: #fff; border: 1px solid var(--brd); border-radius: 12px;
  padding: 16px 18px; box-shadow: var(--shad); overflow: auto;
}
.chart-wrap {
  background: #fff; border: 1px solid var(--brd); border-radius: 12px;
  padding: 14px 16px; box-shadow: var(--shad); overflow: hidden;
}

/* ── Page header ── */
.pg-header {
  background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #0891b2 100%);
  border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;
}
.pg-header h2 { color: #fff; font-weight: 800; margin: 0; font-size: 1.25rem; }
.pg-header p  { color: rgba(255,255,255,.72); margin: 4px 0 0; font-size: .85rem; }

/* ── Inputs ── */
.form-select, .form-control, select {
  background: var(--bg2) !important; color: var(--txt) !important;
  border: 1px solid var(--brd) !important; border-radius: 8px !important;
  font-size: .84rem !important;
}
.form-select:focus, .form-control:focus {
  border-color: var(--acc) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,.14) !important;
}
label, .form-label, .control-label {
  color: var(--mut) !important; font-size: .70rem !important;
  font-weight: 600 !important; text-transform: uppercase; letter-spacing: .5px;
}

/* ── DataGrid ── */
.shiny-data-grid-grid thead th {
  background: #eff6ff !important; color: var(--acc) !important;
  font-size: .68rem !important; text-transform: uppercase; letter-spacing: .4px;
  border-bottom: 2px solid var(--brd) !important; font-weight: 700 !important;
}
.shiny-data-grid-grid tbody tr:hover td { background: #f0f9ff !important; }
.shiny-data-grid-grid { background: transparent !important; }
.shiny-data-grid-grid td,
.shiny-data-grid-grid th { color: var(--txt) !important; font-size: .82rem; }
.shiny-data-grid { --shiny-datagrid-grid-header-background-color: #eff6ff !important; }

/* ── HTML tables (scorecards) ── */
.table { color: var(--txt) !important; font-size: .81rem; width: 100%; border-collapse: collapse; }
.table thead th {
  background: #eff6ff !important; color: var(--acc) !important;
  font-size: .67rem !important; text-transform: uppercase; letter-spacing: .4px;
  border-bottom: 2px solid #bfdbfe !important; white-space: nowrap; padding: 8px 10px !important;
}
.table tbody td { padding: 6px 10px !important; border-color: var(--brd) !important; border-bottom: 1px solid #f1f5f9 !important; }
.table tbody tr:nth-child(even) { background: #f8fafc; }
.table tbody tr:hover { background: #eff6ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--acc); }

/* ── Utilities ── */
.col-center { display: flex; justify-content: center; align-items: center; }
"""
