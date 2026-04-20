"""
tabs/prediction.py
Tab 6 — 2026 Match Winner Prediction
Uses the full ML pipeline (ball-by-ball ELO → logistic regression + boosted model).
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shiny import ui, render, reactive, req
from shinywidgets import output_widget, render_widget

from prediction_engine import engine, KNOWN_VENUES
from data import TEAMS_2026
from theme import apply_theme, sec, ACCENT, MUTED, TEXT, GRID

# ── Helpers ───────────────────────────────────────────────────
def _abbr(name: str) -> str:
    m = re.search(r"\(([^)]+)\)", name)
    return m.group(1) if m else name.split()[0]

_CONF_COLOR = {'High': '#059669', 'Moderate': '#d97706', 'Low': '#94a3b8'}

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
def tab_predict():
    metrics = engine.metrics
    boost   = metrics.get('boost_name', 'GradientBoosting')
    auc     = metrics.get('boost_roc_auc', '—')
    ll      = metrics.get('boost_logloss', '—')
    n_tr    = metrics.get('n_train', '—')
    n_te    = metrics.get('n_test',  '—')

    model_badge = ui.div(
        {"style": ("display:flex;gap:20px;flex-wrap:wrap;align-items:center;"
                   "padding:14px 18px;background:#f0f9ff;"
                   "border:1px solid #bfdbfe;border-radius:12px;margin-bottom:20px;")},
        ui.div(
            {"style": "font-size:.72rem;color:#64748b;font-weight:600;"},
            ui.tags.span({"style":"font-size:1rem;margin-right:6px"}, "🤖"),
            f"Pipeline: Ball-by-Ball ELO  +  LR  +  {boost}"
        ),
        ui.div({"style":"width:1px;background:#bfdbfe;height:28px"}),
        _metric_chip("ROC-AUC",  str(auc)),
        _metric_chip("Log-Loss", str(ll)),
        _metric_chip("Train",    f"{n_tr} matches"),
        _metric_chip("Test",     f"{n_te} matches"),
    )

    venue_choices = {"": "Neutral (no venue bias)"}
    venue_choices.update({v: v for v in KNOWN_VENUES})

    return ui.div(
        ui.div({"class": "pg-header"},
            ui.tags.h2("🔮 2026 Match Winner Prediction"),
            ui.tags.p("Ball-by-ball ELO ratings + logistic regression + gradient boosting")),

        model_badge,

        # ── Team + venue selectors ────────────────────────────
        ui.div({"class": "tbl-wrap", "style": "margin-bottom:20px"},
            ui.layout_columns(
                ui.div(
                    ui.input_select("pred_t1", "Team 1",
                        choices={t: t for t in TEAMS_2026},
                        selected=TEAMS_2026[0]),
                ),
                ui.div({"class": "col-center"},
                    ui.tags.div({"style": (
                        "width:42px;height:42px;border-radius:50%;"
                        "background:#2563eb;color:#fff;font-weight:900;"
                        "font-size:.9rem;display:flex;align-items:center;"
                        "justify-content:center;")}, "VS"),
                ),
                ui.div(
                    ui.input_select("pred_t2", "Team 2",
                        choices={t: t for t in TEAMS_2026},
                        selected=TEAMS_2026[1]),
                ),
                col_widths=[5, 2, 5],
            ),

            ui.div({"style": "margin-top:14px"},
                ui.input_select("pred_venue", "Match Venue (optional)",
                    choices=venue_choices,
                    selected="",
                    width="100%"),
            ),

            ui.div({"style": "margin-top:14px"},
                ui.input_action_button(
                    "pred_go", "🔮  Predict Winner",
                    class_="btn btn-primary fw-bold",
                    style=("padding:10px 32px;border-radius:9px;"
                           "background:#2563eb;border-color:#2563eb;"
                           "font-size:.9rem;letter-spacing:.3px;")),
            ),
        ),

        # ── Dynamic result ────────────────────────────────────
        ui.output_ui("pred_result_card"),

        # ── Feature breakdown chart ───────────────────────────
        ui.output_ui("pred_feature_section"),

        # ── Squad ELO tables ─────────────────────────────────
        sec("Squad ELO Ratings"),
        ui.layout_columns(
            ui.div({"class": "tbl-wrap"},
                ui.output_ui("pred_t1_header"),
                ui.output_ui("pred_squad_t1"),
            ),
            ui.div({"class": "tbl-wrap"},
                ui.output_ui("pred_t2_header"),
                ui.output_ui("pred_squad_t2"),
            ),
            col_widths=[6, 6],
        ),

        # ── Backtest table ────────────────────────────────────
        sec("Season-wise Backtesting (held-out 2023 – 2025)"),
        ui.div({"class": "tbl-wrap"},
            ui.output_data_frame("pred_backtest")),
    )


def _metric_chip(label: str, value: str):
    return ui.div(
        {"style": "text-align:center"},
        ui.div({"style": "font-size:.68rem;color:#64748b;text-transform:uppercase;"
                          "letter-spacing:.5px"}, label),
        ui.div({"style": "font-size:.95rem;font-weight:800;color:#1d4ed8"}, value),
    )

# ══════════════════════════════════════════════════════════════
#  SERVER
# ══════════════════════════════════════════════════════════════
def register_prediction(input, output, session):

    @reactive.calc
    def _result():
        req(input.pred_go())
        t1 = input.pred_t1()
        t2 = input.pred_t2()
        if t1 == t2:
            return None
        venue = input.pred_venue() or None
        return engine.predict_2026(t1, t2, venue)

    # ── Main result card ────────────────────────────────────
    @render.ui
    def pred_result_card():
        if input.pred_go() == 0:
            return ui.div(
                {"class": "tbl-wrap",
                 "style": "text-align:center;padding:44px;color:#94a3b8;margin-bottom:20px"},
                ui.tags.p({"style": "font-size:1rem"},
                    "Select two teams and click  🔮 Predict Winner  to see the result"),
            )
        d = _result()
        if d is None:
            return ui.div({"class": "tbl-wrap", "style": "padding:24px;color:#ef4444"},
                "Please select two different teams.")

        t1s, t2s = _abbr(d['team1']), _abbr(d['team2'])
        ws        = _abbr(d['winner'])
        conf_col  = _CONF_COLOR.get(d['confidence'], '#94a3b8')
        win_left  = d['prob1'] >= d['prob2']

        left_clr  = '#2563eb' if win_left else '#e2e8f0'
        right_clr = '#2563eb' if not win_left else '#e2e8f0'

        return ui.div(
            {"style": (
                "background:linear-gradient(135deg,#eff6ff 0%,#f0f9ff 100%);"
                "border:1px solid #bfdbfe;border-radius:16px;"
                "padding:28px 32px;text-align:center;margin-bottom:20px;")},

            # Team chips
            ui.div({"style": "margin-bottom:16px;display:flex;align-items:center;"
                              "justify-content:center;gap:14px"},
                ui.tags.span({"style":"font-size:1rem;font-weight:700;color:#1e293b"}, t1s),
                ui.tags.span({"style":(
                    "width:36px;height:36px;border-radius:50%;"
                    "background:#2563eb;color:#fff;font-weight:900;font-size:.82rem;"
                    "display:inline-flex;align-items:center;justify-content:center;")}, "VS"),
                ui.tags.span({"style":"font-size:1rem;font-weight:700;color:#1e293b"}, t2s),
            ),

            # Probability bar
            ui.div({"style":"display:flex;border-radius:8px;overflow:hidden;height:14px;margin-bottom:6px;"},
                ui.div({"style":f"flex:{d['prob1']};background:{left_clr};transition:flex .4s"}),
                ui.div({"style":f"flex:{d['prob2']};background:{right_clr};transition:flex .4s"}),
            ),
            ui.div({"style":"display:flex;justify-content:space-between;"
                             "font-size:.78rem;color:#64748b;margin-bottom:22px"},
                ui.tags.span(f"{t1s}  {d['prob1']}%"),
                ui.tags.span(f"{t2s}  {d['prob2']}%"),
            ),

            # Winner announcement
            ui.div({"style":"font-size:.68rem;text-transform:uppercase;letter-spacing:1px;"
                             "color:#64748b;margin-bottom:4px"}, "Predicted Winner"),
            ui.div({"style":"font-size:1.8rem;font-weight:900;color:#1d4ed8;line-height:1.1"},
                f"🏆  {ws}"),
            ui.div({"style":"font-size:2.6rem;font-weight:900;color:#2563eb;line-height:1.1;margin:4px 0"},
                f"{max(d['prob1'], d['prob2'])}%"),

            # Confidence badge
            ui.div({"style":"margin-top:10px"},
                ui.tags.span({"style":(
                    f"display:inline-block;padding:3px 14px;border-radius:20px;"
                    f"background:{conf_col}22;color:{conf_col};"
                    f"font-size:.72rem;font-weight:700;letter-spacing:.5px;")},
                    f"{d['confidence']} Confidence"),
            ),

            # ELO strength mini-row
            ui.div({"style":"display:flex;gap:10px;justify-content:center;"
                             "margin-top:18px;flex-wrap:wrap"},
                _mini_stat("Batting ELO", t1s, d['bat1'], t2s, d['bat2']),
                _mini_stat("Bowling ELO", t1s, d['bowl1'], t2s, d['bowl2']),
                _mini_stat("Form (last 5)", t1s, f"{d['form1']}%", t2s, f"{d['form2']}%"),
            ),
        )

    # ── Feature breakdown chart ──────────────────────────────
    @render.ui
    def pred_feature_section():
        if input.pred_go() == 0 or _result() is None:
            return ui.div()
        return ui.div(
            sec("Key Prediction Factors"),
            ui.div({"class": "chart-wrap"},
                output_widget("pred_feature_chart")),
        )

    @render_widget
    def pred_feature_chart():
        d = _result()
        if d is None:
            return go.Figure()

        feats = d['features']
        labels = list(feats.keys())
        values = list(feats.values())
        t1s = _abbr(d['team1'])
        t2s = _abbr(d['team2'])

        colors = ['#2563eb' if v >= 0 else '#e2e8f0' for v in values]
        hover  = [f"{t1s} advantage: +{v}" if v >= 0
                  else f"{t2s} advantage: +{abs(v)}"
                  for v in values]

        fig = go.Figure(go.Bar(
            x=values, y=labels,
            orientation='h',
            marker_color=colors,
            text=[f"+{v}" if v >= 0 else str(v) for v in values],
            textposition='outside',
            textfont=dict(size=11, color=TEXT),
            hovertext=hover,
            hoverinfo='text',
        ))
        fig.add_vline(x=0, line_width=1.5, line_color=GRID)
        fig.update_layout(
            yaxis=dict(autorange='reversed'),
            xaxis_title=f"← {t2s} advantage  |  {t1s} advantage →",
        )
        apply_theme(fig, "What's driving the prediction?", height=300)
        fig.update_xaxes(title_font_color=MUTED, zerolinecolor=GRID)
        return fig

    # ── Squad ELO headers ───────────────────────────────────
    @render.ui
    def pred_t1_header():
        return _team_header(input.pred_t1())

    @render.ui
    def pred_t2_header():
        return _team_header(input.pred_t2())

    # ── Squad ELO tables ────────────────────────────────────
    @render.ui
    def pred_squad_t1():
        return _render_squad(input.pred_t1())

    @render.ui
    def pred_squad_t2():
        return _render_squad(input.pred_t2())

    # ── Backtest table ───────────────────────────────────────
    @render.data_frame
    def pred_backtest():
        df = engine.get_backtest_df()
        if df.empty:
            df = pd.DataFrame({"Note": ["No backtest data available"]})
        return render.DataGrid(df, width="100%", height="220px")


# ══════════════════════════════════════════════════════════════
#  SHARED RENDER HELPERS  (called from within register_*)
# ══════════════════════════════════════════════════════════════

def _team_header(team_raw: str):
    short = _abbr(team_raw)
    return ui.tags.div(
        {"style": "font-size:.88rem;font-weight:800;color:#2563eb;margin-bottom:12px"},
        f"🏏  {short}")


def _render_squad(team_raw: str):
    sq = engine.get_squad_elo(team_raw)
    if sq.empty:
        return ui.tags.p({"style": "color:#94a3b8"}, "Squad data unavailable")

    sub_s = ("font-size:.64rem;color:#64748b;text-transform:uppercase;"
             "letter-spacing:1px;margin:12px 0 6px;font-weight:700;")

    def _table(df, sort_col, label):
        if df.empty:
            return ui.div()
        df = (df.sort_values(sort_col, ascending=False)
              .reset_index(drop=True))
        df.insert(0, "#", range(1, len(df)+1))
        return ui.div(
            ui.tags.div({"style": sub_s}, label),
            ui.HTML(df[["#","Player", sort_col, "Role"]]
                    .to_html(index=False, classes="table table-sm", border=0)),
        )

    batters    = sq[sq['Role'].isin(['Batter', 'All-rounder', 'Rookie'])]
    bowlers    = sq[sq['Role'].isin(['Bowler', 'All-rounder', 'Rookie'])]
    rookies    = sq[sq['Role'] == 'Rookie']

    return ui.div(
        _table(
            sq[sq['Role'].isin(['Batter','All-rounder'])]
              .sort_values('Bat ELO', ascending=False).head(8),
            'Bat ELO', "🏏 Top Batters by ELO"),
        _table(
            sq[sq['Role'].isin(['Bowler','All-rounder'])]
              .sort_values('Bowl ELO', ascending=False).head(8),
            'Bowl ELO', "🎳 Top Bowlers by ELO"),
        _table(rookies, 'Bat ELO', "🌟 Rookies (Method B Prior)")
        if len(rookies) > 0 else ui.div(),
    )


def _mini_stat(label: str, t1: str, v1, t2: str, v2):
    """Small comparison chip used inside the result card."""
    return ui.div({"style": (
        "background:#fff;border:1px solid #e2e8f0;border-radius:10px;"
        "padding:10px 14px;text-align:center;min-width:140px;")},
        ui.div({"style": "font-size:.62rem;color:#64748b;text-transform:uppercase;"
                          "letter-spacing:.6px;margin-bottom:5px"}, label),
        ui.div({"style": "font-size:.82rem;display:flex;justify-content:center;gap:10px"},
            ui.tags.span({"style": "font-weight:700;color:#1d4ed8"}, f"{t1}: {v1}"),
            ui.tags.span({"style": "color:#cbd5e1"}, "|"),
            ui.tags.span({"style": "font-weight:700;color:#7c3aed"}, f"{t2}: {v2}"),
        ),
    )
