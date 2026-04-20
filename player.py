"""
tabs/player.py
Tab 5 — Player Stats: adaptive pie charts (batting/bowling/all-rounder),
dotted scatter progression, and a season-by-season data table.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shiny import ui, render, reactive
from shinywidgets import output_widget, render_widget

from data import (
    bat_seas, bowl_seas,
    alltime_bat, alltime_bowl,
    wkt_types_all, SEASONS, ALL_PLAYERS,
)
from theme import apply_theme, empty_fig, kcard, sec, ACCENT, MUTED


# ── UI ────────────────────────────────────────────────────────
def tab_player():
    return ui.div(
        ui.div({"class": "pg-header"},
            ui.tags.h2("⭐ Player Statistics"),
            ui.tags.p("Career and season-by-season breakdown for any player")),

        ui.div({"class": "tbl-wrap", "style": "margin-bottom:18px"},
            ui.input_select(
                "ps_player", "Player Name",
                choices={pl: pl for pl in ALL_PLAYERS},
                selected=ALL_PLAYERS[0],
            ),
        ),

        ui.layout_columns(
            # Pie chart + season selector
            ui.div({"class": "chart-wrap"},
                ui.div({"style": "display:flex;justify-content:space-between;"
                                 "align-items:center;margin-bottom:10px"},
                    ui.tags.span(
                        {"style": (f"font-size:.7rem;font-weight:700;"
                                   f"text-transform:uppercase;letter-spacing:1px;"
                                   f"color:{ACCENT}")},
                        "Performance Breakdown",
                    ),
                    ui.input_select(
                        "ps_pie_season", "",
                        choices={str(s): str(s) for s in SEASONS},
                        selected=str(SEASONS[0]),
                        width="110px",
                    ),
                ),
                output_widget("ps_pie"),
            ),
            # Scatter + metric selector
            ui.div({"class": "chart-wrap"},
                ui.div({"style": "display:flex;justify-content:space-between;"
                                 "align-items:center;margin-bottom:10px"},
                    ui.tags.span(
                        {"style": (f"font-size:.7rem;font-weight:700;"
                                   f"text-transform:uppercase;letter-spacing:1px;"
                                   f"color:{ACCENT}")},
                        "Season Progression",
                    ),
                    ui.input_select(
                        "ps_scatter_metric", "",
                        choices={
                            "runs": "Runs", "avg": "Batting Avg",
                            "sr": "Strike Rate", "wickets": "Wickets",
                            "econ": "Economy", "bowl_avg": "Bowl Avg",
                        },
                        selected="runs",
                        width="140px",
                    ),
                ),
                output_widget("ps_scatter"),
            ),
            col_widths=[6, 6],
        ),

        sec("Season-by-Season Statistics"),
        ui.div({"class": "tbl-wrap"},
            ui.output_data_frame("ps_career")),
    )


# ── SERVER ────────────────────────────────────────────────────
def register_player(input, output, session):

    @reactive.calc
    def _ps_type():
        pl = input.ps_player()
        has_bat  = pl in alltime_bat["batter"].values
        has_bowl = (
            pl in alltime_bowl["bowler"].values and
            int(alltime_bowl[alltime_bowl["bowler"] == pl]["total_wkts"].sum()) >= 5 and
            int(alltime_bowl[alltime_bowl["bowler"] == pl]["total_balls"].sum()) >= 60
        )
        if has_bat and has_bowl: return "allrounder"
        if has_bat:              return "batsman"
        return "bowler"

    @reactive.effect
    @reactive.event(_ps_type)
    def _update_metric_choices():
        t = _ps_type()
        if t == "batsman":
            ch, sel = {"runs": "Runs", "avg": "Batting Avg", "sr": "Strike Rate",
                       "fours": "Fours", "sixes": "Sixes"}, "runs"
        elif t == "bowler":
            ch, sel = {"wickets": "Wickets", "econ": "Economy",
                       "bowl_avg": "Bowl Avg", "bowl_sr": "Bowl SR"}, "wickets"
        else:
            ch, sel = {"runs": "Runs", "wickets": "Wickets",
                       "avg": "Batting Avg", "econ": "Economy"}, "runs"
        ui.update_select("ps_scatter_metric", choices=ch, selected=sel)

    @render_widget
    def ps_pie():
        pl = input.ps_player()
        s  = int(input.ps_pie_season())

        bat_r  = bat_seas[(bat_seas["batter"]  == pl) & (bat_seas["season_id"]  == s)]
        bowl_r = bowl_seas[(bowl_seas["bowler"] == pl) & (bowl_seas["season_id"] == s)]

        has_bat  = len(bat_r)  > 0 and bat_r.iloc[0]["balls"]   > 0
        has_bowl = len(bowl_r) > 0 and bowl_r.iloc[0]["balls_b"] > 0

        if not has_bat and not has_bowl:
            return empty_fig(f"No data for {pl} in {s}")

        ncols = 2 if (has_bat and has_bowl) else 1
        subtitles = (["Batting Breakdown", "Wicket Types"] if ncols == 2
                     else ["Batting Breakdown" if has_bat else "Wicket Types"])
        fig = make_subplots(
            rows=1, cols=ncols,
            specs=[[{"type": "pie"}] * ncols],
            subplot_titles=subtitles,
        )

        if has_bat:
            br = bat_r.iloc[0]
            fig.add_trace(go.Pie(
                labels=["Dots", "Singles", "Twos", "Fours", "Sixes"],
                values=[
                    max(int(br["dots"]),  0), max(int(br["ones"]),  0),
                    max(int(br["twos"]),  0), max(int(br["fours"]), 0),
                    max(int(br["sixes"]), 0),
                ],
                hole=0.42,
                marker=dict(
                    colors=["#e2e8f0","#93c5fd","#6ee7b7","#fcd34d","#fca5a5"],
                    line=dict(color="#fff", width=2),
                ),
                textfont_size=11,
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ), row=1, col=1)

        if has_bowl:
            wt = (wkt_types_all[wkt_types_all["bowler"] == pl]
                  .groupby("wicket_kind")["cnt"].sum().reset_index())
            if len(wt) > 0:
                fig.add_trace(go.Pie(
                    labels=wt["wicket_kind"], values=wt["cnt"],
                    hole=0.42,
                    marker=dict(
                        colors=["#93c5fd","#c4b5fd","#fcd34d","#6ee7b7",
                                "#fca5a5","#67e8f9","#fdba74"][:len(wt)],
                        line=dict(color="#fff", width=2),
                    ),
                    textfont_size=11,
                    hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                ), row=1, col=2 if has_bat else 1)

        apply_theme(fig, f"{pl}  ·  {s} Season", height=380)
        fig.update_annotations(font_color=MUTED, font_size=11)
        return fig

    @render_widget
    def ps_scatter():
        pl  = input.ps_player()
        met = input.ps_scatter_metric()

        if met in {"runs", "avg", "sr", "fours", "sixes"}:
            df = bat_seas[bat_seas["batter"] == pl].copy()
            y_label = {
                "runs": "Runs", "avg": "Batting Avg", "sr": "Strike Rate",
                "fours": "Fours", "sixes": "Sixes",
            }[met]
        else:
            df = bowl_seas[bowl_seas["bowler"] == pl].copy()
            y_label = {
                "wickets": "Wickets", "econ": "Economy",
                "bowl_avg": "Bowling Avg", "bowl_sr": "Bowling SR",
            }[met]

        df = df.dropna(subset=[met]).sort_values("season_id")
        if len(df) == 0:
            return empty_fig("Insufficient data for this metric")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["season_id"].astype(str), y=df[met].round(2),
            mode="markers+lines",
            marker=dict(size=10, color="#2563eb", symbol="circle",
                        line=dict(color="#1d4ed8", width=1.5)),
            line=dict(color="#bfdbfe", width=2, dash="dot"),
            hovertemplate=f"Season %{{x}}<br>{y_label}: %{{y}}<extra></extra>",
        ))
        fig.update_xaxes(title_text="Season",   title_font_color=MUTED)
        fig.update_yaxes(title_text=y_label,    title_font_color=MUTED)
        return apply_theme(fig, f"{pl}  ·  {y_label} by Season", height=380)

    @render.data_frame
    def ps_career():
        pl = input.ps_player()
        pt = _ps_type()

        bat_p = bat_seas[bat_seas["batter"] == pl].copy()
        if len(bat_p) > 0:
            bat_p = bat_p[[
                "season_id","matches","runs","balls","fours","sixes","avg","sr"
            ]].rename(columns={
                "season_id": "Season", "matches": "Mat", "runs": "Runs",
                "balls": "Balls", "fours": "4s", "sixes": "6s",
                "avg": "Avg", "sr": "SR",
            })

        bowl_p = bowl_seas[bowl_seas["bowler"] == pl].copy()
        if len(bowl_p) > 0:
            bowl_p = bowl_p[[
                "season_id","matches","wickets","overs","runs_c","econ","bowl_avg","bowl_sr"
            ]].rename(columns={
                "season_id": "Season", "matches": "Mat", "wickets": "Wkts",
                "overs": "Overs", "runs_c": "RC", "econ": "Econ",
                "bowl_avg": "BAvg", "bowl_sr": "BSR",
            })

        if   pt == "batsman"  and len(bat_p)  > 0: df = bat_p
        elif pt == "bowler"   and len(bowl_p) > 0: df = bowl_p
        elif len(bat_p) > 0 and len(bowl_p)  > 0:
            df = bat_p.merge(bowl_p, on=["Season","Mat"], how="outer")
        elif len(bat_p)  > 0: df = bat_p
        elif len(bowl_p) > 0: df = bowl_p
        else:
            df = pd.DataFrame({"Note": ["No data found for this player"]})

        df = df.sort_values("Season", ascending=False).fillna("—")
        return render.DataGrid(df, width="100%", height="360px")
