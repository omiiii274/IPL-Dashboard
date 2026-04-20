"""
tabs/overview.py
Tab 1 — All-time IPL overview: stat cards, team table,
top player charts, and average score progression.
"""

import numpy as np
import plotly.graph_objects as go

from shiny import ui, render
from shinywidgets import output_widget, render_widget

from data import (
    alltime_bat, alltime_bowl, team_stats,
    highest_inn, lowest_inn, best_team,
    most_trophies_team, most_trophies_count,
    avg_by_season,
)
from theme import apply_theme, hbar, kcard, sec, ACCENT, MUTED, BAT_GRAD, BOWL_GRAD
import pandas as pd


# ── UI ────────────────────────────────────────────────────────
def tab_overview():
    return ui.div(
        ui.div({"class": "pg-header"},
            ui.tags.h2("🏏 IPL Overview"),
            ui.tags.p("All-time statistics across every IPL season (2008 – 2025)")),

        ui.layout_columns(
            kcard("Highest Match Score", "ov_hi_score", "ov_hi_sub",  "🔥", "kcard-orange"),
            kcard("Lowest Match Score",  "ov_lo_score", "ov_lo_sub",  "❄️", "kcard-blue"),
            kcard("Best Win Percentage", "ov_best_wp",  "ov_best_wp_sub", "🎯", "kcard-green"),
            kcard("Most Trophy Wins",    "ov_trophies", "ov_trophies_sub","🏆", "kcard-gold"),
            col_widths=[3, 3, 3, 3],
        ),

        sec("Team Statistics"),
        ui.div({"class": "tbl-wrap"},
            ui.output_data_frame("ov_team_tbl")),

        sec("Player Records"),
        ui.layout_columns(
            ui.div({"class": "chart-wrap"}, output_widget("ov_top_bat")),
            ui.div({"class": "chart-wrap"}, output_widget("ov_top_bowl")),
            col_widths=[6, 6],
        ),

        sec("Average Score Progression 2008 – 2025"),
        ui.div({"class": "chart-wrap"}, output_widget("ov_avg_prog")),
    )


# ── SERVER ────────────────────────────────────────────────────
def register_overview(input, output, session):

    @render.text
    def ov_hi_score():
        return f"{int(highest_inn['score'])} runs"

    @render.text
    def ov_hi_sub():
        d = highest_inn["match_date"]
        ds = d.strftime("%d %b %Y") if pd.notna(d) else ""
        return f"{highest_inn['team_batting']}  ·  {ds}"

    @render.text
    def ov_lo_score():
        return f"{int(lowest_inn['score'])} runs"

    @render.text
    def ov_lo_sub():
        d = lowest_inn["match_date"]
        ds = d.strftime("%d %b %Y") if pd.notna(d) else ""
        return f"{lowest_inn['team_batting']}  ·  {ds}"

    @render.text
    def ov_best_wp():
        return f"{best_team['Win %']}%"

    @render.text
    def ov_best_wp_sub():
        return f"{best_team['Team']}  ·  {best_team['Wins']}W / {best_team['Matches']}M"

    @render.text
    def ov_trophies():
        return f"{most_trophies_count} Titles"

    @render.text
    def ov_trophies_sub():
        return most_trophies_team

    @render.data_frame
    def ov_team_tbl():
        df = team_stats[["#", "Team", "Matches", "Wins", "Losses", "Win %"]].copy()
        return render.DataGrid(df, width="100%", height="420px")

    @render_widget
    def ov_top_bat():
        top = alltime_bat.sort_values("total_runs", ascending=False).head(7).iloc[::-1]
        fig = hbar(top, "total_runs", "batter",
                   "Top 7 Run Scorers (All Time)", BAT_GRAD)
        fig.update_traces(
            customdata=np.stack([top["avg"], top["sr"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>Runs: %{x}<br>"
                "Avg: %{customdata[0]}<br>SR: %{customdata[1]}<extra></extra>"),
        )
        fig.update_xaxes(title_text="Total Runs", title_font_color=MUTED)
        return fig

    @render_widget
    def ov_top_bowl():
        top = alltime_bowl.sort_values("total_wkts", ascending=False).head(7).iloc[::-1]
        fig = hbar(top, "total_wkts", "bowler",
                   "Top 7 Wicket Takers (All Time)", BOWL_GRAD)
        fig.update_xaxes(title_text="Total Wickets", title_font_color=MUTED)
        return fig

    @render_widget
    def ov_avg_prog():
        df = avg_by_season.sort_values("season_id")
        fig = go.Figure(go.Bar(
            x=df["season_id"].astype(str),
            y=df["avg_score"],
            marker=dict(
                color=df["avg_score"],
                colorscale=[[0, "#bfdbfe"], [0.5, "#3b82f6"], [1, "#1d4ed8"]],
                showscale=False, line_width=0,
            ),
            text=df["avg_score"], textposition="outside",
            textfont=dict(color=MUTED, size=10),
            hovertemplate="Season %{x}<br>Avg Score: %{y:.1f}<extra></extra>",
        ))
        fig.update_xaxes(title_text="Season",                  title_font_color=MUTED)
        fig.update_yaxes(title_text="Avg Innings Score (runs)", title_font_color=MUTED)
        return apply_theme(fig, "Average Innings Score per Season", height=340)
