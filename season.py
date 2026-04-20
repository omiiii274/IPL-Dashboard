"""
tabs/season.py
Tab 2 — Season Summary: Orange/Purple cap, season winner,
highest/lowest score, and aggregate totals for a chosen season.
"""

from shiny import ui, render, reactive

from data import (
    bat_seas, bowl_seas, matches, inn,
    season_winners, SEASONS,
)
from theme import kcard, sec


# ── UI ────────────────────────────────────────────────────────
def tab_season():
    return ui.div(
        ui.div({"class": "pg-header"},
            ui.tags.h2("📅 Season Summary"),
            ui.tags.p("Key statistics and cap winners for a selected IPL season")),

        ui.div({"class": "tbl-wrap", "style": "margin-bottom:18px"},
            ui.input_select(
                "ss_season", "Season",
                choices={str(s): str(s) for s in SEASONS},
                selected=str(SEASONS[0]),
                width="200px",
            ),
        ),

        ui.layout_columns(
            kcard("🟠 Orange Cap — Most Runs",    "ss_orange", "ss_orange_runs", "🏏", "kcard-orange"),
            kcard("🟣 Purple Cap — Most Wickets", "ss_purple", "ss_purple_wkts", "🎳", "kcard-purple"),
            kcard("🏆 Season Winner",             "ss_winner", "ss_winner_sub",  "🥇", "kcard-gold"),
            col_widths=[4, 4, 4],
        ),

        sec("Batting & Scoring"),
        ui.layout_columns(
            kcard("Highest Score", "ss_hi",      "ss_hi_sub", "↑", "kcard-red"),
            kcard("Lowest Score",  "ss_lo",      "ss_lo_sub", "↓", "kcard-blue"),
            kcard("Total Matches", "ss_matches",  icon="📋"),
            kcard("Average Score", "ss_avg",      icon="〰️"),
            col_widths=[3, 3, 3, 3],
        ),

        sec("Season Totals"),
        ui.layout_columns(
            kcard("Total Runs",    "ss_runs",  icon="🏏", extra_cls="kcard-orange"),
            kcard("Total Wickets", "ss_wkts",  icon="🎳", extra_cls="kcard-purple"),
            kcard("Total Sixes",   "ss_sixes", icon="6️⃣", extra_cls="kcard-green"),
            kcard("Total Fours",   "ss_fours", icon="4️⃣", extra_cls="kcard-blue"),
            col_widths=[3, 3, 3, 3],
        ),
    )


# ── SERVER ────────────────────────────────────────────────────
def register_season(input, output, session):

    @reactive.calc
    def _ss():
        s = int(input.ss_season())
        return {
            "bat":    bat_seas[bat_seas["season_id"] == s],
            "bowl":   bowl_seas[bowl_seas["season_id"] == s],
            "mat":    matches[matches["season_id"] == s],
            "inn":    inn[inn["season_id"] == s],
            "season": s,
        }

    # ── Orange cap ───────────────────────────────────────────
    @render.text
    def ss_orange():
        d = _ss()["bat"]
        if len(d) == 0: return "N/A"
        return d.sort_values("runs", ascending=False).iloc[0]["batter"]

    @render.text
    def ss_orange_runs():
        d = _ss()["bat"]
        if len(d) == 0: return ""
        r = d.sort_values("runs", ascending=False).iloc[0]
        return f"{int(r['runs'])} runs  ·  Avg {r['avg']:.1f}  ·  SR {r['sr']:.1f}"

    # ── Purple cap ───────────────────────────────────────────
    @render.text
    def ss_purple():
        d = _ss()["bowl"]
        if len(d) == 0: return "N/A"
        return d.sort_values("wickets", ascending=False).iloc[0]["bowler"]

    @render.text
    def ss_purple_wkts():
        d = _ss()["bowl"]
        if len(d) == 0: return ""
        r = d.sort_values("wickets", ascending=False).iloc[0]
        return f"{int(r['wickets'])} wickets  ·  Econ {r['econ']:.2f}"

    # ── Season winner ────────────────────────────────────────
    @render.text
    def ss_winner():
        return season_winners.get(_ss()["season"], "N/A")

    @render.text
    def ss_winner_sub():
        s = _ss()["season"]
        w = season_winners.get(s, "N/A")
        if w == "N/A": return ""
        prev = sum(1 for yr, t in season_winners.items() if t == w and yr < s)
        return f"IPL {s} Champions  ·  {prev + 1} title(s)"

    # ── Highest / Lowest score ───────────────────────────────
    @render.text
    def ss_hi():
        i = _ss()["inn"]
        if len(i) == 0: return "N/A"
        return f"{int(i.sort_values('score', ascending=False).iloc[0]['score'])} runs"

    @render.text
    def ss_hi_sub():
        i = _ss()["inn"]
        if len(i) == 0: return ""
        r = i.sort_values("score", ascending=False).iloc[0]
        import pandas as pd
        d = r["match_date"].strftime("%d %b") if pd.notna(r["match_date"]) else ""
        return f"{r['team_batting']}  ·  {d}"

    @render.text
    def ss_lo():
        i = _ss()["inn"][_ss()["inn"]["score"] >= 5]
        if len(i) == 0: return "N/A"
        return f"{int(i.sort_values('score').iloc[0]['score'])} runs"

    @render.text
    def ss_lo_sub():
        i = _ss()["inn"][_ss()["inn"]["score"] >= 5]
        if len(i) == 0: return ""
        r = i.sort_values("score").iloc[0]
        import pandas as pd
        d = r["match_date"].strftime("%d %b") if pd.notna(r["match_date"]) else ""
        return f"{r['team_batting']}  ·  {d}"

    # ── Aggregate counts ─────────────────────────────────────
    @render.text
    def ss_matches():
        return str(len(_ss()["mat"]))

    @render.text
    def ss_avg():
        i = _ss()["inn"]
        return f"{i['score'].mean():.1f}" if len(i) > 0 else "N/A"

    @render.text
    def ss_runs():
        s = _ss()["season"]
        return f"{int(bat_seas[bat_seas['season_id']==s]['runs'].sum()):,}"

    @render.text
    def ss_wkts():
        s = _ss()["season"]
        return str(int(bowl_seas[bowl_seas["season_id"]==s]["wickets"].sum()))

    @render.text
    def ss_sixes():
        s = _ss()["season"]
        return f"{int(bat_seas[bat_seas['season_id']==s]['sixes'].sum()):,}"

    @render.text
    def ss_fours():
        s = _ss()["season"]
        return f"{int(bat_seas[bat_seas['season_id']==s]['fours'].sum()):,}"
