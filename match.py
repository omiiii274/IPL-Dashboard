"""
tabs/match.py
Tab 3 — Match Overview: select any match and view a full
batting / bowling / catches scorecard for both teams.
"""

import json
import numpy as np
import pandas as pd

from shiny import ui, render, reactive, req

from data import balls, matches, SEASONS, BOWL_WKT
from theme import kcard


# ── UI ────────────────────────────────────────────────────────
def tab_match():
    return ui.div(
        ui.div({"class": "pg-header"},
            ui.tags.h2("🏟 Match Overview"),
            ui.tags.p("Detailed batting & bowling scorecard for any match")),

        ui.div({"class": "tbl-wrap", "style": "margin-bottom:18px"},
            ui.layout_columns(
                ui.input_select(
                    "mo_season", "Season",
                    choices={str(s): str(s) for s in SEASONS},
                    selected=str(SEASONS[0]),
                ),
                ui.input_select("mo_match", "Match", choices=[]),
                col_widths=[3, 9],
            ),
        ),

        ui.layout_columns(
            kcard("Match-up", "mo_matchup", icon="⚔️"),
            kcard("Venue",    "mo_venue",   icon="📍"),
            kcard("Result",   "mo_result",  icon="🏁"),
            col_widths=[4, 4, 4],
        ),

        ui.output_ui("mo_scorecards"),
    )


# ── SERVER ────────────────────────────────────────────────────
def register_match(input, output, session):

    @reactive.effect
    @reactive.event(input.mo_season)
    def _update_match_choices():
        s  = int(input.mo_season())
        sm = matches[matches["season_id"] == s].sort_values("match_date")
        ch = {}
        for _, r in sm.iterrows():
            ds = r["match_date"].strftime("%d %b") if pd.notna(r["match_date"]) else ""
            mn = f"#{int(r['match_number'])}" if pd.notna(r["match_number"]) else "#?"
            ch[str(int(r["match_id"]))] = (
                f"{mn}  {ds}  |  {r['team1']} vs {r['team2']}"
            )
        ui.update_select("mo_match", choices=ch)

    @reactive.calc
    def _mo():
        req(input.mo_match())
        mid  = int(input.mo_match())
        mrow = matches[matches["match_id"] == mid]
        if len(mrow) == 0: return None
        return {"row": mrow.iloc[0], "balls": balls[balls["match_id"] == mid]}

    @render.text
    def mo_matchup():
        d = _mo()
        if d is None: return "Select a match"
        return f"{d['row']['team1']}  vs  {d['row']['team2']}"

    @render.text
    def mo_venue():
        d = _mo()
        if d is None: return "—"
        return str(d["row"]["venue"] or "—")

    @render.text
    def mo_result():
        d = _mo()
        if d is None: return "—"
        r = d["row"]
        if r["result"] != "win": return str(r["result"] or "—")
        if pd.notna(r["win_by_runs"]) and r["win_by_runs"] > 0:
            return f"{r['match_winner']} won by {int(r['win_by_runs'])} runs"
        if pd.notna(r["win_by_wickets"]) and r["win_by_wickets"] > 0:
            return f"{r['match_winner']} won by {int(r['win_by_wickets'])} wickets"
        return f"{r['match_winner']} won"

    @render.ui
    def mo_scorecards():
        d = _mo()
        if d is None:
            return ui.div(
                {"class": "tbl-wrap",
                 "style": "text-align:center;padding:48px;color:#94a3b8"},
                ui.tags.p({"style": "font-size:1rem"},
                    "👆 Select a season and match above to view the scorecard"),
            )

        mb    = d["balls"]
        mrow  = d["row"]
        teams = [mrow["team1"], mrow["team2"]]

        # ── helpers ──────────────────────────────────────────
        def _bat(team):
            tb = mb[(mb["team_batting"] == team) & (~mb["is_wide_ball"])]
            if len(tb) == 0: return pd.DataFrame()
            agg = (tb.groupby("batter").agg(
                Runs  = ("batter_runs", "sum"),
                Balls = ("batter_runs", "count"),
                Fours = ("batter_runs", lambda x: (x == 4).sum()),
                Sixes = ("batter_runs", lambda x: (x == 6).sum()),
            ).reset_index().rename(columns={"batter": "Batter"}))
            agg["SR"] = (agg["Runs"] / agg["Balls"] * 100).round(1)
            out_set = set(
                mb[(mb["is_wicket"]) &
                   mb["player_out"].notna() &
                   (mb["player_out"] != "NULL") &
                   (mb["team_batting"] == team)]["player_out"].tolist()
            )
            agg["Status"] = agg["Batter"].apply(
                lambda pl: "out" if pl in out_set else "not out*")
            return agg.sort_values("Runs", ascending=False)

        def _bowl(team):
            bb = mb[(mb["team_bowling"] == team) & (~mb["is_wide_ball"])]
            if len(bb) == 0: return pd.DataFrame()
            agg = (bb.groupby("bowler").agg(
                Balls = ("total_runs", "count"),
                Runs  = ("total_runs", "sum"),
            ).reset_index().rename(columns={"bowler": "Bowler"}))
            wkts = (
                mb[mb["is_wicket"] &
                   mb["wicket_kind"].str.lower().isin(BOWL_WKT) &
                   (mb["team_bowling"] == team)]
                .groupby("bowler").size().reset_index(name="Wickets")
                .rename(columns={"bowler": "Bowler"})
            )
            agg = agg.merge(wkts, on="Bowler", how="left")
            agg["Wickets"] = agg["Wickets"].fillna(0).astype(int)
            agg["Overs"]   = (agg["Balls"] / 6).round(1)
            agg["Econ"]    = (agg["Runs"] / agg["Overs"].replace(0, np.nan)).round(2)
            agg["Bowl SR"] = np.where(
                agg["Wickets"] > 0,
                (agg["Balls"] / agg["Wickets"]).round(1), "-")
            return agg[["Bowler","Overs","Runs","Wickets","Econ","Bowl SR"]].sort_values(
                "Wickets", ascending=False)

        def _catches(team):
            caught = mb[
                (mb["team_bowling"] == team) &
                mb["wicket_kind"].notna() &
                mb["wicket_kind"].str.lower().isin(["caught","caught and bowled"]) &
                mb["fielders_involved"].notna() &
                (mb["fielders_involved"] != "NULL")
            ]
            fielders = []
            for v in caught["fielders_involved"]:
                try:
                    for f in (json.loads(v) or []):
                        if f: fielders.append(str(f).strip())
                except Exception:
                    pass
            if not fielders: return None
            c = pd.Series(fielders).value_counts().reset_index()
            c.columns = ["Fielder", "Catches"]
            return c

        # ── build side-by-side panels ─────────────────────
        sub_s = ("font-size:.65rem;color:#64748b;text-transform:uppercase;"
                 "letter-spacing:1px;margin:14px 0 5px;font-weight:700;")

        panels = []
        for team in teams:
            bat  = _bat(team)
            bowl = _bowl(team)
            ctch = _catches(team)
            items = [
                ui.tags.div(
                    {"style": "font-size:.9rem;font-weight:800;color:#2563eb;margin-bottom:12px"},
                    f"🏏  {team}"),
                ui.tags.div({"style": sub_s}, "Batting"),
                (ui.HTML(bat.to_html(index=False, classes="table table-sm", border=0))
                 if len(bat) > 0 else
                 ui.tags.p({"style": "color:#94a3b8;font-size:.82rem"}, "No batting data")),
                ui.tags.div({"style": sub_s}, "Bowling"),
                (ui.HTML(bowl.to_html(index=False, classes="table table-sm", border=0))
                 if len(bowl) > 0 else
                 ui.tags.p({"style": "color:#94a3b8;font-size:.82rem"}, "No bowling data")),
            ]
            if ctch is not None:
                items += [
                    ui.tags.div({"style": sub_s}, "Catches"),
                    ui.HTML(ctch.to_html(index=False, classes="table table-sm", border=0)),
                ]
            panels.append(ui.div({"class": "tbl-wrap"}, *items))

        return ui.layout_columns(*panels, col_widths=[6, 6])
