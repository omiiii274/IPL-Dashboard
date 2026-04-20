"""
tabs/teams.py
Tab 4 — Teams Per Season

Shows every player who appeared in ball-by-ball data for a given season,
PLUS any extra players added manually via squad_overrides.csv.

How the override file works
---------------------------
Place a file called  squad_overrides.csv  in the ipl_dashboard/ folder.
Columns: Season, Team, Player, Notes  (same layout as the template Excel).
Any row in that file whose (Season, Team, Player) combination is not already
in the ball-by-ball roster will be merged in automatically at startup.
Players added this way are shown with a  ★  prefix so they are easy to spot.
"""

import os
import pandas as pd
from shiny import ui, render
from data import balls, SEASONS
from theme import sec

# ── Load override file once at import time ────────────────────
_BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OVERRIDE = os.path.join(_BASE, "squad_overrides.csv")


def _load_overrides() -> pd.DataFrame:
    """Read squad_overrides.csv if it exists; return empty DataFrame otherwise."""
    if not os.path.exists(_OVERRIDE):
        return pd.DataFrame(columns=["Season", "Team", "Player", "Notes"])
    try:
        df = pd.read_csv(_OVERRIDE)
        df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
        df = df.dropna(subset=["Season", "Team", "Player"])
        df["Season"] = df["Season"].astype(int)
        df["Player"] = df["Player"].astype(str).str.strip()
        df["Team"]   = df["Team"].astype(str).str.strip()
        # Drop any template placeholder rows
        df = df[~df["Player"].str.startswith("(")]
        df = df[~df["Player"].isin(["Example Player Name", "Another Bench Player"])]
        return df
    except Exception:
        return pd.DataFrame(columns=["Season", "Team", "Player", "Notes"])


_overrides    = _load_overrides()
_has_overrides = len(_overrides) > 0


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
def tab_teams():
    if _has_overrides:
        status = ui.div(
            {"style": ("background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;"
                       "padding:10px 16px;margin-bottom:16px;font-size:.82rem;color:#166534;")},
            f"✅  squad_overrides.csv loaded — {len(_overrides)} extra player rows merged. "
            "Players added manually are shown with a ★ prefix.",
        )
    else:
        status = ui.div(
            {"style": ("background:#fffbeb;border:1px solid #fde68a;border-radius:10px;"
                       "padding:10px 16px;margin-bottom:16px;font-size:.82rem;color:#92400e;")},
            "💡  No squad_overrides.csv found in the project folder. "
            "Fill in the template Excel file and save it as squad_overrides.csv "
            "to add bench players who never batted or bowled.",
        )

    return ui.div(
        ui.div({"class": "pg-header"},
            ui.tags.h2("👥 Teams Per Season"),
            ui.tags.p("Squad roster for each franchise — ball-by-ball data "
                      "plus any extras added via squad_overrides.csv")),

        status,

        ui.div({"class": "tbl-wrap", "style": "margin-bottom:18px"},
            ui.input_select(
                "tps_season", "Season",
                choices={str(s): str(s) for s in SEASONS},
                selected=str(SEASONS[0]),
                width="200px",
            ),
        ),

        sec("Season Rosters"),
        ui.div({"class": "tbl-wrap"},
            ui.output_data_frame("tps_roster")),
    )


# ══════════════════════════════════════════════════════════════
#  SERVER
# ══════════════════════════════════════════════════════════════
def register_teams(input, output, session):

    @render.data_frame
    def tps_roster():
        s  = int(input.tps_season())
        sb = balls[balls["season_id"] == s]

        # ── Players found in ball-by-ball data ────────────────
        batters = sb[["team_batting", "batter"]].drop_duplicates().rename(
            columns={"team_batting": "team", "batter": "player"})
        bowlers = sb[["team_bowling", "bowler"]].drop_duplicates().rename(
            columns={"team_bowling": "team", "bowler": "player"})
        roster = (
            pd.concat([batters, bowlers])
            .drop_duplicates()
            .pipe(lambda d: d[d["player"].notna() & (d["player"] != "NULL")])
            .copy()
        )
        roster["_key"] = roster["team"] + "||" + roster["player"]

        # ── Merge manually added players for this season ──────
        if _has_overrides:
            ov = (
                _overrides[_overrides["Season"] == s][["Team", "Player"]]
                .rename(columns={"Team": "team", "Player": "player"})
                .copy()
            )
            ov["_key"] = ov["team"] + "||" + ov["player"]
            new_rows = ov[~ov["_key"].isin(roster["_key"])].copy()
            if len(new_rows) > 0:
                new_rows["player"] = "★ " + new_rows["player"]
                roster = pd.concat(
                    [roster, new_rows[["team", "player"]]], ignore_index=True)

        roster = roster.drop(columns=["_key"], errors="ignore")

        # ── Build wide-format table (one column per team) ─────
        teams_s = sorted(roster["team"].unique())
        if not teams_s:
            return render.DataGrid(
                pd.DataFrame({"Note": [
                    "No ball-by-ball data for this season. "
                    "Add players via squad_overrides.csv."
                ]}),
                width="100%", height="100px",
            )

        def _sort_key(name: str):
            """Regular players first (alpha), then ★ manual additions."""
            return (name.startswith("★"), name.lstrip("★ ").lower())

        max_len = max(len(roster[roster["team"] == t]) for t in teams_s)
        wide = {}
        for t in teams_s:
            pl = sorted(roster[roster["team"] == t]["player"].tolist(), key=_sort_key)
            wide[t] = pl + [""] * (max_len - len(pl))

        return render.DataGrid(pd.DataFrame(wide), width="100%", height="600px")
