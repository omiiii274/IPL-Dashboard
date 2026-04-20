"""
data.py
Loads every CSV/Excel file, runs all preprocessing, builds derived
tables (batting, bowling, team stats, ELO) and exposes them as
module-level names that every tab can simply import.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
def _p(*parts): return os.path.join(BASE, *parts)

# ══════════════════════════════════════════════════════════════
#  RAW DATA
# ══════════════════════════════════════════════════════════════
balls    = pd.read_csv(_p("IPL Data/ball_by_ball_data.csv"), low_memory=False)
matches  = pd.read_csv(_p("IPL Data/ipl_matches_data.csv"))
players  = pd.read_excel(_p("IPL Data/players_data_xlsx.xlsx"))
logos_df = pd.read_csv(_p("IPL Data/teams_logo.csv"))
ipl_2026 = pd.read_csv(_p("IPL Data/IPL_2026.csv"))

# ══════════════════════════════════════════════════════════════
#  TYPE NORMALISATION
# ══════════════════════════════════════════════════════════════

# Boolean columns
for c in ["is_wicket","is_wide_ball","is_no_ball",
          "is_super_over","is_leg_bye","is_bye","is_penalty"]:
    balls[c] = balls[c].astype(str).str.strip().str.upper().isin(
        ["TRUE","1","YES","T"])

matches["match_date"] = pd.to_datetime(
    matches["match_date"], dayfirst=True, errors="coerce")

for df_, col_ in [(matches,"season_id"), (matches,"match_id"),
                  (balls,"season_id"),   (balls,"match_id")]:
    df_[col_] = pd.to_numeric(df_[col_], errors="coerce").astype("Int64")

for col_ in ["match_number","win_by_runs","win_by_wickets"]:
    matches[col_] = pd.to_numeric(matches[col_], errors="coerce")

# ══════════════════════════════════════════════════════════════
#  INNINGS TOTALS
# ══════════════════════════════════════════════════════════════
inn = (
    balls[~balls["is_super_over"]]
    .groupby(["match_id","season_id","team_batting","innings"])["total_runs"]
    .sum().reset_index(name="score")
)
inn = inn.merge(
    matches[["match_id","match_date","venue","team1","team2","match_winner"]],
    on="match_id", how="left")
inn["season_id"] = inn["season_id"].astype("Int64")

# ══════════════════════════════════════════════════════════════
#  BOWLING WICKET TYPES  (reused by multiple tabs)
# ══════════════════════════════════════════════════════════════
BOWL_WKT = frozenset({
    "bowled","caught","lbw","stumped","hit wicket",
    "caught and bowled","obstructing the field"
})

# ══════════════════════════════════════════════════════════════
#  BATTING STATS  (per player × season)
# ══════════════════════════════════════════════════════════════
_nw = balls[~balls["is_wide_ball"]].copy()   # non-wide deliveries

bat_seas = (
    _nw.groupby(["season_id","batter"]).agg(
        runs    = ("batter_runs","sum"),
        balls   = ("batter_runs","count"),
        fours   = ("batter_runs", lambda x: (x==4).sum()),
        sixes   = ("batter_runs", lambda x: (x==6).sum()),
        ones    = ("batter_runs", lambda x: (x==1).sum()),
        twos    = ("batter_runs", lambda x: (x==2).sum()),
        dots    = ("batter_runs", lambda x: (x==0).sum()),
        matches = ("match_id","nunique"),
    ).reset_index()
)
bat_seas["sr"] = (bat_seas["runs"] / bat_seas["balls"] * 100).round(2)

_dis = (
    balls[balls["is_wicket"] &
          balls["player_out"].notna() &
          (balls["player_out"] != "NULL")]
    .groupby(["season_id","player_out"]).size()
    .reset_index(name="dismissals")
    .rename(columns={"player_out":"batter"})
)
bat_seas = bat_seas.merge(_dis, on=["season_id","batter"], how="left")
bat_seas["dismissals"] = bat_seas["dismissals"].fillna(0).astype(int)
bat_seas["avg"] = np.where(
    bat_seas["dismissals"] > 0,
    (bat_seas["runs"] / bat_seas["dismissals"]).round(2),
    bat_seas["runs"].astype(float))

# ══════════════════════════════════════════════════════════════
#  BOWLING STATS  (per player × season)
# ══════════════════════════════════════════════════════════════
_bowl_base = (
    _nw.groupby(["season_id","bowler"]).agg(
        balls_b = ("total_runs","count"),
        runs_c  = ("total_runs","sum"),
        matches = ("match_id","nunique"),
    ).reset_index()
)
_wkts_s = (
    balls[balls["is_wicket"]]
    .assign(_bwkt=lambda d: d["wicket_kind"].str.lower().isin(BOWL_WKT))
    [lambda d: d["_bwkt"]]
    .groupby(["season_id","bowler"]).size()
    .reset_index(name="wickets")
)
bowl_seas = _bowl_base.merge(_wkts_s, on=["season_id","bowler"], how="left")
bowl_seas["wickets"]  = bowl_seas["wickets"].fillna(0).astype(int)
bowl_seas["overs"]    = (bowl_seas["balls_b"] / 6).round(1)
bowl_seas["econ"]     = np.where(bowl_seas["overs"] > 0,
    (bowl_seas["runs_c"] / bowl_seas["overs"]).round(2), 0.0)
bowl_seas["bowl_avg"] = np.where(bowl_seas["wickets"] > 0,
    (bowl_seas["runs_c"] / bowl_seas["wickets"]).round(2), np.nan)
bowl_seas["bowl_sr"]  = np.where(bowl_seas["wickets"] > 0,
    (bowl_seas["balls_b"] / bowl_seas["wickets"]).round(2), np.nan)

# ══════════════════════════════════════════════════════════════
#  ALL-TIME AGGREGATES
# ══════════════════════════════════════════════════════════════
alltime_bat = (
    bat_seas.groupby("batter").agg(
        total_runs    = ("runs","sum"),
        total_balls   = ("balls","sum"),
        total_fours   = ("fours","sum"),
        total_sixes   = ("sixes","sum"),
        dismissals    = ("dismissals","sum"),
        seasons       = ("season_id","nunique"),
        total_matches = ("matches","sum"),
    ).reset_index()
)
alltime_bat["sr"]  = (alltime_bat["total_runs"] / alltime_bat["total_balls"] * 100).round(2)
alltime_bat["avg"] = np.where(
    alltime_bat["dismissals"] > 0,
    (alltime_bat["total_runs"] / alltime_bat["dismissals"]).round(2),
    alltime_bat["total_runs"].astype(float))

alltime_bowl = (
    bowl_seas.groupby("bowler").agg(
        total_wkts    = ("wickets","sum"),
        total_runs_c  = ("runs_c","sum"),
        total_balls   = ("balls_b","sum"),
        seasons       = ("season_id","nunique"),
        total_matches = ("matches","sum"),
    ).reset_index()
)
alltime_bowl["overs"]    = (alltime_bowl["total_balls"] / 6).round(1)
alltime_bowl["econ"]     = np.where(alltime_bowl["overs"] > 0,
    (alltime_bowl["total_runs_c"] / alltime_bowl["overs"]).round(2), 0.0)
alltime_bowl["bowl_avg"] = np.where(alltime_bowl["total_wkts"] > 0,
    (alltime_bowl["total_runs_c"] / alltime_bowl["total_wkts"]).round(2), np.nan)

# Wicket-type breakdown (used by player stats pie chart)
wkt_types_all = (
    balls[balls["is_wicket"] &
          balls["wicket_kind"].notna() &
          (balls["wicket_kind"] != "NULL")]
    .groupby(["bowler","wicket_kind"]).size()
    .reset_index(name="cnt")
)

# ══════════════════════════════════════════════════════════════
#  TEAM STATS
# ══════════════════════════════════════════════════════════════
_all_teams = pd.concat([
    matches[["match_id","team1"]].rename(columns={"team1":"team"}),
    matches[["match_id","team2"]].rename(columns={"team2":"team"}),
])
_team_mp   = _all_teams.groupby("team").size().reset_index(name="matches")
_team_wins = (
    matches[matches["result"]=="win"]
    .groupby("match_winner").size()
    .reset_index(name="wins")
    .rename(columns={"match_winner":"team"})
)
team_stats = _team_mp.merge(_team_wins, on="team", how="left")
team_stats["wins"]    = team_stats["wins"].fillna(0).astype(int)
team_stats["losses"]  = team_stats["matches"] - team_stats["wins"]
team_stats["win_pct"] = (team_stats["wins"] / team_stats["matches"] * 100).round(1)
team_stats = team_stats.sort_values("win_pct", ascending=False).reset_index(drop=True)
team_stats.insert(0, "#", range(1, len(team_stats)+1))
team_stats.rename(columns={
    "team":"Team","matches":"Matches",
    "wins":"Wins","losses":"Losses","win_pct":"Win %"
}, inplace=True)

# ══════════════════════════════════════════════════════════════
#  SEASON WINNERS
# ══════════════════════════════════════════════════════════════
def _season_winner(s):
    g = matches[(matches["season_id"]==s) & (matches["result"]=="win")]
    return g.sort_values("match_date").iloc[-1]["match_winner"] if len(g) else "N/A"

season_winners = {
    s: _season_winner(s)
    for s in sorted(matches["season_id"].dropna().unique())
}

_trophy_cnt = pd.Series(season_winners.values()).value_counts().reset_index()
_trophy_cnt.columns = ["team","trophies"]
most_trophies_team  = _trophy_cnt.iloc[0]["team"]
most_trophies_count = int(_trophy_cnt.iloc[0]["trophies"])

# ══════════════════════════════════════════════════════════════
#  QUICK SUMMARY STATS
# ══════════════════════════════════════════════════════════════
_valid_inn  = inn[inn["score"] >= 10]
highest_inn = _valid_inn.sort_values("score", ascending=False).iloc[0]
lowest_inn  = _valid_inn.sort_values("score").iloc[0]
best_team   = team_stats.sort_values("Win %", ascending=False).iloc[0]

avg_by_season = (
    inn.groupby("season_id")["score"].mean()
    .round(1).reset_index(name="avg_score").dropna()
)
avg_by_season["season_id"] = avg_by_season["season_id"].astype(int)

# ══════════════════════════════════════════════════════════════
#  REFERENCE LISTS
# ══════════════════════════════════════════════════════════════
SEASONS     = sorted(
    matches["season_id"].dropna().unique().astype(int).tolist(), reverse=True)
ALL_PLAYERS = sorted(
    set(alltime_bat["batter"].tolist() + alltime_bowl["bowler"].tolist()))

# ══════════════════════════════════════════════════════════════
#  ELO RATINGS  (2026 prediction)
# ══════════════════════════════════════════════════════════════
def _compute_elo():
    records = {}

    for _, r in alltime_bat.iterrows():
        pl = r["batter"]; elo = 850
        if r["total_balls"] >= 30:
            elo = 900 + min(r["avg"] * 5.5, 320)
            elo += max(min((r["sr"] - 120) * 1.5, 160), -100)
            elo += min(r["seasons"] * 8, 80)
        records[pl] = {"bat_elo": round(elo), "bowl_elo": 800,
                       "runs": int(r["total_runs"]), "wickets": 0}

    for _, r in alltime_bowl.iterrows():
        pl = r["bowler"]; elo = 850
        if r["total_balls"] >= 60 and r["total_wkts"] >= 5:
            elo = 900 + min(r["total_wkts"] * 2.3, 290)
            elo += max(min((8 - r["econ"]) * 22, 150), -120)
            if pd.notna(r["bowl_avg"]):
                elo += max(min((28 - r["bowl_avg"]) * 2.5, 100), -80)
        if pl in records:
            records[pl]["bowl_elo"] = round(elo)
            records[pl]["wickets"]  = int(r["total_wkts"])
        else:
            records[pl] = {"bat_elo": 800, "bowl_elo": round(elo),
                           "runs": 0, "wickets": int(r["total_wkts"])}

    rows = []
    for pl, v in records.items():
        b, k = v["bat_elo"], v["bowl_elo"]
        if k <= 800:   overall = b
        elif b <= 820: overall = k
        else:          overall = round(b * 0.55 + k * 0.45)
        rows.append({
            "Player": pl,
            "Batting ELO": b, "Bowling ELO": k,
            "Overall ELO": overall,
            "Career Runs": v["runs"],
            "Career Wickets": v["wickets"],
        })
    return pd.DataFrame(rows)


_elo_df = _compute_elo()


def _build_2026_elo():
    out = []
    for _, row in ipl_2026.iterrows():
        pl, team = row["Player"], row["Team"]
        m = _elo_df[_elo_df["Player"] == pl]
        if len(m) == 0:
            # fuzzy fall-back: match on last name token
            last = pl.split()[-1]
            m = _elo_df[_elo_df["Player"].str.contains(last, case=False, na=False)]
        elo_val = int(m.iloc[0]["Overall ELO"]) if len(m) > 0 else 950
        out.append({"Player": pl, "Team": team, "ELO Rating": elo_val})
    return pd.DataFrame(out).sort_values("ELO Rating", ascending=False)


elo_2026   = _build_2026_elo()
TEAMS_2026 = sorted(ipl_2026["Team"].unique().tolist())
