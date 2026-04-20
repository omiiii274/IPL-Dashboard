"""
prediction_engine.py
====================
IPL pre-match winner prediction pipeline.

Steps
-----
1. Ball-by-ball batter / bowler ELO ratings (single chronological pass)
   - Phase-aware K-factor  (powerplay / middle / death)
   - Wicket-impact multiplier
   - Season time-decay     (older seasons → lower K)
   - Debutant high-K early career (Method B rookie prior)

2. Pre-match feature engineering (zero leakage — built before ELO update)
   - Batting & bowling unit strength (top-N ELO averages)
   - Recent form (last-5-match win rate)
   - Head-to-head historical win rate
   - Venue batting adjustment (team avg at venue vs league avg)
   - Toss information

3. Model training
   - Baseline : Logistic Regression (sklearn)
   - Boosted  : XGBoost → LightGBM → sklearn GradientBoosting (first found)

4. Evaluation
   - Log-loss, ROC-AUC on held-out seasons (2023-2025)
   - Season-wise accuracy backtesting

5. 2026 match prediction using IPL_2026 squads
"""

import re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict, deque

# ── sklearn (always available) ────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

# ── Optional boosters ─────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

# Normalised outcome score for each batter-run value (batter-centric 0-1)
# Calibrated against IPL historical distribution so equal-ELO expected ≈ 0.40
# (matches the empirical average outcome ≈ 0.38-0.42)
_OUTCOME_SCORE = {0: 0.22, 1: 0.46, 2: 0.58, 3: 0.66, 4: 0.80, 6: 0.95}
_WICKET_SCORE  = 0.02          # wicket = near-zero for batter

# Team-name normalisation (2026 format → historical ball-by-ball format)
_TEAM_NORM: dict[str, str] = {
    'Chennai Super Kings (CSK)':        'Chennai Super Kings',
    'Delhi Capitals (DC)':              'Delhi Capitals',
    'Gujarat Titans (GT)':              'Gujarat Titans',
    'Kolkata Knight Riders (KKR)':      'Kolkata Knight Riders',
    'Lucknow Super Giants (LSG)':       'Lucknow Super Giants',
    'Mumbai Indians (MI)':              'Mumbai Indians',
    'Punjab Kings (PBKS)':              'Punjab Kings',
    'Rajasthan Royals (RR)':            'Rajasthan Royals',
    'Royal Challengers Bengaluru (RCB)':'Royal Challengers Bangalore',
    'Royal Challengers Bengaluru':      'Royal Challengers Bangalore',
    'Sunrisers Hyderabad (SRH)':        'Sunrisers Hyderabad',
    # Legacy names
    'Delhi Daredevils':                 'Delhi Capitals',
    'Deccan Chargers':                  'Sunrisers Hyderabad',
    'Kings XI Punjab':                  'Punjab Kings',
}

FEATURE_COLS = [
    'bat_elo_diff',     # batting-unit ELO advantage for team1
    'bowl_elo_diff',    # bowling-unit ELO advantage for team1
    'form_diff',        # recent-form win-rate difference
    'h2h_rate',         # head-to-head centred win rate (team1 perspective)
    'venue_bat_diff',   # venue batting adjustment (team1 − team2)
    'toss_won_t1',      # 1 if team1 won the toss
    'toss_bat',         # 1 if toss winner chose to bat
]

# ══════════════════════════════════════════════════════════════
#  SMALL HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _norm_team(name: str) -> str:
    return _TEAM_NORM.get(str(name), str(name))

def _norm_venue(v) -> str:
    if not isinstance(v, str) or not v.strip():
        return '__neutral__'
    return re.sub(r'[^a-z0-9]', '', v.lower())[:24]

def _phase(over: int) -> str:
    if over <= 5:  return 'powerplay'
    if over <= 14: return 'middle'
    return 'death'

def _outcome(batter_runs, is_wicket: bool) -> float:
    if is_wicket:
        return _WICKET_SCORE
    return _OUTCOME_SCORE.get(int(batter_runs), 0.50)

# ══════════════════════════════════════════════════════════════
#  ENGINE
# ══════════════════════════════════════════════════════════════

class IPLPredictionEngine:

    # ── Hyper-parameters ──────────────────────────────────────
    INIT_ELO        = 1000.0
    BASE_K          = 18.0
    PHASE_K         = {'powerplay': 0.85, 'middle': 1.00, 'death': 1.20}
    WICKET_MULT     = 2.8
    ROOKIE_THRESH   = 40          # deliveries before K drops to normal
    ROOKIE_K_MULT   = 1.6
    DECAY_HALF_LIFE = 4.0         # years; K halves every 4 seasons
    CURRENT_YEAR    = 2025
    FORM_WINDOW     = 5           # matches for rolling form
    SQUAD_WINDOW    = 3           # past matches used to infer squad
    TOP_BAT         = 7           # top batters to average for strength
    TOP_BOWL        = 5           # top bowlers to average for strength
    TRAIN_UP_TO     = 2022        # seasons ≤ this → training set

    # ─────────────────────────────────────────────────────────
    def __init__(self):
        # Per-player ELO ratings (final values after all matches)
        self._bat_elo  : dict[str, float] = {}
        self._bowl_elo : dict[str, float] = {}
        self._bat_del  : dict[str, int]   = {}   # deliveries faced
        self._bowl_del : dict[str, int]   = {}   # deliveries bowled

        # Rolling squad tracker: team → deque of {batters, bowlers}
        self._squad_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.SQUAD_WINDOW))

        # Running stat records (updated AFTER each match → no leakage)
        self._form : dict[str, list] = defaultdict(list)
        self._h2h  : dict[tuple, list] = defaultdict(lambda: [0, 0])  # [wins, total]
        self._venue_runs  : dict[tuple, list] = defaultdict(list)      # (team, venue) → scores
        self._venue_all   : dict[str,  list]  = defaultdict(list)      # venue → all scores

        # Rookie (Method B) prior ELO — computed after fitting
        self._rookie_bat_elo  = self.INIT_ELO
        self._rookie_bowl_elo = self.INIT_ELO

        # Model artefacts
        self.scaler_lr   : StandardScaler | None = None
        self.model_lr    : LogisticRegression | None = None
        self.model_boost = None       # XGB / LGB / sklearn GBM
        self.boost_name  = ''
        self.feature_cols = FEATURE_COLS
        self.metrics : dict = {}

        # 2026 squad ELO table
        self._squad_df : pd.DataFrame | None = None

        self._fitted = False

    # ══════════════════════════════════════════════════════════
    #  PUBLIC: FIT
    # ══════════════════════════════════════════════════════════

    def fit(self, balls_df: pd.DataFrame, matches_df: pd.DataFrame,
            ipl_2026_df: pd.DataFrame) -> None:
        """
        Run the full pipeline once at startup.
        Processes all historical data in chronological order:
        features built → ELO updated → records updated (no leakage).
        """
        print("[Prediction Engine] Starting fit…")

        # ── Prep ─────────────────────────────────────────────
        matches_df = matches_df.copy()
        matches_df['match_date'] = pd.to_datetime(
            matches_df['match_date'], dayfirst=True, errors='coerce')
        matches_df['season_id'] = pd.to_numeric(
            matches_df['season_id'], errors='coerce').astype('Int64')
        matches_df['match_id'] = pd.to_numeric(
            matches_df['match_id'], errors='coerce').astype('Int64')

        ordered = (matches_df
                   .dropna(subset=['match_date'])
                   .sort_values('match_date')
                   .reset_index(drop=True))

        balls_df = balls_df.copy()
        balls_df['match_id'] = pd.to_numeric(
            balls_df['match_id'], errors='coerce').astype('Int64')
        for col in ['is_wicket','is_wide_ball','is_no_ball','is_super_over']:
            balls_df[col] = balls_df[col].astype(str).str.upper().isin(
                ['TRUE','1','YES','T'])

        grouped = {int(k): v for k, v in balls_df.groupby('match_id')}

        feat_rows: list[dict] = []

        # ── Single chronological pass ─────────────────────────
        for _, match in ordered.iterrows():
            mid    = int(match['match_id']) if pd.notna(match['match_id']) else -1
            t1     = str(match['team1'])
            t2     = str(match['team2'])
            venue  = _norm_venue(match.get('venue'))
            season = int(match['season_id']) if pd.notna(match['season_id']) else 2020
            result = str(match.get('result', ''))
            winner = str(match.get('match_winner', ''))
            toss_w = str(match.get('toss_winner', ''))
            toss_d = str(match.get('toss_decision', ''))

            # 1. Build PRE-MATCH features using CURRENT ELO
            if result == 'win' and winner in (t1, t2):
                won = (winner == t1)
                row = self._pre_match_features(t1, t2, venue, toss_w, toss_d)
                row.update({'match_id': mid, 'season_id': season,
                             'team1_won': int(won)})
                feat_rows.append(row)

            # 2. Process deliveries → update ELO
            mb = grouped.get(mid)
            if mb is not None:
                mb = mb.sort_values(['innings','over_number','ball_number'])
                lineup_t1 = {'batters': set(), 'bowlers': set()}
                lineup_t2 = {'batters': set(), 'bowlers': set()}

                for row_d in mb.itertuples(index=False):
                    try:
                        self._update_one_delivery(row_d, season)
                        # Collect lineup
                        team_bat  = str(row_d.team_batting)
                        team_bowl = str(row_d.team_bowling)
                        batter = str(row_d.batter) if pd.notna(row_d.batter) else ''
                        bowler = str(row_d.bowler) if pd.notna(row_d.bowler) else ''
                        if team_bat == t1 and batter:
                            lineup_t1['batters'].add(batter)
                        if team_bat == t2 and batter:
                            lineup_t2['batters'].add(batter)
                        if team_bowl == t1 and bowler:
                            lineup_t1['bowlers'].add(bowler)
                        if team_bowl == t2 and bowler:
                            lineup_t2['bowlers'].add(bowler)
                    except Exception:
                        continue

                # Update squad history
                self._squad_hist[t1].append(lineup_t1)
                self._squad_hist[t2].append(lineup_t2)

                # Update venue runs
                for team, key in ((t1, 'team_batting'), (t2, 'team_batting')):
                    score = mb.loc[mb[key] == team, 'total_runs'].sum()
                    self._venue_runs[(team, venue)].append(float(score))
                    self._venue_all[venue].append(float(score))

            # 3. Update form / H2H records AFTER using for features
            if result == 'win' and winner in (t1, t2):
                won = (winner == t1)
                self._form[t1].append(won)
                self._form[t2].append(not won)
                self._h2h[(t1, t2)][0] += int(won)
                self._h2h[(t1, t2)][1] += 1

        # ── Compute rookie (Method B) ELO ─────────────────────
        self._compute_rookie_elo()

        # ── Train models ──────────────────────────────────────
        feat_df = pd.DataFrame(feat_rows)
        if len(feat_df) >= 60:
            self._train(feat_df)

        # ── Build 2026 squad ELO table ────────────────────────
        self._squad_df = self._build_squad_table(ipl_2026_df)

        self._fitted = True
        print(f"[Prediction Engine] Done. "
              f"LR ROC-AUC={self.metrics.get('lr_roc_auc','N/A')}  "
              f"Boost={self.boost_name} "
              f"ROC-AUC={self.metrics.get('boost_roc_auc','N/A')}")

    # ══════════════════════════════════════════════════════════
    #  INTERNAL: ELO UPDATE
    # ══════════════════════════════════════════════════════════

    def _compute_k(self, phase: str, season: int,
                   is_wicket: bool, min_del: int) -> float:
        phase_m  = self.PHASE_K[phase]
        wkt_m    = self.WICKET_MULT if is_wicket else 1.0
        rookie_m = self.ROOKIE_K_MULT if min_del < self.ROOKIE_THRESH else 1.0
        years_ago = max(0, self.CURRENT_YEAR - season)
        decay    = max(0.30, 0.5 ** (years_ago / self.DECAY_HALF_LIFE))
        return self.BASE_K * phase_m * wkt_m * rookie_m * decay

    def _update_one_delivery(self, row, season: int) -> None:
        if row.is_wide_ball:
            return
        batter = str(row.batter) if pd.notna(row.batter) else ''
        bowler = str(row.bowler) if pd.notna(row.bowler) else ''
        if not batter or not bowler or batter == 'NULL' or bowler == 'NULL':
            return

        is_wkt   = bool(row.is_wicket) and not bool(row.is_no_ball)
        over_n   = int(row.over_number) if pd.notna(row.over_number) else 10
        actual   = _outcome(row.batter_runs, is_wkt)
        phase    = _phase(over_n)

        bat_d = self._bat_del.get(batter, 0)
        bwl_d = self._bowl_del.get(bowler, 0)
        k     = self._compute_k(phase, season, is_wkt, min(bat_d, bwl_d))

        b_elo = self._bat_elo.get(batter, self.INIT_ELO)
        p_elo = self._bowl_elo.get(bowler, self.INIT_ELO)
        expected = 1.0 / (1.0 + 10.0 ** (-(b_elo - p_elo) / 400.0))

        delta = k * (actual - expected)
        self._bat_elo[batter]  = b_elo + delta
        self._bowl_elo[bowler] = p_elo - delta
        self._bat_del[batter]  = bat_d + 1
        self._bowl_del[bowler] = bwl_d + 1

    def _compute_rookie_elo(self) -> None:
        """Method B: use median ELO of low-experience players as debutant prior."""
        lo, hi = 30, 250
        bat_r  = [v for k, v in self._bat_elo.items()
                  if lo <= self._bat_del.get(k, 0) <= hi]
        bowl_r = [v for k, v in self._bowl_elo.items()
                  if lo <= self._bowl_del.get(k, 0) <= hi]
        self._rookie_bat_elo  = float(np.median(bat_r))  if bat_r  else self.INIT_ELO
        self._rookie_bowl_elo = float(np.median(bowl_r)) if bowl_r else self.INIT_ELO

    # ══════════════════════════════════════════════════════════
    #  INTERNAL: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════

    def _get_bat_elo(self, player: str) -> float:
        return self._bat_elo.get(player, self._rookie_bat_elo)

    def _get_bowl_elo(self, player: str) -> float:
        return self._bowl_elo.get(player, self._rookie_bowl_elo)

    def _team_strength(self, team: str) -> tuple[float, float]:
        """Batting and bowling ELO unit strength from recent-squad history."""
        batters: set[str] = set()
        bowlers: set[str] = set()
        for entry in self._squad_hist[team]:
            batters |= entry.get('batters', set())
            bowlers |= entry.get('bowlers', set())

        bat_elos  = sorted([self._get_bat_elo(p)  for p in batters if p],
                           reverse=True)[:self.TOP_BAT]
        bowl_elos = sorted([self._get_bowl_elo(p) for p in bowlers if p],
                           reverse=True)[:self.TOP_BOWL]

        bat_str  = float(np.mean(bat_elos))  if bat_elos  else self.INIT_ELO
        bowl_str = float(np.mean(bowl_elos)) if bowl_elos else self.INIT_ELO
        return bat_str, bowl_str

    def _team_form(self, team: str) -> float:
        h = self._form[team][-self.FORM_WINDOW:]
        return sum(h) / len(h) if h else 0.50

    def _h2h_rate(self, t1: str, t2: str) -> float:
        """Team1 historical win rate vs team2, centred at 0."""
        w1, n1 = self._h2h.get((t1, t2), [0, 0])
        w2, n2 = self._h2h.get((t2, t1), [0, 0])
        total  = n1 + n2
        wins_t1 = w1 + (n2 - w2)
        return wins_t1 / total - 0.5 if total >= 3 else 0.0

    def _venue_adj(self, team: str, venue: str) -> float:
        """Team's avg batting score at venue minus league average (centred)."""
        team_sc = self._venue_runs.get((team, venue), [])
        all_sc  = self._venue_all.get(venue, [])
        if len(team_sc) < 3 or len(all_sc) < 6:
            return 0.0
        return float(np.mean(team_sc) - np.mean(all_sc))

    def _pre_match_features(self, t1: str, t2: str, venue: str,
                            toss_w: str, toss_d: str) -> dict:
        bat1, bowl1 = self._team_strength(t1)
        bat2, bowl2 = self._team_strength(t2)
        return {
            'bat_elo_diff':   bat1  - bat2,
            'bowl_elo_diff':  bowl1 - bowl2,
            'form_diff':      self._team_form(t1) - self._team_form(t2),
            'h2h_rate':       self._h2h_rate(t1, t2),
            'venue_bat_diff': self._venue_adj(t1, venue) - self._venue_adj(t2, venue),
            'toss_won_t1':    1.0 if toss_w == t1 else 0.0,
            'toss_bat':       1.0 if toss_d == 'bat' else 0.0,
        }

    # ══════════════════════════════════════════════════════════
    #  INTERNAL: MODEL TRAINING
    # ══════════════════════════════════════════════════════════

    def _train(self, feat_df: pd.DataFrame) -> None:
        fc = self.feature_cols
        train = feat_df[feat_df['season_id'] <= self.TRAIN_UP_TO]
        test  = feat_df[feat_df['season_id'] >  self.TRAIN_UP_TO]

        if len(train) < 40 or len(test) < 10:
            return

        X_tr = train[fc].fillna(0).values
        y_tr = train['team1_won'].values
        X_te = test[fc].fillna(0).values
        y_te = test['team1_won'].values

        # ── Logistic Regression baseline ──────────────────────
        self.scaler_lr = StandardScaler()
        X_tr_s = self.scaler_lr.fit_transform(X_tr)
        X_te_s = self.scaler_lr.transform(X_te)

        self.model_lr = LogisticRegression(C=0.8, max_iter=500, random_state=42)
        self.model_lr.fit(X_tr_s, y_tr)
        p_lr = self.model_lr.predict_proba(X_te_s)[:, 1]

        # ── Boosted model ─────────────────────────────────────
        if HAS_XGB:
            self.model_boost = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, gamma=1.0,
                random_state=42, eval_metric='logloss', verbosity=0)
            self.model_boost.fit(X_tr, y_tr,
                                 eval_set=[(X_te, y_te)], verbose=False)
            self.boost_name = 'XGBoost'
        elif HAS_LGB:
            self.model_boost = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.04,
                num_leaves=15, min_child_samples=10,
                random_state=42, verbose=-1)
            self.model_boost.fit(X_tr, y_tr)
            self.boost_name = 'LightGBM'
        else:
            self.model_boost = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.04,
                subsample=0.8, min_samples_leaf=10, random_state=42)
            self.model_boost.fit(X_tr, y_tr)
            self.boost_name = 'GradientBoosting'

        p_boost = self.model_boost.predict_proba(X_te)[:, 1]

        # ── Metrics ───────────────────────────────────────────
        def _safe_auc(y, p):
            return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5

        self.metrics = {
            'n_train':        int(len(X_tr)),
            'n_test':         int(len(X_te)),
            'lr_logloss':     round(log_loss(y_te, p_lr),    4),
            'lr_roc_auc':     round(_safe_auc(y_te, p_lr),   4),
            'boost_logloss':  round(log_loss(y_te, p_boost), 4),
            'boost_roc_auc':  round(_safe_auc(y_te, p_boost),4),
            'boost_name':     self.boost_name,
        }

        # ── Season-wise backtesting ───────────────────────────
        backtest = {}
        for s in sorted(test['season_id'].dropna().unique()):
            sub = test[test['season_id'] == s]
            if len(sub) < 5:
                continue
            Xs = self.scaler_lr.transform(sub[fc].fillna(0).values)
            ys = sub['team1_won'].values
            ps = self.model_boost.predict_proba(sub[fc].fillna(0).values)[:, 1]
            acc = float(np.mean((ps > 0.5).astype(int) == ys))
            backtest[int(s)] = {
                'n':   int(len(ys)),
                'acc': round(acc, 3),
                'auc': round(_safe_auc(ys, ps), 3),
            }
        self.metrics['backtest'] = backtest

    # ══════════════════════════════════════════════════════════
    #  INTERNAL: 2026 SQUAD TABLE
    # ══════════════════════════════════════════════════════════

    def _build_squad_table(self, ipl_2026_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, r in ipl_2026_df.iterrows():
            pl   = str(r['Player'])
            team = str(r['Team'])
            bat  = self._bat_elo.get(pl, self._rookie_bat_elo)
            bowl = self._bowl_elo.get(pl, self._rookie_bowl_elo)
            bd   = self._bat_del.get(pl, 0)
            pd_  = self._bowl_del.get(pl, 0)

            if   bd  > 2 * pd_ and bd  > 30: role = 'Batter'
            elif pd_ > 2 * bd  and pd_ > 30: role = 'Bowler'
            elif bd  > 10 or pd_ > 10:        role = 'All-rounder'
            else:                              role = 'Rookie'

            rows.append({
                'Player':    pl,
                'Team':      team,
                'Role':      role,
                'Bat ELO':   round(bat,  1),
                'Bowl ELO':  round(bowl, 1),
                'Bat Balls': bd,
                'Bowl Balls': pd_,
            })
        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════
    #  PUBLIC: PREDICT
    # ══════════════════════════════════════════════════════════

    def predict_2026(self, team1_raw: str, team2_raw: str,
                     venue_raw: str | None = None) -> dict | None:
        """
        Predict the winner of a 2026 match between two IPL teams.
        Returns a results dict consumed by the Shiny tab.
        """
        if not self._fitted or self.model_lr is None:
            return None

        t1 = _norm_team(team1_raw)
        t2 = _norm_team(team2_raw)
        venue = _norm_venue(venue_raw or '')

        sq1 = self._squad_df[self._squad_df['Team'] == team1_raw]
        sq2 = self._squad_df[self._squad_df['Team'] == team2_raw]

        def _bat_str(sq):
            elos = sq.sort_values('Bat ELO', ascending=False).head(self.TOP_BAT)['Bat ELO']
            return float(elos.mean()) if len(elos) else self.INIT_ELO

        def _bowl_str(sq):
            elos = sq.sort_values('Bowl ELO', ascending=False).head(self.TOP_BOWL)['Bowl ELO']
            return float(elos.mean()) if len(elos) else self.INIT_ELO

        bat1, bat2   = _bat_str(sq1), _bat_str(sq2)
        bowl1, bowl2 = _bowl_str(sq1), _bowl_str(sq2)
        form1 = self._team_form(t1)
        form2 = self._team_form(t2)
        h2h   = self._h2h_rate(t1, t2)
        vd1   = self._venue_adj(t1, venue)
        vd2   = self._venue_adj(t2, venue)

        feat_raw = {
            'bat_elo_diff':   bat1  - bat2,
            'bowl_elo_diff':  bowl1 - bowl2,
            'form_diff':      form1 - form2,
            'h2h_rate':       h2h,
            'venue_bat_diff': vd1 - vd2,
            'toss_won_t1':    0.5,    # unknown pre-match → neutral
            'toss_bat':       0.5,
        }
        X = np.array([[feat_raw[c] for c in self.feature_cols]])

        p_lr    = float(self.model_lr.predict_proba(self.scaler_lr.transform(X))[0, 1])
        p_boost = float(self.model_boost.predict_proba(X)[0, 1])
        # Weighted ensemble: 40 % LR + 60 % Boost
        prob1 = max(0.05, min(0.95, 0.40 * p_lr + 0.60 * p_boost))
        prob2 = 1.0 - prob1

        # Feature labels for display
        feature_display = {
            'Batting Strength (ELO)':   round(bat1 - bat2,        1),
            'Bowling Strength (ELO)':   round(bowl1 - bowl2,      1),
            'Recent Form (last 5)':     round((form1 - form2) * 100, 1),
            'Head-to-Head (%)':         round(h2h * 100,           1),
            'Venue Advantage (runs)':   round(vd1 - vd2,           1),
        }

        return {
            'team1':      team1_raw,
            'team2':      team2_raw,
            'prob1':      round(prob1 * 100, 1),
            'prob2':      round(prob2 * 100, 1),
            'winner':     team1_raw if prob1 >= prob2 else team2_raw,
            'confidence': ('High'     if abs(prob1 - 0.5) > 0.18 else
                           'Moderate' if abs(prob1 - 0.5) > 0.08 else 'Low'),
            'features':   feature_display,
            'bat1': round(bat1, 1),  'bat2': round(bat2, 1),
            'bowl1': round(bowl1,1), 'bowl2': round(bowl2,1),
            'form1': round(form1*100,1), 'form2': round(form2*100,1),
            'metrics': self.metrics,
        }

    def get_squad_elo(self, team_raw: str) -> pd.DataFrame:
        """Return the full squad ELO table for a 2026 team."""
        if self._squad_df is None:
            return pd.DataFrame()
        return self._squad_df[self._squad_df['Team'] == team_raw].copy()

    def get_backtest_df(self) -> pd.DataFrame:
        """Return season-wise backtest results as a DataFrame."""
        bt = self.metrics.get('backtest', {})
        if not bt:
            return pd.DataFrame()
        rows = [{'Season': s, 'Matches': v['n'],
                 'Accuracy': f"{v['acc']*100:.0f}%",
                 'ROC-AUC':  v['auc']}
                for s, v in sorted(bt.items())]
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  MODULE-LEVEL INSTANTIATION  (runs once at startup)
# ══════════════════════════════════════════════════════════════
from data import balls, matches, ipl_2026

# Expose venue list for the UI dropdown
KNOWN_VENUES = sorted(matches['venue'].dropna().unique().tolist())

engine = IPLPredictionEngine()
engine.fit(balls, matches, ipl_2026)
