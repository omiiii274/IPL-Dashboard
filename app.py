"""
app.py  —  IPL Analytics Dashboard (2008 – 2025)
==========================================
Entry point. Imports UI builders and server registrars from
each tab module, assembles the layout, and starts the app.

Project structure
-----------------
app.py              ← you are here
data.py             ← all data loading, preprocessing & ELO
theme.py            ← colours, Plotly helpers, CSS, shared UI components
tabs/
  __init__.py
  overview.py       ← Tab 1: All-time overview
  season.py         ← Tab 2: Season summary
  match.py          ← Tab 3: Match scorecard
  teams.py          ← Tab 4: Teams per season
  player.py         ← Tab 5: Player stats
  prediction.py     ← Tab 6: 2026 match winner prediction
"""

from shiny import App, ui

from theme import CSS

from tabs.overview   import tab_overview,  register_overview
from tabs.season     import tab_season,    register_season
from tabs.match      import tab_match,     register_match
from tabs.teams      import tab_teams,     register_teams
from tabs.player     import tab_player,    register_player
from tabs.prediction import tab_predict,   register_prediction

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(CSS),
        ui.tags.title("IPL Dashboard"),
    ),
    ui.navset_pill_list(
        ui.nav_panel("📊  Overview",         tab_overview()),
        ui.nav_panel("📅  Season Summary",   tab_season()),
        ui.nav_panel("🏟  Match Overview",   tab_match()),
        ui.nav_panel("👥  Teams Per Season", tab_teams()),
        ui.nav_panel("⭐  Player Stats",      tab_player()),
        ui.nav_panel("🔮  2026 Prediction",  tab_predict()),
        id="main_nav",
        widths=(2, 10),
        header=ui.div(
            {"class": "sb-brand"},
            ui.tags.span({"class": "sb-title"}, "🏏 IPL Analytics"),
            ui.tags.span({"class": "sb-sub"},   "2008 – 2025"),
        ),
    ),
)

# ══════════════════════════════════════════════════════════════
#  SERVER
# ══════════════════════════════════════════════════════════════
def server(input, output, session):
    register_overview(input, output, session)
    register_season(input, output, session)
    register_match(input, output, session)
    register_teams(input, output, session)
    register_player(input, output, session)
    register_prediction(input, output, session)

# ══════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════
app = App(app_ui, server)
