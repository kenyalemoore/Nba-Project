"""
data_loader.py  (file 1 of 4)

Reads the Excel files (one file per season, one sheet per team) and builds
ONE table with one row per player per season.

The other files import the settings below, so change them here.
"""

import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# SETTINGS (shared by all files)
# ------------------------------------------------------------

# "data" and "answers" folders must sit next to this file.
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"          # one Excel file per season
ANSWERS_FOLDER = PROJECT_FOLDER / "answers"    # real top scorers (.txt)

# Per-game stat columns every sheet must have.
PER_GAME_STATS = [
    "MP", "FG", "FGA", "3P", "3PA",
    "2P", "2PA", "FT", "FTA",
    "ORB", "DRB", "TRB", "AST",
    "STL", "BLK", "TOV", "PF", "PTS"
]

# Games in each season (2020-21 was shortened). Used to count games missed.
SEASON_LENGTHS = {
    "2018-19 Season": 82,
    "2019-20 Season": 82,
    "2020-21 Season": 72,
    "2021-22 Season": 82,
    "2022-23 Season": 82,
    "2023-24 Season": 82,
    "2024-25 Season": 82,
    "2025-26 Season": 82
}

# Seasons from oldest to newest.
SEASON_ORDER = [
    "2018-19 Season", "2019-20 Season", "2020-21 Season",
    "2021-22 Season", "2022-23 Season", "2023-24 Season",
    "2024-25 Season", "2025-26 Season"
]

# A row only counts as a player if its position is one of these.
# (This filters out headers, notes and other junk rows.)
VALID_POSITIONS = [
    "PG", "SG", "SF", "PF", "C",
    "G", "F", "GF", "FG", "FC", "CF"
]


# ------------------------------------------------------------
# STEP 1: FIND THE FILES
# ------------------------------------------------------------

def find_season_files():
    """Return every season Excel file in the data folder, sorted."""
    # Files starting with "~$" are temporary files Excel creates; skip them.
    season_files = sorted(
        file for file in DATA_FOLDER.glob("*.xlsx")
        if not file.name.startswith("~$")
    )

    if not season_files:
        raise FileNotFoundError(f"No Excel files found in: {DATA_FOLDER}")

    return season_files


# ------------------------------------------------------------
# STEP 2: CLEAN ONE TEAM SHEET
# ------------------------------------------------------------

def clean_team_sheet(df, team, season_name):
    """Clean one team's sheet. Returns None if the sheet can't be used."""

    # Every column we need must be there.
    required_columns = ["Player", "Pos", "Age", "G", "GS"] + PER_GAME_STATS
    missing_columns = [c for c in required_columns if c not in df.columns]

    if missing_columns:
        print(f"Skipping {season_name} - {team}; missing: {missing_columns}")
        return None

    # Drop rows with no name and the "Team Totals" row.
    df = df[df["Player"].notna()].copy()
    df = df[df["Player"].astype(str).str.strip() != "Team Totals"]

    # Make the stat columns numbers (anything unreadable becomes blank).
    for column in ["Age", "G", "GS"] + PER_GAME_STATS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep only normal player rows.
    has_valid_position = df["Pos"].astype(str).str.strip().isin(VALID_POSITIONS)
    has_sane_minutes = df["MP"].between(0, 48)
    played_a_game = df["G"] > 0

    df = df[has_valid_position & has_sane_minutes & played_a_game].copy()

    df["Season"] = season_name
    df["Team"] = team
    return df


# ------------------------------------------------------------
# STEP 3: MERGE A TRADED PLAYER INTO ONE ROW
# ------------------------------------------------------------

def combine_player_records(player, group, season_name):
    """Turn all of one player's rows in a season into a single row.

    A traded player appears on more than one team sheet.
    """
    # Remove identical rows that were copied across sheets.
    group = group.drop_duplicates(subset=["Age", "G", "GS"] + PER_GAME_STATS)

    total_games = group["G"].sum()
    if pd.isna(total_games) or total_games <= 0:
        return None

    record = {
        "Player": player,
        "Season": season_name,
        "Age": group["Age"].mean(),
        "G": total_games,
        "GS": group["GS"].sum(),
        "Pos": group["Pos"].astype(str).mode().iloc[0],
    }

    # Average each stat, weighted by games played with each team.
    # Example: 20 PPG in 30 games + 10 PPG in 10 games
    #          = (20*30 + 10*10) / 40 = 17.5 PPG
    for stat in PER_GAME_STATS:
        record[stat] = (group[stat] * group["G"]).sum() / total_games

    # Games missed = games in the season - games played (never below 0).
    record["Games_Missed"] = max(0, SEASON_LENGTHS[season_name] - total_games)

    return record


# ------------------------------------------------------------
# STEP 4: LOAD ONE SEASON
# ------------------------------------------------------------

def load_season(file):
    """Read one season's Excel file. Returns one row per player."""
    season_name = file.stem      # file name without ".xlsx"

    sheets = pd.read_excel(file, sheet_name=None)   # one sheet per team
    team_tables = []
    seen_sheets = []

    for team, df in sheets.items():
        # Skip a sheet that is an exact copy of an earlier one.
        if any(df.equals(previous) for previous in seen_sheets):
            print(f"Duplicate sheet skipped: {season_name} - {team}")
            continue
        seen_sheets.append(df.copy())

        cleaned = clean_team_sheet(df, team, season_name)
        if cleaned is not None:
            team_tables.append(cleaned)

    if not team_tables:
        return None

    # Stack all teams, then merge each player's rows into one.
    season_data = pd.concat(team_tables, ignore_index=True)
    player_rows = []

    for player, group in season_data.groupby("Player"):
        record = combine_player_records(player, group, season_name)
        if record is not None:
            player_rows.append(record)

    return pd.DataFrame(player_rows)


# ------------------------------------------------------------
# STEP 5: LOAD ALL SEASONS
# ------------------------------------------------------------

def load_all_seasons(verbose=False):
    """Load every season into one table (one row per player per season)."""
    all_seasons = []

    for file in find_season_files():
        if file.stem not in SEASON_LENGTHS:
            print(f"Skipping unrecognized season file: {file.name}")
            continue

        season_table = load_season(file)
        if season_table is not None:
            all_seasons.append(season_table)

    if not all_seasons:
        raise ValueError("No season data was successfully loaded.")

    nba_data = pd.concat(all_seasons, ignore_index=True)

    if verbose:
        print("\nTotal player-season records:", len(nba_data))
    return nba_data


# ------------------------------------------------------------
# OPTIONAL: SANITY CHECKS (only print, never change the data)
# ------------------------------------------------------------

def check_data_quality(nba_data):
    """Print a few checks so you can confirm the data looks right."""

    # A traded player should be merged into one row.
    print("\nAlec Burks - 2018-19:")
    print(
        nba_data[
            (nba_data["Player"] == "Alec Burks")
            & (nba_data["Season"] == "2018-19 Season")
        ][["Player", "Season", "G", "GS", "Games_Missed", "PTS"]]
    )

    print("\nData Quality Check (these should all be 0)")
    print("Duplicate player-season rows:",
          nba_data.duplicated(subset=["Player", "Season"]).sum())
    print("Players with GS greater than G:",
          (nba_data["GS"] > nba_data["G"]).sum())

    print("\nPlayers per season:")
    print(nba_data.groupby("Season")["Player"].count())

    print("\nTop 10 scorers - 2025-26:")
    print(
        nba_data[nba_data["Season"] == "2025-26 Season"]
        .sort_values("PTS", ascending=False)
        [["Player", "G", "GS", "PTS"]]
        .head(10)
        .to_string(index=False)
    )

    # More than 82 games is impossible, so list any that show up.
    over_82 = nba_data[nba_data["G"] > 82]
    print("\nPlayers with more than 82 games:", len(over_82))
    if len(over_82):
        print(
            over_82[["Player", "Season", "G", "GS"]]
            .sort_values(["Season", "G"], ascending=[True, False])
            .to_string(index=False)
        )
