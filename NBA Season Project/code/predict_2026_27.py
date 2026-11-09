"""
predict_2026_27.py   <-- RUN THIS FILE:  python predict_2026_27.py

Predicts points per game (PPG) for the 2026-27 season and shows the
predicted top 25 scorers (also saved to predictions_2026-27.txt).

How it works:
    1. Load every season we have (2018-19 to 2025-26).
    2. Add a blank "2026-27" row for every player who played in 2025-26.
       Their past seasons are the real data, so the model can predict them.
    3. Train the same model as main.py on ALL completed seasons.
    4. Predict the blank rows and rank the players.

Keep this file in the same folder as the other 4 files.
"""

import numpy as np
import pandas as pd

from data_loader import PROJECT_FOLDER, SEASON_ORDER, load_all_seasons
from features import build_features
from model import train_linear_regression, predict_ppg, raw_equation


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

TARGET_SEASON = "2026-27 Season"

# Same choice as main.py: "current", "recent_heavy" or "equal".
WEIGHT_SCHEME = "current"

# We can't know games played in 2026-27, so a player needs this many games
# LAST season to be listed. (Set to 0 to include everybody.)
MIN_GAMES_LAST_SEASON = 30

TOP_N_SHOWN = 25
OUTPUT_TXT = PROJECT_FOLDER / "predictions_2026-27.txt"

LINE = "=" * 78

# Tell the project about the new season. (features.py uses this same list.)
if TARGET_SEASON not in SEASON_ORDER:
    SEASON_ORDER.append(TARGET_SEASON)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def add_placeholder_rows(nba_data):
    """Add one blank 2026-27 row for each player from the latest real season."""
    latest_season = SEASON_ORDER[SEASON_ORDER.index(TARGET_SEASON) - 1]
    latest = nba_data[nba_data["Season"] == latest_season]

    if latest.empty:
        raise ValueError(f"No data found for {latest_season}.")

    blank_rows = pd.DataFrame({
        "Player": latest["Player"].to_numpy(),
        "Pos": latest["Pos"].to_numpy(),
        "Season": TARGET_SEASON,
    })
    return pd.concat([nba_data, blank_rows], ignore_index=True)


def format_table(leaderboard):
    """Return the top-N table as lines of text."""
    header = f"{'Rank':>4}  {'Player':<26}{'Pos':<5}{'Last PPG':>9}{'Pred PPG':>10}"
    lines = [header, "-" * len(header)]

    for _, row in leaderboard.head(TOP_N_SHOWN).iterrows():
        lines.append(
            f"{int(row['Rank']):>4}  {row['Player']:<26}{row['Pos']:<5}"
            f"{row['PPG_Last_Season']:>9.1f}"
            f"{row['Predicted_PPG_2026_27']:>10.1f}"
        )
    return lines


# ------------------------------------------------------------
# RUN EVERYTHING
# ------------------------------------------------------------

def main():
    nba_data = load_all_seasons()
    print(
        f"Loaded {len(nba_data)} player-seasons from "
        f"{nba_data['Season'].nunique()} seasons."
    )

    # ---- Build inputs (past seasons are real, 2026-27 rows are blank) ----
    combined = add_placeholder_rows(nba_data)
    model_data, feature_columns = build_features(
        combined, weight_scheme=WEIGHT_SCHEME
    )

    # ---- Train on every completed season, predict 2026-27 ----
    training_seasons = SEASON_ORDER[1:SEASON_ORDER.index(TARGET_SEASON)]
    train_data = model_data[model_data["Season"].isin(training_seasons)]
    test_data = model_data[model_data["Season"] == TARGET_SEASON].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Training or prediction data is empty.")

    print(
        f"Trained on {len(train_data)} player-seasons from "
        f"{len(training_seasons)} seasons; "
        f"predicting {len(test_data)} players."
    )

    X_train = train_data[feature_columns].to_numpy(dtype=float)
    y_train = train_data["PTS"].to_numpy(dtype=float)
    X_test = test_data[feature_columns].to_numpy(dtype=float)

    weights, bias, means, stds = train_linear_regression(X_train, y_train)
    predictions = predict_ppg(X_test, weights, bias, means, stds)
    predictions = np.maximum(predictions, 0)      # PPG can't be negative

    # ---- Rank the players ----
    results = pd.DataFrame({
        "Player": test_data["Player"].to_numpy(),
        "Pos": test_data["Pos"].to_numpy(),
        "Games_Last_Season": test_data["Prev1_G"].to_numpy(),
        "PPG_Last_Season": test_data["Prev1_PTS"].to_numpy(),
        "Predicted_PPG_2026_27": predictions,
    })

    leaderboard = results[results["Games_Last_Season"] >= MIN_GAMES_LAST_SEASON]
    leaderboard = leaderboard.sort_values(
        "Predicted_PPG_2026_27", ascending=False
    ).reset_index(drop=True)
    leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

    table_lines = format_table(leaderboard)

    # ---- Print and save ----
    title = f"PREDICTED TOP {TOP_N_SHOWN} SCORERS - {TARGET_SEASON.upper()}"
    subtitle = f"(players with at least {MIN_GAMES_LAST_SEASON} games last season)"

    print(f"\n{LINE}\n{title}\n{subtitle}\n{LINE}")
    print("\n".join(table_lines))

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"{title}\n{subtitle}\n\n")
        f.write("\n".join(table_lines) + "\n")
    print(f"\nSaved to: {OUTPUT_TXT}")

    # ---- The equation the model learned ----
    equation = raw_equation(weights, bias, means, stds, feature_columns)
    print(f"\n{LINE}\nEQUATION USED (trained on all completed seasons)\n{LINE}")
    for name, number in equation.items():
        print(f"  {name:<24}{number:>10.4f}")

    print(
        "\nNotes: rookies and players with no earlier NBA season can't be "
        "predicted.\nRetirements, long injuries and role changes are not "
        "accounted for."
    )


if __name__ == "__main__":
    main()
