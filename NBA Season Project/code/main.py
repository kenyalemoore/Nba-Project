
"""
python main.py
 
PURPOSE
    Test how well the model predicts points per game (PPG) by "predicting"
    seasons that have already happened (2020-21 through 2025-26).
 
HOW IT WORKS
    1. Load and clean the season data           (data_loader.py)
    2. Build each player's weighted-history inputs   (features.py)
    3. For each test season, train ONLY on earlier seasons, 
    then predict that season                      (model.py)
    4. Compare our predicted top 10 scorers with the real top 10
    5. Compare our average miss to a simple baseline:
       "every player scores what he did last year"
 
OUTPUT
    A side-by-side top 10 for each season, a results table, and a summary
    showing whether the model beat the baseline.
 
FOLDER SETUP
    All .py files in one folder, next to:
        data/      one Excel file per season
        answers/   one .txt file per season with the real top scorers
 
To predict the 2026-27 season, run predict_2026_27.py instead.
"""

import numpy as np
import pandas as pd

from data_loader import SEASON_ORDER, load_all_seasons, check_data_quality
from features import build_features, NONPOSITIVE_FEATURES
from model import (
    train_linear_regression,
    predict_ppg,
    pick_predicted_top10,
    load_actual_top10,
    compare_top10
)


# True = also print data checks and feature previews.
SHOW_SETUP_DETAILS = False

# How much each past season counts: "current", "recent_heavy" or "equal"
# (the weights are listed in features.py).
WEIGHT_SCHEME = "current"

# A player needs this many games to be allowed in a predicted top 10.
MIN_GAMES_FOR_TOP10 = 58

LINE = "=" * 78

def print_side_by_side(predicted_top10, actual_top10):
    """Print our list next to the real list. A '*' means the player is in both."""
    real_names = set(actual_top10["Name_Key"])
    predicted_names = set(predicted_top10["Name_Key"])

    left_lines = []
    for _, row in predicted_top10.iterrows():
        mark = "*" if row["Name_Key"] in real_names else " "
        left_lines.append(
            f"{mark} {int(row['Predicted_Rank']):>2}. "
            f"{row['Player']:<24}{row['Predicted_PPG']:>5.1f}"
        )

    right_lines = []
    for _, row in actual_top10.iterrows():
        mark = "*" if row["Name_Key"] in predicted_names else " "
        right_lines.append(
            f"{mark} {int(row['Rank']):>2}. "
            f"{row['Player']:<24}{row['PTS']:>5.1f}"
        )

    # Pad the shorter list so the columns line up.
    height = max(len(left_lines), len(right_lines))
    left_lines += [""] * (height - len(left_lines))
    right_lines += [""] * (height - len(right_lines))

    print(f"  {'OUR PREDICTION (PPG)':<36}{'REAL RESULT (PPG)'}")
    print(f"  {'-' * 34}  {'-' * 34}")
    for left, right in zip(left_lines, right_lines):
        print(f"  {left:<36}{right}")
    print("  (* = player is in both lists)")

def test_one_season(test_season, model_data, feature_columns):
    """Train on earlier seasons, predict `test_season`, and score the result.

    This is "walk-forward" testing: we only train on seasons BEFORE the one
    we predict, like real life where you can't see the future.
    """
    # 2018-19 has no earlier season, so training starts at the 2nd season.
    training_seasons = SEASON_ORDER[1:SEASON_ORDER.index(test_season)]
    train_data = model_data[model_data["Season"].isin(training_seasons)]
    test_data = model_data[model_data["Season"] == test_season].copy()

    if train_data.empty or test_data.empty:
        print(f"Skipping {test_season}: training or test data is empty.")
        return None

    print(f"\n{LINE}")
    print(test_season.upper())
    print(
        f"Trained on {len(train_data)} player-seasons "
        f"from {len(training_seasons)} earlier season(s); "
        f"predicted {len(test_data)} players"
    )
    print(LINE)

    # X = inputs, y = the right answer (actual PPG).
    X_train = train_data[feature_columns].to_numpy(dtype=float)
    y_train = train_data["PTS"].to_numpy(dtype=float)
    X_test = test_data[feature_columns].to_numpy(dtype=float)
    y_test = test_data["PTS"].to_numpy(dtype=float)

    # Train (some inputs may never get a positive weight), then predict.
    limited_idx = [
        feature_columns.index(name)
        for name in NONPOSITIVE_FEATURES if name in feature_columns
    ]
    weights, bias, means, stds = train_linear_regression(
        X_train, y_train, nonpositive_idx=limited_idx
    )
    predictions = predict_ppg(X_test, weights, bias, means, stds)

    # ---- How far off were we across ALL players? ----
    model_avg_miss = np.mean(np.abs(predictions - y_test))
    model_big_miss = np.sqrt(np.mean((predictions - y_test) ** 2))  # big errors count extra

    # The "lazy guess" to beat: every player scores what he did last year.
    last_year_guess = test_data["Prev1_PTS"].to_numpy(dtype=float)
    last_year_avg_miss = np.mean(np.abs(last_year_guess - y_test))

    # ---- How good was our top 10? ----
    # Only players with enough actual games can fill a top-10 slot.
    # (This does not affect training or the average miss above.)
    eligible = test_data["G"].to_numpy() >= MIN_GAMES_FOR_TOP10
    predicted_top10 = pick_predicted_top10(
        test_data.loc[eligible].copy(), predictions[eligible]
    )
    actual_top10 = load_actual_top10(test_season)
    top10 = compare_top10(predicted_top10, actual_top10)

    print_side_by_side(predicted_top10, actual_top10)

    print(
        f"\n  Picked correctly: {top10['correct_picks']} of "
        f"{len(predicted_top10)} ({top10['pct_picks_correct'] * 100:.0f}%)"
    )
    print(
        f"  Real top scorers found: {top10['correct_picks']} of "
        f"{len(actual_top10)} ({top10['pct_real_found'] * 100:.0f}%)"
    )
    if not np.isnan(top10["correct_picks_miss"]):
        print(
            f"  Avg miss on the players we got right: "
            f"{top10['correct_picks_miss']:.2f} PPG"
        )
    print(
        f"  Avg miss, all {len(test_data)} players: {model_avg_miss:.2f} PPG "
        f"(last-year guess: {last_year_avg_miss:.2f} PPG)"
    )

    result_row = {
        "Season": test_season,
        "Players_Tested": len(test_data),
        "Model_Avg_Miss_PPG": model_avg_miss,
        "Model_Big_Miss_Score": model_big_miss,
        "Last_Year_Guess_Avg_Miss_PPG": last_year_avg_miss,
        "Top10_Players_Correct": top10["correct_picks"],
        "Our_List_Size": len(predicted_top10),
        "Real_List_Size": len(actual_top10),
        "Pct_Of_Our_Picks_Correct": top10["pct_picks_correct"] * 100,
        "Pct_Of_Real_Top10_Found": top10["pct_real_found"] * 100,
        "Avg_PPG_Miss_On_Correct_Picks": top10["correct_picks_miss"]
    }
    return result_row

def print_final_results(results_df):
    """Print one table (one row per season) and a short summary."""
    if results_df.empty:
        print("No validation results were produced.")
        return

    print(f"\n\n{LINE}")
    print("RESULTS BY SEASON")
    print(LINE)
    print(
        f"{'Season':<10}{'Players':>9}{'Model':>9}{'Last-Year':>12}"
        f"{'Model':>10}{'Top-10':>10}{'Picks':>9}{'Real Top-10':>14}"
    )
    print(
        f"{'':<10}{'Tested':>9}{'Miss':>9}{'Guess Miss':>12}"
        f"{'Better?':>10}{'Correct':>10}{'Right':>9}{'Found':>14}"
    )
    print("-" * 83)

    for _, row in results_df.iterrows():
        beat_guess = row["Model_Avg_Miss_PPG"] < row["Last_Year_Guess_Avg_Miss_PPG"]
        correct_text = (
            f"{int(row['Top10_Players_Correct'])} of {int(row['Our_List_Size'])}"
        )
        print(
            f"{row['Season'].replace(' Season', ''):<10}"
            f"{int(row['Players_Tested']):>9}"
            f"{row['Model_Avg_Miss_PPG']:>9.2f}"
            f"{row['Last_Year_Guess_Avg_Miss_PPG']:>12.2f}"
            f"{'Yes' if beat_guess else 'No':>10}"
            f"{correct_text:>10}"
            f"{row['Pct_Of_Our_Picks_Correct']:>8.0f}%"
            f"{row['Pct_Of_Real_Top10_Found']:>13.0f}%"
        )

    # ---- Averages across all test seasons ----
    avg_model_miss = results_df["Model_Avg_Miss_PPG"].mean()
    avg_last_year_miss = results_df["Last_Year_Guess_Avg_Miss_PPG"].mean()
    avg_correct = results_df["Top10_Players_Correct"].mean()
    avg_picks_right = results_df["Pct_Of_Our_Picks_Correct"].mean()
    avg_real_found = results_df["Pct_Of_Real_Top10_Found"].mean()
    seasons_won = (
        results_df["Model_Avg_Miss_PPG"]
        < results_df["Last_Year_Guess_Avg_Miss_PPG"]
    ).sum()

    print("-" * 83)
    print(
        f"{'AVERAGE':<10}{'':>9}"
        f"{avg_model_miss:>9.2f}"
        f"{avg_last_year_miss:>12.2f}"
        f"{f'{seasons_won} of {len(results_df)}':>10}"
        f"{avg_correct:>10.1f}"
        f"{avg_picks_right:>8.0f}%"
        f"{avg_real_found:>13.0f}%"
    )

    print("\nHow to read the table:")
    print("  Model Miss        = average PPG our prediction was off (lower is better)")
    print("  Last-Year Guess   = the same, if we assumed every player repeats last")
    print("                      year. We want to beat this.")
    print("  Top-10 Correct    = how many of our top scorers were in the real top 10")
    print("                      (ties can make a list longer than 10)")
    print("  Picks Right       = % of our picks that were correct")
    print("  Real Top-10 Found = % of the real top scorers that we picked")

    # ---- Summary ----
    difference = avg_last_year_miss - avg_model_miss

    print(f"\n\n{LINE}")
    print("SUMMARY")
    print(LINE)
    print(f"  Our model's average miss:           {avg_model_miss:.2f} points per game")
    print(f"  'Same as last year' average miss:   {avg_last_year_miss:.2f} points per game")
    if difference > 0:
        print(f"  Our model is more accurate by:      {difference:.2f} points per game")
    else:
        print(f"  Our model is LESS accurate by:      {abs(difference):.2f} points per game")
    print(f"  Seasons our model beat the guess:   {seasons_won} of {len(results_df)}")
    print(f"  Real top-10 scorers we picked:      {avg_correct:.1f} out of 10 per season")

def main():
    nba_data = load_all_seasons(verbose=SHOW_SETUP_DETAILS)
    if SHOW_SETUP_DETAILS:
        check_data_quality(nba_data)

    print(
        f"Loaded {len(nba_data)} player-seasons from "
        f"{nba_data['Season'].nunique()} seasons."
    )

    model_data, feature_columns = build_features(
        nba_data,
        verbose=SHOW_SETUP_DETAILS,
        weight_scheme=WEIGHT_SCHEME,
    )

    # Test every season from 2020-21 on (2018-19 and 2019-20 are training only).
    results = []
    for test_season in SEASON_ORDER[2:]:
        result_row = test_one_season(test_season, model_data, feature_columns)
        if result_row is not None:
            results.append(result_row)

    print_final_results(pd.DataFrame(results))


if __name__ == "__main__":
    main()
