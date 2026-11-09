"""
model.py  (file 3 of 4)

The prediction model (linear regression trained with gradient descent)
plus helpers that build a "top 10 scorers" list and compare it to the real one.

    predicted PPG = intercept + (number x input) + (number x input) + ...
"""

import unicodedata

import numpy as np
import pandas as pd

from data_loader import ANSWERS_FOLDER


# ------------------------------------------------------------
# NAME CLEANING (so names match between files)
# ------------------------------------------------------------

def normalize_player_name(name):
    """Make names comparable: no accents, no spaces, lowercase.

    "Luka Dončić" and "Luka Doncic" both become "lukadoncic".
    """
    # Split accented letters (ć -> c + accent mark), then drop the marks.
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Keep only letters and numbers.
    return "".join(c.lower() for c in name if c.isalnum())


# ------------------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------------------

# How many gradient-descent steps to take.
ITERATIONS = 10000

# Ridge penalty. 0 = plain regression (no penalty).
L2_PENALTY = 0.0


# ------------------------------------------------------------
# THE MODEL
# ------------------------------------------------------------

def train_linear_regression(X, y, learning_rate=0.00001, iterations=None,
                            nonpositive_idx=(), l2=None):
    """Learn the weights and bias from training data.

    X = inputs (one row per player-season)
    y = right answers (that player's actual PPG)
    nonpositive_idx = positions of inputs whose weight can never go above 0

    Gradient descent repeatedly nudges the numbers to shrink the total
    squared error. (The error is summed, not averaged, which is why the
    learning rate is so small.)
    """
    if iterations is None:
        iterations = ITERATIONS
    if l2 is None:
        l2 = L2_PENALTY
    nonpositive_idx = list(nonpositive_idx)

    # Put every input on the same scale: (value - average) / spread.
    # A column that never changes has spread 0, so use 1 to avoid dividing by 0.
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1
    X_scaled = (X - means) / stds

    weights = np.zeros(X.shape[1])    # start every weight at 0
    bias = 0.0                        # start the intercept at 0
    rows = len(y)

    for _ in range(iterations):
        predictions = X_scaled @ weights + bias
        errors = predictions - y                 # prediction minus the truth

        # Gradient = the direction of MORE error, so we step the other way.
        weight_gradient = X_scaled.T @ errors + l2 * rows * weights
        bias_gradient = np.sum(errors)

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

        # Sign limit: a limited weight that goes above 0 is pushed back to 0.
        if nonpositive_idx:
            weights[nonpositive_idx] = np.minimum(weights[nonpositive_idx], 0.0)

    # Return means and stds so new data can be scaled the same way.
    return weights, bias, means, stds


def predict_ppg(X, weights, bias, means, stds):
    """Predict PPG for new rows (scaled with the TRAINING averages)."""
    X_scaled = (X - means) / stds
    return X_scaled @ weights + bias


def raw_equation(weights, bias, means, stds, feature_columns):
    """Convert the learned numbers back to the original units.

    Returns {"Intercept": ..., "<input name>": ..., ...}
    meaning: predicted PPG = Intercept + sum(number x input)
    """
    numbers = weights / stds
    intercept = bias - np.sum(weights * means / stds)

    equation = {"Intercept": intercept}
    for name, number in zip(feature_columns, numbers):
        equation[name] = number
    return equation


# ------------------------------------------------------------
# TOP-10 LIST HELPERS
# ------------------------------------------------------------

def pick_predicted_top10(test_data, predictions):
    """Return the 10 players with the highest predicted PPG.

    Players tied for 10th are all included, so the list can be longer than 10.
    """
    table = test_data[["Player", "PTS"]].copy()
    table["Predicted_PPG"] = np.round(predictions, 1)
    table["Name_Key"] = table["Player"].apply(normalize_player_name)

    # Ties share a rank (two players tied for 4th both get 4; next gets 6).
    table["Predicted_Rank"] = (
        table["Predicted_PPG"].rank(method="min", ascending=False).astype(int)
    )

    top10 = table[table["Predicted_Rank"] <= 10]
    return top10.sort_values(
        ["Predicted_Rank", "Predicted_PPG"], ascending=[True, False]
    )


def load_actual_top10(test_season):
    """Read the real top scorers for a season from the answers folder."""
    answer_file = ANSWERS_FOLDER / f"{test_season}.txt"

    if not answer_file.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_file}")

    actual = pd.read_csv(answer_file, skiprows=3)    # first 3 lines are titles
    actual.columns = actual.columns.str.strip()

    required = ["Rank", "Player", "PTS"]
    missing = [c for c in required if c not in actual.columns]
    if missing:
        raise ValueError(
            f"{answer_file.name} is missing columns: {missing}. "
            f"Found: {list(actual.columns)}"
        )

    actual["Name_Key"] = actual["Player"].apply(normalize_player_name)
    return actual


def compare_top10(predicted_top10, actual_top10):
    """Compare our top 10 with the real top 10.

    Returns a dictionary with:
        correct_picks       - players in BOTH lists
        pct_picks_correct   - correct picks / size of our list
        pct_real_found      - correct picks / size of the real list
        correct_picks_miss  - average PPG miss on the correct picks
    """
    # An "inner" merge keeps only players found in both lists.
    matches = predicted_top10.merge(
        actual_top10[["Name_Key", "PTS"]],
        on="Name_Key",
        how="inner",
        suffixes=("_predicted", "_actual")
    )
    correct_picks = len(matches)

    if correct_picks > 0:
        miss = np.mean(np.abs(matches["Predicted_PPG"] - matches["PTS_actual"]))
    else:
        miss = np.nan

    return {
        "correct_picks": correct_picks,
        "pct_picks_correct":
            correct_picks / len(predicted_top10) if len(predicted_top10) else 0,
        "pct_real_found":
            correct_picks / len(actual_top10) if len(actual_top10) else 0,
        "correct_picks_miss": miss,
    }
