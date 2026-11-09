"""
features.py  (file 2 of 4)

Builds the model's inputs. For each player-season we look at the player's
PAST seasons and blend them into "weighted" stats (recent seasons count more).
"""

import numpy as np
import pandas as pd

from data_loader import SEASON_ORDER


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

# Stats we pull from past seasons.
# (G is only used to tell which past seasons a player actually played.)
FEATURES_TO_AVERAGE = [
    "PTS", "MP", "FGA", "FTA", "G",
    "eFG%", "FT%", "Games_Missed", "Age"
]

# How much each past season counts, newest season first.
#   "two"  = weights when the player has 2 past seasons
#   "top"  = weights for the 2 newest seasons when there are 3 or more
#   "tail" = total weight shared by all older seasons
# With 1 past season the weight is simply 1.0.
WEIGHT_SCHEMES = {
    "current":      {"two": [0.75, 0.25], "top": [0.50, 0.45], "tail": 0.05},
    "recent_heavy": {"two": [0.90, 0.10], "top": [0.75, 0.20], "tail": 0.05},
    "equal":        None,    # every past season counts the same
}


def get_history_weights(number_of_seasons, scheme="current"):
    """Return the weights for a player's past seasons, newest first."""
    if number_of_seasons <= 0:
        return []
    if number_of_seasons == 1:
        return [1.0]

    settings = WEIGHT_SCHEMES[scheme]
    if settings is None:
        return [1.0 / number_of_seasons] * number_of_seasons

    if number_of_seasons == 2:
        return list(settings["two"])

    # 3 or more seasons: two big weights, then the old seasons split the tail.
    older_weight = settings["tail"] / (number_of_seasons - 2)
    return list(settings["top"]) + [older_weight] * (number_of_seasons - 2)


# ------------------------------------------------------------
# STEP 1: ATTACH EACH PLAYER'S PAST SEASONS
# ------------------------------------------------------------

def attach_previous_seasons(nba_data):
    """Add columns like Prev1_PTS (last season) and Prev2_PTS (2 seasons ago)."""
    model_data = nba_data.copy()

    for lag in range(1, len(SEASON_ORDER)):
        lag_column = f"Previous_Season_{lag}"

        # For each season, which season was `lag` seasons earlier?
        earlier_season = {
            SEASON_ORDER[i]: SEASON_ORDER[i - lag]
            for i in range(lag, len(SEASON_ORDER))
        }
        model_data[lag_column] = model_data["Season"].map(earlier_season)

        # Rename the stats to Prev{lag}_... and join them on player + season.
        history = nba_data[["Player", "Season"] + FEATURES_TO_AVERAGE].rename(
            columns={
                "Season": lag_column,
                **{stat: f"Prev{lag}_{stat}" for stat in FEATURES_TO_AVERAGE},
            }
        )
        model_data = model_data.merge(
            history, on=["Player", lag_column], how="left"
        )

    # Keep only players who played last season.
    return model_data.dropna(subset=["Prev1_PTS", "Prev1_G"]).copy()


# ------------------------------------------------------------
# STEP 2: BLEND THE PAST SEASONS INTO "WEIGHTED" STATS
# ------------------------------------------------------------

def add_weighted_features(model_data, scheme="current"):
    """Add Weighted_<stat> columns: a weighted average of past seasons."""
    lags = range(1, len(SEASON_ORDER))

    # Which past seasons did each player actually play?
    games = model_data[[f"Prev{lag}_G" for lag in lags]].to_numpy(dtype=float)
    played = ~np.isnan(games) & (games > 0)

    # Build each player's weights (0 for seasons they didn't play).
    weight_matrix = np.zeros(games.shape)
    for row in range(len(model_data)):
        played_positions = np.where(played[row])[0]
        weight_matrix[row, played_positions] = get_history_weights(
            len(played_positions), scheme
        )

    has_history = played.any(axis=1)

    for stat in FEATURES_TO_AVERAGE:
        values = model_data[[f"Prev{lag}_{stat}" for lag in lags]]
        values = np.where(played, values.to_numpy(dtype=float), 0.0)

        weighted = (weight_matrix * values).sum(axis=1)
        model_data[f"Weighted_{stat}"] = np.where(has_history, weighted, np.nan)

    return model_data


# ------------------------------------------------------------
# STEP 3: BUILD EVERYTHING
# ------------------------------------------------------------

def add_season_stats(nba_data):
    """Work out shooting efficiency for each player-season."""
    # Effective FG% = (FG + 0.5 * 3P) / FGA
    nba_data["eFG%"] = (nba_data["FG"] + 0.5 * nba_data["3P"]) / nba_data["FGA"]
    nba_data["FT%"] = nba_data["FT"] / nba_data["FTA"]

    # Dividing by 0 gives infinity or blank. Set those to 0.
    nba_data[["eFG%", "FT%"]] = (
        nba_data[["eFG%", "FT%"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    return nba_data


def build_features(nba_data, verbose=False, use_position_orb=False,
                   weight_scheme="current"):
    """Return (model_data, feature_columns).

    model_data      = one row per player-season with all inputs
    feature_columns = the inputs the model uses

    (use_position_orb is not used; it is only kept so older scripts still work.)
    """
    nba_data = add_season_stats(nba_data.copy())

    missing = sorted(set(FEATURES_TO_AVERAGE + ["Season", "Player"])
                     - set(nba_data.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    model_data = attach_previous_seasons(nba_data)
    model_data = add_weighted_features(model_data, weight_scheme)

    # Combined inputs (shots x how well they go in).
    model_data["Weighted_FGA_x_eFG"] = (
        model_data["Weighted_FGA"] * model_data["Weighted_eFG%"]
    )
    model_data["Weighted_FTA_x_FT"] = (
        model_data["Weighted_FTA"] * model_data["Weighted_FT%"]
    )

    # The inputs the model uses:
    # Predicted PPG = B0 + B1*(FGA x eFG%) + B2*(FTA x FT%) + B3*Age
    #                 + B4*PPG last season + B5*MP
    #                 + B6*games missed last season + B7*eFG%
    feature_columns = [
        "Weighted_FGA_x_eFG",
        "Weighted_FTA_x_FT",
        "Weighted_Age",
        "Prev1_PTS",
        "Weighted_MP",
        "Prev1_Games_Missed",
        "Weighted_eFG%"
    ]

    # Replace any infinity or blank values with 0.
    model_data[feature_columns] = (
        model_data[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    if verbose:
        print("\nFeature engineering complete")
        print("Total model records:", len(model_data))
        print("Feature count:", len(feature_columns))
        print("\nRecords by season:")
        print(model_data.groupby("Season")["Player"].count())
        print("\nFeature preview:")
        print(
            model_data[["Player", "Season", "PTS"] + feature_columns]
            .head(10).to_string(index=False)
        )

    return model_data, feature_columns
