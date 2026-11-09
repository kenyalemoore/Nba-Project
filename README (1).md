# NBA Points Per Game Predictor

A Python program that uses **linear regression** to predict NBA players' points per game (PPG). It is tested on past seasons (2020-21 through 2025-26) and then used to predict the top scorers for the **2026-27 season**.

**Author:** Kenyale Moore (individual midterm project)

## What it does

- Loads per-game stats from the 2018-19 through 2025-26 seasons and builds one clean row per player per season.
- Predicts each player's PPG from his past seasons, with recent seasons counting more than older ones.
- Tests the model with **walk-forward testing**: for each season, it trains only on earlier seasons, then predicts that season, just like real life where you can't see the future.
- Compares each predicted top 10 with the real top 10, and compares the model's average miss to a simple baseline: *"every player scores what he did last year."*
- Predicts the top 25 scorers for 2026-27 and saves them to a text file.

## Project files

| File | What it does |
|------|--------------|
| `data_loader.py` | Reads the Excel files, cleans them, and merges traded players into one row per season. Also holds shared settings (season order, folders). |
| `features.py` | Builds the model inputs from each player's past seasons. |
| `model.py` | The linear regression model (gradient descent) and the top-10 comparison helpers. |
| `main.py` | **Run this to test the model on past seasons.** |
| `predict_2026_27.py` | **Run this to predict the 2026-27 season.** |

## Setup

**Requirements:** Python 3.8+ and these packages:

```
pip install pandas numpy openpyxl
```

**Folder layout** (the `data` and `answers` folders must sit in the same folder as the `.py` files):

```
NBA Season Project/
└── code/
    ├── data_loader.py
    ├── features.py
    ├── model.py
    ├── main.py
    ├── predict_2026_27.py
    ├── data/
    │   ├── 2018-19 Season.xlsx
    │   ├── 2019-20 Season.xlsx
    │   └── ... (one file per season, through 2025-26)
    └── answers/
        ├── 2020-21 Season.txt
        └── ... (one file per test season)
```

### Data format

- **`data/`**: one Excel file per season, named exactly like `2018-19 Season.xlsx`. Each file has one sheet per team. Every sheet needs these columns: `Player`, `Pos`, `Age`, `G`, `GS`, `MP`, `FG`, `FGA`, `3P`, `3PA`, `2P`, `2PA`, `FT`, `FTA`, `ORB`, `DRB`, `TRB`, `AST`, `STL`, `BLK`, `TOV`, `PF`, `PTS` (per-game stats).
- **`answers/`**: one text file per test season, named like `2020-21 Season.txt`. The first 3 lines are skipped as titles, then the file is comma-separated with the columns `Rank`, `Player`, `PTS` (the real top scorers).

The data files are not included in this repository. You will need to add your own.

## How to run

Test the model on past seasons:

```
python main.py
```

Predict the 2026-27 season:

```
python predict_2026_27.py
```

The prediction is printed and saved to `predictions_2026-27.txt`.

## How the model works

The model is multiple linear regression trained with gradient descent:

```
Predicted PPG = B0 + B1*(FGA x eFG%) + B2*(FTA x FT%) + B3*Age
                + B4*PPG last season + B5*Minutes
                + B6*Games missed last season + B7*eFG%
```

- **Weighted history:** Stats like FGA, FTA, minutes, age and eFG% are averaged over a player's past seasons. The newest seasons count the most (50% / 45% for the two newest, and 5% shared by all older seasons). You can change this with `WEIGHT_SCHEME` (`"current"`, `"recent_heavy"` or `"equal"`).
- **eFG%** (effective field goal %) = (FG + 0.5 x 3P) / FGA.
- **Traded players:** Rows from different teams are merged into one row, averaged by games played with each team.
- **Training:** Inputs are scaled, then gradient descent runs for 10,000 steps (settings are in `model.py`).

## Evaluation

For each test season, `main.py` prints:

- a side-by-side predicted vs. real top 10,
- how many of the real top scorers the model found,
- the average miss across all players (in PPG) versus the "same as last year" baseline,
- a final results table and summary.

To count as a pick in the top 10, a player must have played at least 58 games that season (`MIN_GAMES_FOR_TOP10` in `main.py`). This rule does not affect training or the average miss.

## Limitations

- Rookies and players with no earlier NBA season can't be predicted.
- Injuries, trades, retirements and role changes that happen after the data ends are not accounted for.
- For 2026-27, a player needs at least 30 games last season to be listed (`MIN_GAMES_LAST_SEASON` in `predict_2026_27.py`), since future games played can't be known.
