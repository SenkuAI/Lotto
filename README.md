# 🎱 Lotto Lab

Machine-learning picks for every PCSO game (6/58, 6/55, 6/49, 6/45, 6/42, 6D, 4D, Swertres, EZ2), one card per game and draw time:

1. the **latest result**,
2. the **5 combinations made before that draw**, scored against it,
3. **5 new combinations** for the next draw,
4. an honest **model check** against plain chance.

Picks are a pure function of the results, so they only change when a new result arrives.

> Lottery draws are random. The model check on every card shows whether the model beats random picks; so far it doesn't. For entertainment only. Not affiliated with PCSO.

## How it works

| File | Role |
| --- | --- |
| `app.py` | Streamlit page |
| `lotto_core.py` | Game rules, data loading/cleaning, draw-time slots |
| `lotto_model.py` | Features, gradient boosting + logistic regression, combination builder, hold-out test |
| `scripts/build_data.py` | Rebuilds `data/draws.csv.gz` from `Lotto results.xlsx` (repairs combos Excel turned into dates) |

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the data from the spreadsheet

```powershell
pip install openpyxl
python scripts/build_data.py
```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → repo `Lotto`, branch `main`, main file path `app.py`.
3. Deploy. No secrets are needed. The first load trains the models (a few seconds).

## Data

`data/draws.csv.gz` (2016 to date) ships with the repo. On load, the newest rows from the public
[`lottotayo/pcso-lotto-results`](https://huggingface.co/datasets/lottotayo/pcso-lotto-results) dataset are merged in; if that fails the bundled file is used.
