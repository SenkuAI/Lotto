# 🎱 Lotto Lab

One big, simple page of PCSO lotto picks. Every game (6/58, 6/55, 6/49, 6/45, 6/42, 6D, 4D, Swertres, EZ2) and every draw time gets a card showing:

- the **latest result**,
- **our last pick** and how many numbers it matched,
- the **pick for the next draw** (plus backups).

Picks are calculated only from past results, so they stay the same until a new result arrives. Then the old pick is checked against it and a new one is made.

The pick blends long-run frequency, recent trend, short-term trend and overdue numbers, then applies a balance filter (odd/even, high/low, typical sum, no long runs of consecutive numbers).

> Every draw is random. No method can guarantee a win. Not affiliated with PCSO.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → repo `Lotto`, branch `main`, main file path `app.py`.
3. Deploy. No secrets are needed.

## Data

`data/seed_draws.csv.gz` ships with the repo. On load (cached 30 minutes) the app adds the newest rows from the public
[`lottotayo/pcso-lotto-results`](https://huggingface.co/datasets/lottotayo/pcso-lotto-results) dataset, and falls back to the bundled file if that fails.
