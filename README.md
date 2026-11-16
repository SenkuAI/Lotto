# 🎱 Lotto Lab

A Streamlit app for exploring PCSO lotto results: latest draws, number statistics, jackpot history, a ticket checker, a quick-pick generator and odds for every game (6/58, 6/55, 6/49, 6/45, 6/42, 6D, 4D, Swertres, EZ2).

> Every draw is independent. The statistics describe history; they cannot predict the next draw. Not affiliated with PCSO.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → pick this repo (`Lotto`), branch `main`, main file path `app.py`.
3. Deploy. No secrets are needed.

## Data

- `data/seed_draws.csv.gz` ships with the repo so the app always works.
- On load (cached for 1 hour) the app tops it up with the newest rows from the public
  [`lottotayo/pcso-lotto-results`](https://huggingface.co/datasets/lottotayo/pcso-lotto-results) dataset.
  If that request fails, the bundled data is used.
