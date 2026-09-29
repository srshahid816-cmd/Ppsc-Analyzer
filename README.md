# PPSC Examiner AI — Auto Past-Paper Engine

A Streamlit dashboard that automatically discovers public PPSC past-paper pages, downloads readable material, ranks it by job relevance and recency, analyzes recurring concepts, and asks an AI model to generate a fresh 25/50/100-MCQ mock paper.

## What it does

1. Enter the PPSC post/job name.
2. Click **Automatically Find Past Papers**.
3. The app checks configured public PPSC archives; no manual PDF upload is required.
4. It ranks material using post/job terms + paper relevance + year, with newer papers weighted more heavily.
5. Pattern Lab shows the local corpus and frequent terms.
6. Paper Maker generates a new paper with:
   - 4 options
   - answer
   - explanation
   - subject/topic
   - difficulty
   - evidence confidence
   - source basis
7. Dashboard summarizes repeated concepts, transfer questions, and new concepts.

## Important

The archive websites are third-party public sources, not the official PPSC commission. Their pages can change or become unavailable. The app therefore records the source URL and does not call any future question a guarantee.

## Setup

```bash
pip install -r requirements.txt
```

Set an API key and model, for example:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="YOUR_SUPPORTED_MODEL"
streamlit run app.py
```

For Streamlit Cloud, put the key in **Settings → Secrets** and run the app normally.

## GitHub

Push `app.py`, `requirements.txt`, `README.md`, and the `data/` folder. Do not commit your API key.
