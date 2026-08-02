# NLPPlayground — Phase 1 MVP

Interactive Python & NLP learning app. Built with Streamlit, NLTK, and TextBlob.

## What's in this build

- **Guided Learning → Module 1: Python Basics** — 3 interactive lessons (variables/strings, lists/loops, functions), each with editable input, live output, and downloadable code.
- **Quick Tools** — Sentiment Analysis, Keyword Extraction, Named Entity Recognition. Paste your own text or use a bundled sample movie-review dataset.
- Responsive, mobile-friendly layout with a custom-styled header and cards.
- Verified: installs cleanly, boots without errors, all three NLP tools tested end-to-end.

**Not in this build yet** (by design — see the phased roadmap): topic modeling, clustering, classification, login/accounts, AI-explain button, multi-text comparison, batch/URL upload. These are Iteration 2+ per the plan.

## Run locally

```bash
cd nlpplayground
pip install -r requirements.txt
streamlit run app.py
```

First run downloads a few small NLTK data packages automatically (cached after that).

## Deploy free — Streamlit Community Cloud (~10 minutes)

1. **Create a GitHub repo** (e.g. `nlpplayground`) and push this folder:
   ```bash
   cd nlpplayground
   git init
   git add .
   git commit -m "NLPPlayground Phase 1 MVP"
   git branch -M main
   git remote add origin https://github.com/<your-username>/nlpplayground.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"** → select your `nlpplayground` repo → branch `main` → main file path `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` and launches automatically.
5. You'll get a public URL like `https://nlpplayground-<random>.streamlit.app` — share this immediately.

**Note:** the free tier app "sleeps" after ~12 hours of no traffic. A visitor loading the URL wakes it back up in under a minute — no action needed from you.

## Known limitations (honest notes)

- NLTK's built-in NER (`ne_chunk`) is lightweight and sometimes misclassifies entities (e.g. a company name tagged as PERSON). This is flagged in-app. It's fine for teaching, not for production-grade extraction — Phase 2+ can add spaCy or transformer-based NER.
- No data is stored or persisted — every session is stateless (by design for Phase 1; no backend yet).
- Free-tier LLM/AI features (from our roadmap discussion) are **not** included in this build — they need API keys and a bit more testing time than this 12-hour window allows responsibly.

## Next iteration (per the phased plan)

Iteration 2: Module 2 (Python for Text), preprocessing/tokenization/lemmatization lessons, Topic Modeling + Clustering + Classification quick tools, more sample datasets (AG News, CoNLL-NER, BBC).
