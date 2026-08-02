# NLPPlayground — Phase 1 MVP

Interactive Python, R & NLP learning app. Built with Streamlit, NLTK, and TextBlob.

## What's in this build

- **Language track selector** — 🐍 Python (live-executed) or Ⓡ R (real code + precomputed, hand-verified example output — this app runs on Python, so R isn't executed live).
- **Guided Learning → Module 1: Fundamentals** — 8 lessons per language track (What is Python/R?, Variables & Types, Strings & Text, Lists/Vectors & Loops, Conditionals, Functions, Dictionaries/Named Lists, DataFrames/Data Frames). Each lesson has a plain-language analogy, a "why this matters for NLP" expander, a "common mistakes" expander, editable input (Python) or a labeled worked example (R), and downloadable code.
- **Quick Tools** — Sentiment Analysis, Keyword Extraction, Named Entity Recognition, in both Python (live) and R (worked-example) versions. Paste your own text (Python) or use a bundled sample movie-review dataset.
- Responsive, mobile-friendly layout with a custom-styled header and cards.
- Verified: installs cleanly, boots without errors; all Python tools tested end-to-end; every R example output hand-computed and checked before being written in — nothing fabricated.

**Not in this build yet** (by design — see the phased roadmap): topic modeling, clustering, classification, login/accounts, AI-explain button, multi-text comparison, batch/URL upload, live R execution. These are later iterations per the plan.

## Run locally

**Python:**
```bash
cd nlpplayground
pip install -r requirements.txt
streamlit run app.py
```
First run downloads a few small NLTK data packages automatically (cached after that).

**R** (only needed if you want to run the downloaded `.R` snippets yourself — the app itself doesn't require R):
1. Install R from [r-project.org](https://www.r-project.org/) or via your OS package manager.
2. Some Quick Tools snippets need one extra package: `install.packages("stringr")` (the snippet includes this line as a comment/reminder at the top).
3. Run the downloaded `.R` file in RStudio, or from a terminal: `Rscript your_downloaded_file.R`.

## Running the downloaded code snippets in Google Colab

Every lesson and tool has a "📥 Download this code" button. Here's what happens when you take that file to Colab:

**Python snippets:**
- Lessons 0–7 (Fundamentals) use only Python's standard library — paste into a Colab cell and run, no setup needed.
- The Sentiment / Keyword Extraction / NER Quick Tools need one extra setup step first — each downloaded snippet includes the exact `pip install` or `nltk.download()` lines as comments at the top. Uncomment and run those once per Colab session.

**R snippets:**
- Colab's default runtime is Python. To run R, either:
  - Go to **Runtime → Change runtime type → R** in an existing notebook, or
  - Open **[colab.to/r](https://colab.to/r)** directly for a fresh R-only notebook.
- Fundamentals lessons (0–7) use only base R — no extra installs needed once the R runtime is selected.
- The Sentiment and Keyword Extraction Quick Tools use the `stringr` package — run `install.packages("stringr")` first if it's not already available in your Colab R environment (the snippet includes this as a comment). The NER tool uses only base R.
- We haven't been able to verify exactly which R packages ship preinstalled in Colab's R runtime by default, so the `install.packages()` line is included defensively — it's a harmless no-op if the package is already there.

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
- **R code is not live-executed.** This app runs on Python (Streamlit); R output shown is a precomputed, hand-verified worked example, clearly labeled as such in-app. Real interactive R execution (via an R runtime + subprocess) was deliberately deferred — it adds real deploy risk, similar to the spaCy-vs-NLTK tradeoff made for the Python NER tool.
- R Quick Tools use a small, hard-coded illustrative lexicon/logic (not a real package like `tidytext`'s `bing`/`afinn` lexicons) specifically so every output could be verified by hand rather than guessed.
- No data is stored or persisted — every session is stateless (by design for Phase 1; no backend yet).
- Free-tier LLM/AI features (from our roadmap discussion) are **not** included in this build — they need API keys and a bit more testing time than this 12-hour window allows responsibly.

## Next iteration (per the phased plan)

Iteration 2: preprocessing/tokenization/lemmatization deep-dive lessons, Topic Modeling + Clustering + Classification quick tools (Python and R), more sample datasets (AG News, CoNLL-NER, BBC), and evaluating whether live R execution is worth the added deploy complexity.
