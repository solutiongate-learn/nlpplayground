"""
NLPPlayground - Phase 1 MVP
Learn Python, R & NLP through interactive, code-visible exercises.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (see README.md)
"""

import streamlit as st
import pandas as pd
from collections import Counter
import re
import string
import matplotlib
matplotlib.use("Agg")  # headless backend — this app never opens a GUI window
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# NLTK setup (cached so it only downloads once per app instance)
# ---------------------------------------------------------------------------
import nltk

@st.cache_resource(show_spinner="Setting up NLP engine (first run only)...")
def setup_nltk():
    resources = [
        "punkt", "punkt_tab", "stopwords",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker", "maxent_ne_chunker_tab", "words",
        "vader_lexicon",
    ]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass  # some resource names vary by NLTK version; safe to skip
    return True

setup_nltk()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer

# ---------------------------------------------------------------------------
# spaCy setup — LAZY on purpose (only loads into memory on first actual use,
# not at app boot). The model itself is installed at BUILD time via a direct
# wheel URL in requirements.txt, NOT downloaded at runtime — Streamlit
# Cloud's running container has no write permission to install packages
# after deployment, so a runtime `pip install` / `spacy download` call
# fails silently and retries forever. This was a real bug we hit and fixed:
# see requirements.txt for the en_core_web_sm wheel dependency.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading spaCy model (first use this session)...")
def setup_spacy():
    import spacy
    return spacy.load("en_core_web_sm")


def get_spacy_model():
    """Call this only at the point spaCy is actually needed — never at import time."""
    try:
        return setup_spacy(), True
    except Exception:
        return None, False

# ---------------------------------------------------------------------------
# USAGE ANALYTICS (optional, fails silently if not configured)
#
# Design notes (so future edits don't accidentally break this):
#
# - Streamlit Community Cloud's local filesystem is EPHEMERAL. It resets on
#   every redeploy and on periodic container restarts. Anything written to a
#   local file or SQLite DB here will eventually disappear. To get numbers
#   that persist across restarts, we log events to a Google Sheet instead —
#   free, no server to run, and the user (Sudhanshu) can open it directly.
#
# - This entire block is OPTIONAL. If `st.secrets["gcp_service_account"]`
#   isn't configured (e.g. running locally without setting it up, or a fresh
#   clone), every function below degrades to a harmless no-op. Analytics
#   must never be able to crash the learning app for a real learner — that
#   would be a much worse outcome than losing a few log rows.
#
# - "Time spent" here is an ESTIMATE, not a precise measurement: it's the gap
#   between a session's first logged event and its most recent one. Streamlit
#   has no heartbeat/ping while a tab sits open with no interaction, so a
#   learner who opens a tab and reads quietly for 10 minutes without
#   clicking anything will not show 10 minutes of "time spent" — only actual
#   interaction gaps are visible. This estimate should be labeled as such
#   anywhere it's displayed, not presented as an exact figure.
#
# - Event log schema (one row per event, one flat sheet, easy to pandas
#   groupby): timestamp_utc, date, hour, session_id, event_type, detail
#     - event_type "session_start": once per new session_id
#     - event_type "page_view": logged whenever language/mode/track/lesson
#       navigation actually changes (not on every single rerun/widget tweak,
#       to avoid flooding the sheet)
#     - event_type "heartbeat": a lightweight "still here" ping logged
#       alongside page_view events, used only to compute the last-seen
#       timestamp for the time-spent estimate
# ---------------------------------------------------------------------------
import datetime as _dt
import uuid as _uuid

ANALYTICS_SHEET_NAME = "NLPPlayground Analytics"
ANALYTICS_WORKSHEET = "events"


def _safe_secret(key, default=None):
    """Reads one key from st.secrets without ever raising.

    st.secrets.get()/`in` both raise StreamlitSecretNotFoundError (not
    KeyError) when there is no secrets.toml file on disk AT ALL — which is
    the normal, expected state for local dev or a fresh clone that hasn't
    set up analytics yet. A plain dict-like .get() would silently swallow a
    missing KEY but not a missing FILE, so we wrap this explicitly to
    guarantee analytics/admin-view code never crashes the app either way.
    """
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def _get_analytics_worksheet():
    """Returns a gspread Worksheet, or None if analytics isn't configured.

    Cached as a resource so we only authenticate once per app instance, not
    on every single rerun (every widget interaction reruns the whole script).
    """
    try:
        creds_dict_raw = _safe_secret("gcp_service_account")
        if not creds_dict_raw:
            return None
        import gspread
        creds_dict = dict(creds_dict_raw)
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open(ANALYTICS_SHEET_NAME)
        try:
            ws = sh.worksheet(ANALYTICS_WORKSHEET)
        except Exception:
            # First run — sheet exists but the "events" tab doesn't yet.
            ws = sh.add_worksheet(title=ANALYTICS_WORKSHEET, rows=1000, cols=6)
            ws.append_row(
                ["timestamp_utc", "date", "hour", "session_id", "event_type", "detail"]
            )
        return ws
    except Exception:
        # Analytics must never break the app for a real learner. Any
        # misconfiguration (missing sheet, revoked access, bad credentials,
        # network hiccup) just silently disables logging for this session.
        return None


def _log_event(event_type, detail=""):
    """Append one row to the analytics sheet. No-ops safely if unconfigured."""
    ws = _get_analytics_worksheet()
    if ws is None:
        return
    try:
        now = _dt.datetime.utcnow()
        ws.append_row(
            [
                now.isoformat(timespec="seconds") + "Z",
                now.strftime("%Y-%m-%d"),
                now.strftime("%H"),
                st.session_state.get("_analytics_session_id", ""),
                event_type,
                detail,
            ]
        )
    except Exception:
        pass  # network hiccups, rate limits, etc. — never crash on this


def track_session_and_page(language, mode, track, lesson_label):
    """Call once per rerun from the main script body. Cheap no-op if the
    navigation state hasn't actually changed since the last rerun, so we
    don't flood the sheet with a row on every keystroke inside a text box.
    """
    if "_analytics_session_id" not in st.session_state:
        st.session_state["_analytics_session_id"] = _uuid.uuid4().hex[:12]
        _log_event("session_start")

    nav_fingerprint = f"{language} | {mode} | {track or ''} | {lesson_label or ''}"
    if st.session_state.get("_analytics_last_nav") != nav_fingerprint:
        st.session_state["_analytics_last_nav"] = nav_fingerprint
        _log_event("page_view", nav_fingerprint)


@st.cache_data(ttl=30, show_spinner=False)
def load_analytics_events():
    """Reads the full event log as a DataFrame for the admin view.
    Cached for 30s so the admin view doesn't hammer the Sheets API on every
    rerun while you're looking at it.
    """
    ws = _get_analytics_worksheet()
    if ws is None:
        return None
    try:
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(
                columns=["timestamp_utc", "date", "hour", "session_id", "event_type", "detail"]
            )
        return pd.DataFrame(records)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NLPPlayground",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Page title — plain text, no card/gradient box. (Previously this was a
       purple gradient banner; removed per explicit request — it read as an
       unnecessary decorative box rather than a real design choice.) */
    .page-title {
        margin-bottom: 1.2rem;
    }
    .page-title h1 {
        font-size: 2rem;
        margin-bottom: 0.2rem;
        color: inherit;
    }
    .page-title p {
        font-size: 1.05rem;
        opacity: 0.7;
        margin: 0;
    }

    /* Card style container.
       NOTE: this used to be `background: #ffffff10` (white at ~6% opacity),
       which was written assuming a dark page background. On the light theme
       this app actually uses, that resolved to an almost invisible fill —
       the card still reserved its full padding but visually looked like
       plain unstyled whitespace. Fixed to a real light tint that's visible
       against the white page background. */
    .card {
        background: #6C63FF0d;
        border: 1px solid rgba(108,99,255,0.18);
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.9rem;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-green { background: #16a34a22; color: #16a34a; }
    .badge-blue  { background: #2563eb22; color: #2563eb; }
    .badge-amber { background: #d9770622; color: #d97706; }
    .badge-nltk  { background: #2563eb22; color: #2563eb; }
    .badge-spacy { background: #7c3aed22; color: #7c3aed; }

    /* Library section divider */
    .lib-section {
        border-top: 2px solid rgba(120,120,120,0.15);
        margin-top: 1.5rem;
        padding-top: 1rem;
    }

    /* Module header block — gives every module a consistent, scannable identity */
    .module-head {
        border-left: 4px solid #6C63FF;
        padding: 0.2rem 0 0.2rem 1rem;
        margin-bottom: 1.4rem;
    }
    .module-head h2 {
        margin: 0.4rem 0 0.3rem 0;
        font-size: 1.75rem;
        line-height: 1.2;
    }
    .module-head p {
        margin: 0;
        opacity: 0.75;
        font-size: 0.97rem;
        max-width: 65ch;
    }

    /* Constrain long-form prose to a readable measure. The app runs in wide
       layout for the two-column code/output panes, but unconstrained body text
       stretches to unreadable line lengths on a large monitor. ~75 characters
       is the conventional upper bound for comfortable reading. */
    .main .block-container p,
    .main .block-container li {
        max-width: 78ch;
    }

    /* ---------------------------------------------------------------------
       Vertical rhythm tightening.
       Streamlit's default spacing between stacked elements is generous
       enough that a long lesson page reads as mostly empty space once you
       have a dozen elements stacked — a documented, known characteristic of
       Streamlit's default layout (see e.g. streamlit/streamlit#6867, a
       still-open feature request for a "compact layout" mode, and multiple
       community threads describing the out-of-the-box spacing as wasteful).

       NOTE ON SELECTOR CHOICE: `[data-testid="stMetric"]` and
       `[data-testid="stMetricLabel"]` below are confirmed, actively-used
       selectors (independently verified via Streamlit community CSS
       examples). Deeper container-internals selectors like
       "stVerticalBlockBorderWrapper" are NOT part of Streamlit's documented
       public API and were NOT independently verifiable in this environment
       (no browser available to inspect live rendered HTML) — so, to avoid
       shipping CSS on an unverified guess, this section is deliberately
       limited to selectors that are either (a) independently confirmed, or
       (b) already proven working in THIS app's existing CSS above
       (.block-container, headings). If you want the deeper per-element gap
       tightened further, that needs a browser DevTools inspection first. */
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3 {
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
    }

    /* Trim the overall top padding of the page — the default reserves
       noticeably more headroom above the first element than a page with a
       hero banner (which already supplies its own visual top anchor) needs.

       ALSO: even with layout="wide" set in st.set_page_config, Streamlit's
       .block-container still caps out at a fixed max-width and centers
       itself in the viewport — on a browser window wider than that cap
       (common on any modern monitor), this leaves a large dead strip of
       empty space on the right that has nothing to do with content amount,
       just the container's own width ceiling. This is a well-documented
       Streamlit behavior (see Streamlit community thread "Any way to expand
       the display width?" and "Change main app width") with a standard,
       widely-used fix: override max-width to actually use the full
       available width. */
    .main .block-container {
        padding-top: 1.6rem;
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* st.metric: denser by default (small label sitting far above a large
       number, with generous padding on all sides). Bringing the label closer
       to its value reads as one connected unit instead of two floating
       pieces of text with dead space between them. */
    div[data-testid="stMetric"] {
        background: #6C63FF08;
        border-radius: 10px;
        padding: 0.6rem 0.9rem 0.5rem;
    }
    div[data-testid="stMetricLabel"] {
        margin-bottom: 0.1rem;
    }

    /* Shown only on small screens, where Streamlit collapses the sidebar behind
       the ☰ menu. Since ALL navigation now lives in the sidebar, mobile users
       need to be told where the lesson list went. */
    .mobile-hint { display: none; }

    /* Mobile */
    @media (max-width: 768px) {
        .page-title h1 { font-size: 1.5rem; }
        .page-title p { font-size: 0.9rem; }
        .module-head h2 { font-size: 1.35rem; }
        .mobile-hint {
            display: block;
            background: #2563eb18;
            border: 1px solid #2563eb44;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        /* Long code lines should scroll, never squash the page wider than the screen */
        .main .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    }

    footer, #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HIDDEN ADMIN VIEW — usage analytics dashboard
#
# Only reachable via a secret URL query param (?admin=1) AND a password
# check against st.secrets — never appears in the sidebar nav, so ordinary
# learners never see it or know it exists. If it's not configured (no
# secrets set), this whole block is skipped and the app behaves exactly as
# before. This MUST run before the sidebar/nav below so it can render its
# own standalone page and st.stop() without touching normal app state.
# ---------------------------------------------------------------------------
if st.query_params.get("admin") == "1":
    st.title("NLPPlayground — Usage Dashboard")
    st.caption("Hidden admin view. Not linked from anywhere in the app's navigation.")

    _admin_secret = _safe_secret("admin_password")
    if not _admin_secret:
        st.error(
            "Admin dashboard isn't configured yet. Add an `admin_password` value "
            "to this app's Streamlit secrets to enable it."
        )
        st.stop()

    _entered = st.text_input("Admin password", type="password", key="_admin_pw")
    if _entered != _admin_secret:
        if _entered:
            st.error("Incorrect password.")
        st.stop()

    st.success("Authenticated.")
    _events_df = load_analytics_events()

    if _events_df is None:
        st.warning(
            "Analytics isn't configured (or the Google Sheet isn't reachable). "
            "See ANALYTICS-SETUP.md for setup steps. No events have been logged."
        )
        st.stop()

    if _events_df.empty:
        st.info("Analytics is configured, but no events have been logged yet.")
        st.stop()

    # --- Derive per-session summaries (session_start -> most recent event) ---
    _events_df["timestamp_utc"] = pd.to_datetime(_events_df["timestamp_utc"], errors="coerce")
    _events_df = _events_df.dropna(subset=["timestamp_utc"])

    _sessions = (
        _events_df.groupby("session_id")["timestamp_utc"]
        .agg(first_seen="min", last_seen="max")
        .reset_index()
    )
    _sessions["est_minutes"] = (
        (_sessions["last_seen"] - _sessions["first_seen"]).dt.total_seconds() / 60
    ).round(1)

    total_sessions = len(_sessions)
    total_page_views = int((_events_df["event_type"] == "page_view").sum())
    avg_minutes = round(_sessions["est_minutes"].mean(), 1) if total_sessions else 0.0
    median_minutes = round(_sessions["est_minutes"].median(), 1) if total_sessions else 0.0

    st.markdown("### Headline numbers")
    st.caption(
        "\"Time spent\" is an ESTIMATE — the gap between a session's first and "
        "most recent logged interaction. Streamlit has no way to detect a tab "
        "sitting open with no clicks, so quiet reading time between "
        "interactions is not visible here. Treat these as directional, not exact."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total sessions", total_sessions)
    m2.metric("Total page views", total_page_views)
    m3.metric("Avg. est. time/session", f"{avg_minutes} min")
    m4.metric("Median est. time/session", f"{median_minutes} min")

    st.markdown("---")
    st.markdown("### Sessions by day")
    _by_day = _events_df[_events_df["event_type"] == "session_start"]["date"].value_counts().sort_index()
    if not _by_day.empty:
        st.bar_chart(_by_day)
    else:
        st.caption("No session_start events yet.")

    st.markdown("### Page views by hour of day (UTC)")
    _by_hour = _events_df[_events_df["event_type"] == "page_view"]["hour"].value_counts().sort_index()
    if not _by_hour.empty:
        st.bar_chart(_by_hour)
    else:
        st.caption("No page_view events yet.")

    st.markdown("### Most-viewed pages/lessons")
    _top_pages = (
        _events_df[_events_df["event_type"] == "page_view"]["detail"]
        .value_counts()
        .head(20)
    )
    if not _top_pages.empty:
        st.dataframe(_top_pages.rename("views"), use_container_width=True)
    else:
        st.caption("No page_view events yet.")

    with st.expander("Raw event log (most recent 500 rows)"):
        st.dataframe(
            _events_df.sort_values("timestamp_utc", ascending=False).head(500),
            use_container_width=True,
            hide_index=True,
        )

    st.stop()

# ---------------------------------------------------------------------------
# Sample dataset (small, bundled inline so there is zero setup friction)
# ---------------------------------------------------------------------------
# NOTE ON CONTENT LICENSING: these sample reviews are original text written for
# this app, deliberately containing NO real people, companies, products or works.
# Earlier drafts named real film directors; that was removed because inventing
# opinions and attributing them near real names is an avoidable risk with no
# teaching benefit. See CONTENT-LICENSES.md for the full content provenance.
SAMPLE_REVIEWS = pd.DataFrame({
    "review": [
        "This film was absolutely fantastic! The acting was superb and the story kept me hooked till the end.",
        "Waste of time. The plot made no sense and the pacing was painfully slow.",
        "It was okay, nothing special but not terrible either. Decent way to spend an evening.",
        "One of the best films I've seen this year. Brilliant direction and outstanding cinematography.",
        "Terrible acting, weak script, and the ending felt completely rushed. Very disappointed.",
        "A heartwarming story with great performances from the entire cast. Highly recommended!",
        "The special effects were amazing but the storyline was confusing and hard to follow.",
        "I loved every minute of it. The director really outdid herself with this masterpiece.",
        "Boring from start to finish. I almost fell asleep twice.",
        "Solid film overall. Great soundtrack, good performances, worth watching once.",
    ],
    # Hand-labeled by reading each review — used for the Classification Quick Tool.
    # Kept honest rather than optimized: review 3 and 7 are genuinely mixed/neutral,
    # not forced into positive/negative to make the demo look cleaner.
    "sentiment": [
        "positive", "negative", "neutral", "positive", "negative",
        "positive", "neutral", "positive", "negative", "positive",
    ],
})

# Public-domain corpora available through NLTK, offered as sample text.
# Every entry here is PUBLIC DOMAIN per NLTK's own DATASET-LICENSES.md, chosen
# specifically so the app carries no content-licensing restrictions.
PUBLIC_DOMAIN_CORPORA = {
    "Jane Austen — Emma (1816, Project Gutenberg)": ("gutenberg", "austen-emma.txt"),
    "Lewis Carroll — Alice in Wonderland (1865, Project Gutenberg)": ("gutenberg", "carroll-alice.txt"),
    "Shakespeare — Hamlet (Project Gutenberg)": ("gutenberg", "shakespeare-hamlet.txt"),
    "US Inaugural Address — 1861 (Lincoln)": ("inaugural", "1861-Lincoln.txt"),
    "US Inaugural Address — 1933 (Roosevelt)": ("inaugural", "1933-Roosevelt.txt"),
    "Universal Declaration of Human Rights (English)": ("udhr", "English-Latin1"),
}


@st.cache_resource(show_spinner="Fetching public-domain text (first use only)...")
def load_public_domain_text(corpus_name: str, file_id: str, max_chars: int = 20000) -> str:
    """Load a PUBLIC DOMAIN text from NLTK's corpora.

    Downloaded lazily on first request rather than at boot — same lesson we
    learned from the spaCy loading bug: never make every visitor wait for a
    resource only some of them will use.
    """
    nltk.download(corpus_name, quiet=True)
    if corpus_name == "gutenberg":
        from nltk.corpus import gutenberg as _c
    elif corpus_name == "inaugural":
        from nltk.corpus import inaugural as _c
    else:
        from nltk.corpus import udhr as _c
    return _c.raw(file_id)[:max_chars]


# ---------------------------------------------------------------------------
# MULTI-DOCUMENT CORPORA — for Classification and Clustering, which need many
# documents to be meaningful (unlike the single-text tools above).
#
# We deliberately do NOT use NLTK's movie_reviews or brown corpora here even
# though they're the standard teaching datasets for this exact purpose —
# checked their actual licenses: movie_reviews is "distributed with permission
# from the authors" (Pang & Lee) and brown is explicitly non-commercial-only.
# Both are fine for NLTK's own redistribution but carry the same restricted
# category already flagged for maxent_ne_chunker in CONTENT-LICENSES.md — this
# app's standing rule is public-domain-with-no-limitations, so we use genuinely
# public-domain multi-document sources instead: US inaugural addresses (US
# government works) and Gutenberg books.
#
# Bonus: inaugural addresses carry REAL, non-fabricated metadata for
# classification — the year and president are literally in the filename
# (e.g. "1861-Lincoln.txt") — so "predict the era from the speech" is a
# genuine supervised task with zero invented labels.
# ---------------------------------------------------------------------------
MULTI_DOC_MAX_CHARS = 6000   # per document — long enough to be real text, short enough to stay fast
MULTI_DOC_MAX_UPLOAD_FILES = 8
MULTI_DOC_MAX_UPLOAD_CHARS = 20000  # per uploaded file


def _era_from_year(year: int) -> str:
    """Mechanical bucketing from a real year — not an invented/subjective label."""
    if year < 1900:
        return "19th century (pre-1900)"
    elif year < 2000:
        return "20th century (1900-1999)"
    else:
        return "21st century (2000+)"


@st.cache_resource(show_spinner="Loading all 60 US inaugural addresses (first use only)...")
def load_inaugural_corpus(max_chars: int = MULTI_DOC_MAX_CHARS):
    """Every US inaugural address, 1789-2025. Returns list of dicts with the
    year/president pulled straight from NLTK's own filenames — genuine
    metadata, not something we assigned by reading the text."""
    nltk.download("inaugural", quiet=True)
    from nltk.corpus import inaugural
    docs = []
    for fid in inaugural.fileids():
        year = int(fid[:4])
        president = fid[5:-4]
        docs.append({
            "label": f"{year} {president}",
            "year": year,
            "era": _era_from_year(year),
            "text": inaugural.raw(fid)[:max_chars],
        })
    return docs


@st.cache_resource(show_spinner="Loading 18 public-domain books (first use only)...")
def load_gutenberg_corpus(max_chars: int = MULTI_DOC_MAX_CHARS):
    """Opening portion of each of NLTK's 18 bundled Gutenberg texts. Capped per
    book — this is a stylistic-similarity demo (whose writing does this
    resemble?), not full-book analysis, and we say so in the UI."""
    nltk.download("gutenberg", quiet=True)
    from nltk.corpus import gutenberg
    docs = []
    for fid in gutenberg.fileids():
        docs.append({
            "label": fid.replace(".txt", ""),
            "author": fid.split("-")[0],
            "text": gutenberg.raw(fid)[:max_chars],
        })
    return docs


AUTHOR_PASSAGE_SENTENCES = 10  # sentences per passage — long enough to carry style, short enough for many examples


@st.cache_resource(show_spinner="Splitting 9 books into passages (first use only)...")
def load_author_passages_corpus(sentences_per_passage: int = AUTHOR_PASSAGE_SENTENCES):
    """Splits the 3 NLTK Gutenberg authors who have 3+ books each (Austen,
    Chesterton, Shakespeare — 9 books total) into ~10-sentence passages.

    WHY: the 18-book Gutenberg corpus has only 1-3 books per author, which is
    too few *rows* for a real train/test split even though each book is huge.
    Splitting into passages turns 9 books into several hundred genuine,
    non-overlapping training examples, while keeping the label 100%
    mechanical — "which book did NLTK file this text under" — never a guess
    or an invented category. This is a stylometry task (whose writing style
    does this passage resemble?), not a claim about content or quality.
    """
    nltk.download("gutenberg", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.corpus import gutenberg

    eligible_authors = {"austen", "chesterton", "shakespeare"}
    docs = []
    for fid in gutenberg.fileids():
        author = fid.split("-")[0]
        if author not in eligible_authors:
            continue
        sents = sent_tokenize(gutenberg.raw(fid))
        for i in range(0, len(sents), sentences_per_passage):
            chunk = sents[i:i + sentences_per_passage]
            if len(chunk) < sentences_per_passage:
                break  # drop a short trailing partial passage
            docs.append({
                "label": f"{fid.replace('.txt', '')} #{i // sentences_per_passage}",
                "author": author,
                "book": fid.replace(".txt", ""),
                "text": " ".join(chunk),
            })
    return docs


def render_multi_doc_picker(key_prefix: str):
    """Shared UI for picking a multi-document corpus: built-in public-domain
    sets, or the user's own uploaded files. Returns a list of dicts each with
    at least 'label' and 'text' keys, or an empty list if nothing is ready yet.

    Used by both the Classification & Clustering guided lessons and the
    upgraded Quick Tools versions, so the two stay consistent.
    """
    source = st.radio(
        "Multi-document corpus",
        [
            "US Inaugural Addresses (60 speeches, 1789–2025, public domain)",
            "Gutenberg books (18 novels/texts, opening portion, public domain)",
            "Upload your own files",
        ],
        key=f"{key_prefix}_source",
    )

    if source.startswith("US Inaugural"):
        return load_inaugural_corpus()

    if source.startswith("Gutenberg"):
        return load_gutenberg_corpus()

    # --- Upload path ---
    uploaded = st.file_uploader(
        f"Upload up to {MULTI_DOC_MAX_UPLOAD_FILES} files (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key=f"{key_prefix}_upload",
    )
    if not uploaded:
        st.info("Upload at least 2 files to enable classification/clustering on your own text.")
        return []
    if len(uploaded) > MULTI_DOC_MAX_UPLOAD_FILES:
        st.warning(
            f"Only the first {MULTI_DOC_MAX_UPLOAD_FILES} files will be used "
            f"(you uploaded {len(uploaded)}) — this keeps the demo fast on a "
            "shared, memory-limited server."
        )
    docs = []
    for f in uploaded[:MULTI_DOC_MAX_UPLOAD_FILES]:
        try:
            if f.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = f.read().decode("utf-8", errors="replace")
            docs.append({"label": f.name, "text": text[:MULTI_DOC_MAX_UPLOAD_CHARS]})
        except Exception as e:
            st.error(f"Couldn't read {f.name}: {e}")
    return docs


DOC_BOX_COUNT = 10


def render_document_boxes(key_prefix: str):
    """10 editable document boxes for TF/TF-IDF: each one is either pasted text
    or an uploaded PDF, pre-filled with our 10 sample reviews as a replaceable
    starting point. Nothing is (re-)analyzed until the Analyze button is
    pressed — with 10 boxes, recomputing on every keystroke/blur would make
    the page jump around constantly.

    Returns the list of (label, text) pairs from the LAST time Analyze was
    pressed (persisted across reruns), or None if it hasn't been pressed yet
    this session. Shared key_prefix across lessons means whatever's filled in
    on one lesson is still there when you move to the next.
    """
    st.caption(
        f"Fill in up to {DOC_BOX_COUNT} documents below — type your own text or upload a "
        "PDF per box. Pre-filled with our sample reviews as a starting point; replace "
        "any or all of them. Nothing is analyzed until you press **Analyze** at the "
        "bottom — editing 10 boxes shouldn't recompute anything on every keystroke."
    )

    defaults = SAMPLE_REVIEWS["review"].tolist()
    box_texts = []
    cols = st.columns(2)
    for i in range(DOC_BOX_COUNT):
        with cols[i % 2]:
            with st.expander(f"Document {i + 1}", expanded=(i < 2)):
                input_mode = st.radio(
                    "Input type", ["Type text", "Upload PDF"],
                    horizontal=True, key=f"{key_prefix}_mode_{i}",
                    label_visibility="collapsed",
                )
                if input_mode == "Upload PDF":
                    up = st.file_uploader(
                        "PDF", type=["pdf"], key=f"{key_prefix}_pdf_{i}",
                        label_visibility="collapsed",
                    )
                    box_text = ""
                    if up is not None:
                        try:
                            from pypdf import PdfReader
                            reader = PdfReader(up)
                            box_text = "\n".join(
                                page.extract_text() or "" for page in reader.pages
                            )[:MULTI_DOC_MAX_UPLOAD_CHARS]
                        except Exception as e:
                            st.error(f"Couldn't read this PDF: {e}")
                else:
                    default_text = defaults[i] if i < len(defaults) else ""
                    box_text = st.text_area(
                        "Text", value=default_text, height=100,
                        key=f"{key_prefix}_text_{i}", label_visibility="collapsed",
                    )
                box_texts.append(box_text.strip())

    non_empty = sum(1 for t in box_texts if t)
    analyze = st.button(
        f"Analyze ({non_empty} of {DOC_BOX_COUNT} documents have text)",
        key=f"{key_prefix}_analyze", type="primary",
    )

    result_key = f"{key_prefix}_result"
    if analyze:
        st.session_state[result_key] = [
            (f"Document {i + 1}", t) for i, t in enumerate(box_texts) if t
        ]
    return st.session_state.get(result_key)


# Shared example sentences used across BOTH the NLTK and spaCy guided tracks,
# so lessons that cover the same task (tokenization, stopwords, POS, NER) run
# on identical input — differences you see are real library differences, not
# different text. Chosen deliberately: one has a contraction + possessives
# (good for tokenization/stopwords), the other is entity-rich (good for
# POS tagging / NER / dependency parsing).
#
# CONTENT NOTE: these sentences are original text using deliberately fictional
# names ("Example Corp", "Jane Doe") following the RFC 2606 convention of
# reserved example identifiers. An earlier draft used an invented company name
# that turned out to belong to several real businesses — attributing invented
# executives and financial figures to a real company name is a genuine risk,
# so all placeholder names here are unambiguously non-real.
SHARED_TEXT_TOKENS_DEFAULT = "I don't think Example Corp's Q3 numbers were as strong as everyone's expecting."
SHARED_TEXT_ENTITIES_DEFAULT = (
    "Dr. Jane Doe, the CFO of Example Corp, announced on Monday "
    "that the company will invest $2.5 million in its new Bengaluru office by March 2027."
)


def render_shared_text_editor():
    """Editable, scrollable, persisted textboxes for the example text used by
    every lesson in BOTH the NLTK and spaCy tracks. Returns the current
    (possibly edited) (tokens_text, entities_text) pair.

    Called at the top of each track's render block and reassigned onto the
    SAME variable names (SHARED_TEXT_TOKENS / SHARED_TEXT_ENTITIES) that all
    ~35 existing lesson call-sites already reference below — this is a flat,
    top-level script (not wrapped in a function), so that reassignment flows
    through automatically without touching any of those call sites.

    Both tracks read from the SAME session-state keys, preserving the
    deliberate invariant that NLTK vs spaCy differences are real library
    differences, not an artifact of different input text.
    """
    # Pending-value pattern (same fix used for lesson prev/next): a widget's
    # own key can't be reassigned after the widget is instantiated, so a Reset
    # button must stash its target value and apply it BEFORE the text_area
    # below is created on the next run.
    for widget_key, default in (
        ("shared_text_tokens_box", SHARED_TEXT_TOKENS_DEFAULT),
        ("shared_text_entities_box", SHARED_TEXT_ENTITIES_DEFAULT),
    ):
        pending = st.session_state.pop(f"pending_{widget_key}", None)
        if pending is not None:
            st.session_state[widget_key] = pending
        elif widget_key not in st.session_state:
            st.session_state[widget_key] = default

    with st.expander("Edit the shared example text (used by every lesson below)"):
        st.caption(
            "Both the NLTK and spaCy tracks use the **same** text for every lesson, so any "
            "differences you see are genuine library differences — not caused by different "
            "input. Edit here and it updates every lesson in both tracks; reset any time."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tokens example** — Tokenization, Stopwords, Stemming/Lemmatization")
            st.text_area(
                "Tokens example", height=120, key="shared_text_tokens_box",
                label_visibility="collapsed",
            )
            if st.button("Reset to default", key="reset_tokens_text"):
                st.session_state["pending_shared_text_tokens_box"] = SHARED_TEXT_TOKENS_DEFAULT
                st.rerun()
        with c2:
            st.markdown("**Entities example** — POS, NER, Sentiment, Keyword Extraction, Word Cloud, Dependency Parsing")
            st.text_area(
                "Entities example", height=120, key="shared_text_entities_box",
                label_visibility="collapsed",
            )
            if st.button("Reset to default", key="reset_entities_text"):
                st.session_state["pending_shared_text_entities_box"] = SHARED_TEXT_ENTITIES_DEFAULT
                st.rerun()

    tokens_val = st.session_state["shared_text_tokens_box"].strip() or SHARED_TEXT_TOKENS_DEFAULT
    entities_val = st.session_state["shared_text_entities_box"].strip() or SHARED_TEXT_ENTITIES_DEFAULT
    return tokens_val, entities_val

# ---------------------------------------------------------------------------
# QUICK TOOLS MAP — single source of truth, referenced by both the tool
# selectbox and the Start Here overview's "Interactive tools" count. Keeping
# these in one place after a hardcoded count (4) silently went stale the
# moment Word Cloud, Classification, and Clustering were added — three new
# tools shipped and the landing page kept advertising the old number.
# ---------------------------------------------------------------------------
QUICK_TOOLS = [
    "Preprocessing Pipeline",
    "Sentiment Analysis",
    "Keyword Extraction",
    "Named Entity Recognition",
    "Word Cloud",
    "Classification",
    "Clustering",
]

# ---------------------------------------------------------------------------
# CURRICULUM MAP — single source of truth for navigation.
#
# Lessons used to live inside st.tabs(). That was replaced because tab strips
# of 7-8 long labels overflow horizontally on a phone (hiding later lessons
# entirely), because tabs imply parallel peers when these lessons are strictly
# sequential, and because Streamlit renders EVERY tab's body on every run —
# which is what caused the StreamlitDuplicateElementId bug earlier.
# ---------------------------------------------------------------------------
PY_TRACKS = {
    "Fundamentals": {
        "blurb": "Zero assumptions. Everything you need before touching NLP.",
        "level": "Beginner",
        "lessons": [
            "0 What is Python?", "1 Variables & Types", "2 Strings & Text",
            "3 Lists & Loops", "4 Conditionals", "5 Functions",
            "6 Dictionaries", "7 DataFrames",
        ],
    },
    "Working with Text Data": {
        "blurb": "Real text lives in files, in odd encodings, inside PDFs. Bridge module.",
        "level": "Intermediate",
        "lessons": [
            "0 Reading & Writing Files", "1 Encodings", "2 Regular Expressions",
            "3 Cleaning Text", "4 Text in pandas", "5 Reading PDFs",
            "6 Errors & Debugging",
        ],
    },
    "NLTK": {
        "blurb": "The classic, rule-based toolkit. See how NLP works under the hood.",
        "level": "NLP",
        "lessons": [
            "0 Tokenization", "1 Stopwords", "2 Stemming",
            "3 POS Tagging", "4 NER", "5 Sentiment (VADER)",
            "6 Keyword Extraction", "7 Word Cloud",
            "8 Frequency Distributions",
        ],
    },
    "spaCy": {
        "blurb": "The modern, model-based library. Same tasks, different tradeoffs.",
        "level": "NLP",
        "lessons": [
            "0 Tokenization", "1 Stopwords", "2 Lemmatization",
            "3 POS Tagging", "4 NER", "5 Dependency Parsing",
            "6 Sentiment", "7 Keyword Extraction", "8 Word Cloud",
        ],
    },
    "Classification & Clustering": {
        "blurb": "From counting words to machine learning: TF → TF-IDF → predicting labels → finding groups.",
        "level": "Machine Learning",
        "lessons": [
            "0 Term Frequency (TF)", "1 TF-IDF (needs a corpus)",
            "2 Naive Bayes Classification", "3 K-Means Clustering",
        ],
    },
}

R_LESSONS = [
    "0 What is R?", "1 Variables & Types", "2 Strings & Text",
    "3 Vectors & Loops", "4 Conditionals", "5 Functions",
    "6 Named Lists", "7 Data Frames",
]

TOTAL_PY_LESSONS = sum(len(t["lessons"]) for t in PY_TRACKS.values())


def render_overview():
    """Landing screen: show the whole learning path before dropping anyone into it.

    Previously the app opened directly inside Lesson 0 with no sense of what
    else existed or how the modules related to each other.
    """
    st.markdown(
        "<div class='card'><b>New here? This is the whole path.</b> Work top to bottom, "
        "or jump straight to whatever you need — every module is reachable from the "
        "sidebar at any time.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Python lessons", TOTAL_PY_LESSONS)
    c2.metric("Modules", len(PY_TRACKS))
    c3.metric("Interactive tools", len(QUICK_TOOLS))

    st.markdown("### The Python path")
    steps = [
        ("", "Fundamentals", "Beginner", PY_TRACKS["Fundamentals"],
         "Never written Python? Start here."),
        ("", "Working with Text Data", "Intermediate", PY_TRACKS["Working with Text Data"],
         "Comfortable with basics but never opened a file or hit an encoding error?"),
        ("", "NLTK", "NLP", PY_TRACKS["NLTK"],
         "Ready for NLP. The classic, rule-based toolkit."),
        ("", "spaCy", "NLP", PY_TRACKS["spaCy"],
         "The same tasks in a modern, model-based library — built for comparison."),
        ("", "Classification & Clustering", "Machine Learning", PY_TRACKS["Classification & Clustering"],
         "TF → TF-IDF → predicting labels → finding groups, on 60 real inaugural addresses."),
    ]
    for icon, name, level, meta, who in steps:
        track_key = f"{icon} {name}"
        with st.container(border=True):
            a, b = st.columns([3, 1])
            with a:
                st.markdown(f"**{icon} {name}** &nbsp; <span class='badge badge-blue'>{level}</span>",
                            unsafe_allow_html=True)
                st.caption(who)
                st.caption("· ".join(l.split(" ", 1)[1] for l in meta["lessons"]))
            with b:
                st.metric("Lessons", len(meta["lessons"]))
            if st.button(f"Start {name} →", key=f"overview_start_{track_key}", use_container_width=True):
                st.session_state["pending_nav_language"] = "Python"
                st.session_state["pending_nav_mode"] = "Guided Learning"
                st.session_state["pending_nav_track"] = track_key
                st.session_state[f"pending_nav_lesson_{track_key}"] = meta["lessons"][0]
                st.rerun()

    st.markdown("### Or skip the lessons")
    t1, t2 = st.columns(2)
    with t1:
        with st.container(border=True):
            st.markdown("**Quick Tools**")
            st.caption(
                "Load your own text — paste, upload a .txt/.csv/.pdf, or pick a "
                "public-domain classic — then run sentiment, keywords, entity "
                "recognition, word clouds, classification, or clustering on it. "
                "NLTK and spaCy results shown side by side."
            )
            if st.button("Open Quick Tools →", key="overview_start_quicktools", use_container_width=True):
                st.session_state["pending_nav_mode"] = "Quick Tools"
                st.rerun()
    with t2:
        with st.container(border=True):
            st.markdown("**R reference track**")
            st.caption(
                "Real, correct R code covering the same Fundamentals ground. Output is "
                "precomputed rather than live — this app runs on Python. Useful as a "
                "reference; Python is the interactive path."
            )
            if st.button("Open R track →", key="overview_start_r", use_container_width=True):
                st.session_state["pending_nav_language"] = "R"
                st.session_state["pending_nav_mode"] = "Guided Learning"
                st.rerun()

    st.info(
        "**Not sure where to start?** If you've never written code, go to "
        "**Guided Learning → Fundamentals**. If you can already write a loop and "
        "a function, start at **Working with Text Data**. If you just want to see NLP "
        "do something, go straight to **Quick Tools**."
    )

    with st.expander("Where everything fits in the NLP pipeline"):
        st.caption(
            "A real NLP/ML workflow has a shape: clean the text, turn it into numbers, "
            "then either predict a label or find groups. Every lesson and tool below maps "
            "onto one stage of that shape."
        )
        taxonomy = pd.DataFrame([
            ("Fundamentals", "Foundations", "Python itself — no NLP yet"),
            ("Working with Text Data", "Foundations", "Files, encodings, regex — getting real text INTO Python"),
            ("NLTK / spaCy — Tokenization, Stopwords, Stemming/Lemmatization", "Preprocessing", "Cleaning and normalizing text"),
            ("NLTK / spaCy — POS Tagging, NER, Dependency Parsing", "Feature Extraction", "Pulling structured information out of text"),
            ("NLTK — Frequency Distributions", "Descriptive Statistics", "FreqDist, most_common(), hapaxes — the numeric summary of a document's vocabulary"),
            ("NLTK — Sentiment (VADER)", "Classification-adjacent", "Rule-based label assignment, no training involved"),
            ("Quick Tools — Preprocessing Pipeline", "Preprocessing", "The cleaning stage above, made visible and toggleable"),
            ("Quick Tools — Keyword Extraction", "Feature Extraction", "Term Frequency (TF), informally"),
            ("Quick Tools — Word Cloud", "Visualization", "A picture of Feature Extraction's output"),
            ("Classification & Clustering — TF, TF-IDF", "Feature Extraction", "Turning text into numbers properly, with a real multi-document corpus"),
            ("Classification & Clustering — Naive Bayes", "Classification", "Predicting a label (era) from learned examples"),
            ("Classification & Clustering — K-Means", "Clustering", "Finding groups with no labels at all"),
        ], columns=["Where", "Category", "What it actually does"])
        st.dataframe(taxonomy, use_container_width=True, hide_index=True)


def lesson_footer(track_key: str, idx: int, lessons: list, state_key: str):
    """Previous / Next navigation plus honest position indicator.

    Replaces the old 'mark lesson complete' buttons, whose progress bar measured
    button clicks rather than actual position.
    """
    st.markdown("<div class='lib-section'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])

    # NOTE: we must NOT write st.session_state[state_key] here — Streamlit raises
    # StreamlitAPIException if you modify a widget's key after that widget has
    # been instantiated, and the lesson radio in the sidebar already ran this
    # pass. Instead we stash a *pending* value which the sidebar applies at the
    # start of the next run, before the widget is created.
    with c1:
        if idx > 0 and st.button("← Previous", key=f"prev_{state_key}", use_container_width=True):
            st.session_state[f"pending_{state_key}"] = lessons[idx - 1]
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center; opacity:0.7; padding-top:0.5rem'>"
            f"Lesson <b>{idx + 1}</b> of <b>{len(lessons)}</b> · {track_key}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if idx < len(lessons) - 1 and st.button("Next →", key=f"next_{state_key}", use_container_width=True):
            st.session_state[f"pending_{state_key}"] = lessons[idx + 1]
            st.rerun()


# ---------------------------------------------------------------------------
# Helpers: reusable code + output panels
# ---------------------------------------------------------------------------
def code_and_output(code: str, render_output, key: str):
    """Shows LIVE Python code on the left, real computed output on the right.

    `key` must be unique per call site — Streamlit runs all st.tabs()
    content in the same script run, so identical widgets need distinct
    keys or they collide (StreamlitDuplicateElementId).
    """
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Python code")
        st.code(code, language="python")
        st.download_button(
            "Download this code",
            data=code,
            file_name=f"{key}.py",
            mime="text/x-python",
            use_container_width=True,
            key=f"dl_{key}",
        )
    with c2:
        st.caption("Output")
        render_output()


def code_and_output_r(code: str, render_output, key: str):
    """Shows R code on the left and a precomputed example output on the right.

    This app runs on Python (Streamlit), so R code here is NOT executed live —
    the output shown is a worked, hand-verified example, clearly labeled as such.
    """
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("R code")
        st.code(code, language="r")
        st.download_button(
            "Download this code",
            data=code,
            file_name=f"{key}.R",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_{key}",
        )
    with c2:
        st.caption("Example output (precomputed)")
        st.caption("_This output isn't live-executed — run the code yourself in RStudio or Posit Cloud to experiment._")
        render_output()


def render_r_practice_box(key: str, placeholder: str):
    """An editable scratchpad for practicing R syntax, honestly labeled.

    This app has no R runtime — there's no way to actually execute whatever
    gets typed here. A box that looked responsive without being responsive
    would be dishonest UI, so this is explicit: it's for practicing syntax by
    typing it out, not for seeing your own input actually run. The output
    shown elsewhere on the lesson is always the fixed, verified example.
    """
    st.text_area(
        "Practice writing R code here",
        value=placeholder, height=100, key=key,
    )
    st.caption(
        "This box doesn't execute — this app runs on Python, so there's no R "
        "interpreter here to run what you type. Use it to practice R syntax by writing "
        "it out; paste into RStudio, Posit Cloud, or an online R environment to see it "
        "actually run. The output above/below stays the verified example regardless."
    )


def render_entities(doc, key: str = ""):
    """Visualise named entities inline using spaCy's built-in displaCy renderer.

    Deliberately uses displaCy rather than adding a charting/visualisation
    dependency: it ships inside spaCy, so this adds ZERO new packages to
    requirements.txt and carries none of the deploy risk that the
    en_core_web_sm runtime-install bug taught us to avoid.
    """
    if not doc.ents:
        # Rendering an entity view with no entities produces an empty box and a
        # spaCy W006 warning. Say something useful instead.
        st.info(
            "No entities found in this text, so there's nothing to highlight. "
            "Entity recognition needs names of people, organisations, places, "
            "dates or amounts — try text that contains some."
        )
        return False
    try:
        from spacy import displacy
        html = displacy.render(doc, style="ent", page=False)
        # displaCy emits a fixed line-height that looks cramped in Streamlit.
        html = html.replace("line-height: 2.5", "line-height: 2.8")
        st.markdown(html, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.caption(f"(Entity highlighting unavailable: {e})")
        return False


def render_dependency_tree(doc, key: str = ""):
    """Visualise the dependency parse of the FIRST sentence only.

    Restricted to one sentence on purpose — displaCy's SVG grows very wide with
    long input and becomes unreadable rather than illustrative.
    """
    try:
        from spacy import displacy
        import streamlit.components.v1 as components
        sents = list(doc.sents)
        if not sents:
            st.info("No sentence detected to diagram.")
            return False
        svg = displacy.render(
            sents[0], style="dep", page=False,
            options={"compact": True, "distance": 90, "bg": "transparent"},
        )
        components.html(f"<div style='overflow-x:auto'>{svg}</div>", height=380, scrolling=True)
        if len(sents) > 1:
            st.caption(f"Showing sentence 1 of {len(sents)} — long diagrams become unreadable.")
        return True
    except Exception as e:
        st.caption(f"(Dependency diagram unavailable: {e})")
        return False


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="page-title">
        <h1>NLPPlayground</h1>
        <p>Learn Python, R &amp; NLP by writing, running, and seeing real code — no setup required.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Only rendered on narrow screens (see .mobile-hint in CSS above).
st.markdown(
    "<div class='mobile-hint'><b>On a phone?</b> Tap the <b>☰</b> menu at the top-left "
    "to choose a module and lesson. You can also move through lessons with the "
    "<b>Previous / Next</b> buttons at the bottom of each page — no menu needed.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR — owns ALL navigation.
# Previously `mode` lived here while the track selector sat in the main body,
# so two levels of the same hierarchy appeared in two different places.
# ---------------------------------------------------------------------------
with st.sidebar:
    # Apply any pending jumps requested by buttons elsewhere on the page (the
    # Start Here cards, Previous/Next, etc.) BEFORE the widgets below are
    # instantiated — Streamlit forbids writing to a widget's key afterwards.
    for _key in ("nav_language", "nav_mode", "nav_track"):
        _pending = st.session_state.pop(f"pending_{_key}", None)
        if _pending is not None:
            st.session_state[_key] = _pending

    st.markdown("### Language")
    language = st.radio(
        "Language track",
        ["Python", "R"],
        horizontal=True,
        label_visibility="collapsed",
        key="nav_language",
    )
    if language == "R":
        st.caption(
            "**Reference track** — real R code, precomputed output (this app runs on "
            "Python). Fundamentals only."
        )

    st.markdown("### Where to")
    mode = st.radio(
        "Choose your path",
        ["Start Here", "Guided Learning", "Quick Tools", "About"],
        label_visibility="collapsed",
        key="nav_mode",
    )

    # --- Lesson navigator: only shown when it is actually relevant ---
    track = None
    lesson_idx = 0
    if mode == "Guided Learning":
        st.markdown("---")
        if language == "Python":
            st.markdown("### Module")
            track = st.radio(
                "Module",
                list(PY_TRACKS.keys()),
                label_visibility="collapsed",
                key="nav_track",
            )
            st.caption(PY_TRACKS[track]["blurb"])
            lessons = PY_TRACKS[track]["lessons"]
            state_key = f"nav_lesson_{track}"
        else:
            st.markdown("### Module")
            st.caption("R covers Fundamentals only.")
            track = "Fundamentals"
            lessons = R_LESSONS
            state_key = "nav_lesson_r"

        # Apply any pending jump requested by the Previous/Next buttons on the
        # previous run. This MUST happen before the radio below is instantiated.
        pending = st.session_state.pop(f"pending_{state_key}", None)
        if pending is not None and pending in lessons:
            st.session_state[state_key] = pending

        if state_key not in st.session_state:
            st.session_state[state_key] = lessons[0]
        # Guard against a stale value if the lesson list ever changes.
        if st.session_state[state_key] not in lessons:
            st.session_state[state_key] = lessons[0]

        st.markdown("### Lesson")
        chosen_lesson = st.radio(
            "Lesson",
            lessons,
            label_visibility="collapsed",
            key=state_key,
        )
        lesson_idx = lessons.index(chosen_lesson)

        st.progress((lesson_idx + 1) / len(lessons))
        st.caption(f"Lesson {lesson_idx + 1} of {len(lessons)} in this module")

    st.markdown("---")
    st.caption("Demo tool — please don't paste confidential or sensitive text.")
    st.caption("All content is original or public domain — see CONTENT-LICENSES.md")

    # Usage analytics — see the block near the top of this file for the full
    # design rationale. This call is a no-op if analytics isn't configured.
    track_session_and_page(
        language, mode, track,
        lessons[lesson_idx] if mode == "Guided Learning" else None,
    )

# ===========================================================================
# PYTHON TRACK
# ===========================================================================
if language == "Python":

    # -----------------------------------------------------------------------
    # GUIDED LEARNING — Module 1: Python Fundamentals
    # -----------------------------------------------------------------------
    if mode == "Start Here":
        render_overview()

    elif mode == "Guided Learning" and track == "Fundamentals":
        st.markdown(
            "<div class='module-head'><span class='badge badge-green'>Module 1 · Beginner</span>"
            "<h2>Python Fundamentals</h2>"
            "<p>Zero assumptions. By the end you'll have everything the NLP modules "
            "assume you already know.</p></div>",
            unsafe_allow_html=True,
        )


        # --- Lesson 0: What is Python? ---
        if lesson_idx == 0:
            st.subheader("What is Python?")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>3 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Think of **Python code like a recipe**: a list of exact steps the computer "
                "follows *in order, one at a time, without skipping anything*. Unlike a human "
                "reading a recipe, the computer won't guess what you meant — it does precisely "
                "what each line says. That's both the challenge and the power of programming."
            )
            st.info(
                "**Python is a programming language** — a way to give a computer precise, "
                "step-by-step instructions. Each line of code you see below is one instruction. "
                "Python is the most widely used language for AI, data analysis, and NLP because "
                "it's readable and has huge libraries built for exactly this kind of work."
            )

            with st.expander("Why Python for NLP specifically?"):
                st.markdown(
                    "Almost every major NLP tool — NLTK, spaCy, Hugging Face Transformers, "
                    "scikit-learn — is written in or built for Python. When a company builds "
                    "a chatbot, a spam filter, or a sentiment dashboard, the underlying code is "
                    "very likely Python. Learning Python here isn't just an exercise — it's the "
                    "same language used in real production NLP systems."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- Forgetting quotes around text: `print(Hello)` fails, `print(\"Hello\")` works.\n"
                    "- Mixing up `print(x)` (shows the value) with just writing `x` on its own line "
                    "(only works in some interactive tools, not in a saved script).\n"
                    "- Capitalization matters: `Print()` is not the same as `print()`."
                )

            greeting_name = st.text_input("What's your name?", "Learner")

            code = f'''# Setup: none needed — uses Python's built-in print(), no installs required.
name = "{greeting_name}"
print("Hello,", name, "- welcome to NLPPlayground!")'''

            def show_intro_output():
                st.write("**Output:**")
                st.code(f"Hello, {greeting_name} - welcome to NLPPlayground!", language="text")
                st.caption(
                    "`print()` displays text on screen. This is usually the first "
                    "thing anyone learns in any programming language."
                )

            code_and_output(code, show_intro_output, key="lesson0_intro")


        # --- Lesson 1: Variables & Data Types ---
        if lesson_idx == 1:
            st.subheader("Variables & Data Types")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Picture a **variable as a labeled box**. You put a value inside, write a name "
                "on the box, and later you can grab that value again just by using the name — "
                "no need to remember or retype the actual value."
            )
            st.info(
                "A **variable** is a labeled container for a value. Python has a few core "
                "**data types**: whole numbers (`int`), decimals (`float`), text (`str`), and "
                "true/false values (`bool`). Knowing the type of your data matters — you can't "
                "do math on text, for example."
            )

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Text data is almost always stored as the `str` type. When you later count "
                    "words, calculate a sentiment score (a `float`), or check whether a word "
                    "appears (`bool` — True/False), you're relying on Python knowing exactly "
                    "what type of data it's working with. Type mix-ups are one of the most "
                    "common bugs in real NLP code."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- `\"5\" + 3` fails — you can't add text and a number directly (`TypeError`).\n"
                    "- `5 + 3` gives `8`, but `\"5\" + \"3\"` gives `\"53\"` (string concatenation, not addition!).\n"
                    "- Use `type(x)` any time you're unsure what kind of data you're working with."
                )

            col_a, col_b = st.columns(2)
            with col_a:
                num_input = st.number_input("Enter a number:", value=7)
            with col_b:
                word_input = st.text_input("Enter a word:", "sentiment")

            code = f'''# Setup: none needed — uses Python's built-in types, no installs required.
age = {num_input}
word = "{word_input}"
is_learning = True

print(type(age))          # <class 'int'> or 'float'
print(type(word))         # <class 'str'>
print(type(is_learning))  # <class 'bool'>
print(age * 2)             # numbers support math
print(word * 2)            # strings support repetition, not math!'''

            def show_types_output():
                st.write(f"**Type of `{num_input}`:**", type(num_input).__name__)
                st.write(f"**Type of `\"{word_input}\"`:**", "str")
                st.write("**Type of `True`:**", "bool")
                st.write(f"**`{num_input} * 2` =**", num_input * 2)
                st.write(f"**`\"{word_input}\" * 2` =**", word_input * 2)

            code_and_output(code, show_types_output, key="lesson1_types")


        # --- Lesson 2: Strings & Text ---
        if lesson_idx == 2:
            st.subheader("Strings & Text")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "A **string** is just a sequence of characters — letters, numbers, spaces, "
                "punctuation — wrapped in quotes. Think of it as beads on a string: each "
                "character sits in an exact position, and Python lets you grab, count, or "
                "rearrange those beads however you need."
            )
            st.info("A **string** is text. Python gives you built-in tools to inspect and transform it — this is the foundation of every NLP task later.")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Virtually every NLP pipeline starts by cleaning and manipulating strings: "
                    "lowercasing, removing punctuation, splitting into words. Before any "
                    "sentiment analysis, keyword extraction, or entity recognition can happen, "
                    "your text first passes through string operations exactly like the ones here."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- Case sensitivity: `\"Apple\"` and `\"apple\"` are treated as *completely "
                    "different* strings unless you `.lower()` them first.\n"
                    "- Extra spaces matter: `\"hello \"` is not the same as `\"hello\"`.\n"
                    "- `.split()` with no argument splits on any whitespace — spaces, tabs, newlines."
                )

            user_string = st.text_area(
                "Type any sentence:", "NLPPlayground makes learning NLP fun",
                height=100, key="lesson2_strings_input",
            )

            code = f'''# Setup: none needed — uses Python's built-in string methods, no installs required.
s = "{user_string}"

print(len(s))          # number of characters
print(s.upper())       # all uppercase
print(s.split())       # split into a list of words
print(s[::-1])         # reverse the string'''

            def show_string_output():
                st.metric("Length", len(user_string))
                st.write("**Uppercase:**", user_string.upper())
                st.write("**Split into words:**", user_string.split())
                st.write("**Reversed:**", user_string[::-1])

            code_and_output(code, show_string_output, key="lesson2_strings")


        # --- Lesson 3: Lists & Loops ---
        if lesson_idx == 3:
            st.subheader("Lists & Loops")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "A **list** is an ordered collection — like a shopping list, but for any kind "
                "of value. A **loop** lets you say \"do this one thing for every item on the "
                "list\" instead of writing the same instruction over and over by hand."
            )
            st.info("A **list** stores multiple items. **Loops** let you process each item one at a time — the foundation of analyzing many texts.")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "A document is really just a list of sentences, and a sentence is a list "
                    "of words. Nearly every NLP task — counting word frequency, checking each "
                    "word against a stopword list, scoring each sentence for sentiment — is a "
                    "loop running over a list. Once you're comfortable with this pattern, most "
                    "of NLP is just applying it to text."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- Python lists are **zero-indexed** — the first item is `words[0]`, not `words[1]`.\n"
                    "- Forgetting the colon `:` at the end of a `for` line, or the indentation "
                    "underneath it — Python uses indentation to know what's *inside* the loop.\n"
                    "- `max(words, key=len)` finds the *longest* word, not the largest number — "
                    "`key=len` tells Python what to compare by."
                )

            items_raw = st.text_area(
                "Enter a few words, comma-separated:", "python, nlp, data, ai, learning",
                height=100, key="lesson3_lists_input",
            )
            items = [i.strip() for i in items_raw.split(",") if i.strip()]

            code = f'''# Setup: none needed — uses Python's built-in list/loop syntax, no installs required.
words = {items}

for word in words:
    print(word, "->", len(word), "characters")

longest = max(words, key=len)
print("Longest word:", longest)'''

            def show_list_output():
                for w in items:
                    st.write(f"`{w}` → {len(w)} characters")
                if items:
                    st.success(f"Longest word: **{max(items, key=len)}**")

            code_and_output(code, show_list_output, key="lesson3_lists")


        # --- Lesson 4: Conditionals ---
        if lesson_idx == 4:
            st.subheader("Conditionals (if / elif / else)")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Think of a **conditional as a fork in the road**: \"if this is true, go this "
                "way; otherwise, go that way instead.\" It's how code makes decisions rather "
                "than just running the same steps blindly every time."
            )
            st.info(
                "**Conditionals** let code make decisions. This is exactly how sentiment "
                "analysis turns a raw score into a label like 'Positive' or 'Negative' — "
                "which you'll see for real in Quick Tools."
            )

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Sentiment analysis, spam detection, and many classification tasks boil "
                    "down to conditionals underneath: \"if the score is above this threshold, "
                    "call it positive.\" Even modern AI models ultimately produce a number that "
                    "a conditional turns into a human-readable label, exactly like this lesson."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- `=` assigns a value, `==` compares two values — mixing them up is one of "
                    "the most common bugs in any language.\n"
                    "- Order matters in `if / elif / else`: Python checks each condition top to "
                    "bottom and stops at the *first* one that's true.\n"
                    "- Don't forget the final `else` — without it, values that don't match any "
                    "condition are silently skipped."
                )

            score_input = st.slider("Pretend this is a sentiment score:", -1.0, 1.0, 0.3, 0.1)

            code = f'''# Setup: none needed — uses Python's built-in if/elif/else, no installs required.
score = {score_input}

if score > 0.1:
    label = "Positive"
elif score < -0.1:
    label = "Negative"
else:
    label = "Neutral"

print(label)'''

            def show_conditional_output():
                if score_input > 0.1:
                    label = "Positive "
                elif score_input < -0.1:
                    label = "Negative "
                else:
                    label = "Neutral "
                st.metric("Label", label)
                st.caption("Try dragging the slider to different values and watch the label change.")

            code_and_output(code, show_conditional_output, key="lesson4_conditionals")


        # --- Lesson 5: Functions ---
        if lesson_idx == 5:
            st.subheader("Functions")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>7 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "A **function is a mini-recipe you can reuse**. Instead of rewriting the same "
                "steps every time, you define them once, give the recipe a name, and \"call\" "
                "it whenever you need it — with different ingredients (inputs) each time."
            )
            st.info("A **function** packages reusable logic. This is exactly how real NLP pipelines clean text before analysis.")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Every real NLP pipeline is built from small, reusable functions chained "
                    "together: `clean_text()`, `tokenize()`, `remove_stopwords()`, "
                    "`get_sentiment()`. Once you can write one function like `clean_text()` "
                    "below, you already understand the basic building block of every NLP tool "
                    "you'll use in the Quick Tools section."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- Forgetting `return`: a function can *do* work without giving anything "
                    "back, unless you explicitly `return` a value.\n"
                    "- Defining a function doesn't run it — you still have to *call* it, e.g. "
                    "`clean_text(user_text)`.\n"
                    "- Parameter names inside the function (like `text`) are just local labels — "
                    "they don't need to match the variable name you pass in."
                )

            user_text = st.text_area(
                "Paste a messy sentence (extra punctuation, CAPS, etc.):",
                "WOW!!! This Product is AMAZING... totally worth it!!",
            )

            code = '''# Setup: none needed — "string" is part of Python's standard library, no installs required.
import string

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

result = clean_text(user_text)
print(result)'''

            def show_clean_output():
                st.write("**Before:**")
                st.code(user_text, language="text")
                st.write("**After `clean_text()`:**")
                st.code(clean_text(user_text), language="text")

            code_and_output(code, show_clean_output, key="lesson5_functions")


        # --- Lesson 6: Dictionaries ---
        if lesson_idx == 6:
            st.subheader("Dictionaries")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>6 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Think of a **dictionary like a real phonebook**: instead of scanning every "
                "entry one by one (like a list), you look up a *key* (a name) and instantly "
                "get its *value* (a phone number). Dictionaries store `key: value` pairs."
            )
            st.info(
                "A **dictionary** (`dict`) maps unique keys to values. Unlike lists, you "
                "don't access items by position — you access them by name."
            )

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Word frequency counting — exactly what the **Keyword Extraction** Quick "
                    "Tool does — is fundamentally a dictionary: each word is a key, and its "
                    "count is the value. Stopword lookups, sentiment lexicons (word → score), "
                    "and label mappings are all dictionaries under the hood."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- Accessing a missing key with `d[\"missing\"]` raises a `KeyError` — use "
                    "`d.get(\"missing\", 0)` to safely provide a default instead.\n"
                    "- Dictionary keys must be unique — assigning the same key twice overwrites "
                    "the previous value rather than adding a second entry.\n"
                    "- Dictionaries are unordered by *meaning* even though Python (3.7+) happens "
                    "to preserve insertion order — don't rely on dict order the way you would a list."
                )

            sentence_input = st.text_area(
                "Type a sentence to count word frequency:",
                "the cat sat on the mat",
                height=100, key="lesson6_dict_input",
            )

            code = f'''# Setup: none needed — uses Python's built-in dict, no installs required.
text = "{sentence_input}"
word_counts = {{}}

for word in text.split():
    word_counts[word] = word_counts.get(word, 0) + 1

print(word_counts)'''

            def show_dict_output():
                word_counts = {}
                for w in sentence_input.split():
                    word_counts[w] = word_counts.get(w, 0) + 1
                st.write("**Word counts:**", word_counts)
                if word_counts:
                    top_word = max(word_counts, key=word_counts.get)
                    st.success(f"Most frequent word: **{top_word}** ({word_counts[top_word]}x)")

            code_and_output(code, show_dict_output, key="lesson6_dictionaries")


        # --- Lesson 7: DataFrames ---
        if lesson_idx == 7:
            st.subheader("DataFrames")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>6 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "A **DataFrame** (from the `pandas` library) is a table in code — rows and "
                "named columns, just like a spreadsheet. You've already seen DataFrames in "
                "action: every result table in **Quick Tools** (keywords, entities) is one."
            )
            st.info(
                "`pandas.DataFrame` is the standard way Python handles tabular data. It's "
                "usually built from a dictionary of equal-length lists — one list per column."
            )

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Every Quick Tool result you've clicked through — keyword frequency "
                    "tables, entity lists — is a `pandas.DataFrame` under the hood. Once you "
                    "recognize this pattern, you can build your own result tables from any "
                    "NLP output: words, scores, labels, counts."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- All lists passed into `pd.DataFrame({...})` must be the **same length**, "
                    "or you'll get an error.\n"
                    "- Column names are just dictionary keys — case-sensitive strings.\n"
                    "- `sort_values()` returns a *new* sorted DataFrame; it doesn't sort in place "
                    "unless you pass `inplace=True`."
                )

            words_raw = st.text_input(
                "Enter a few words, comma-separated:",
                "python, nlp, data, ai, learning",
                key="df_words_input",
            )
            words_list = [w.strip() for w in words_raw.split(",") if w.strip()]

            code = f'''# Setup (run once, e.g. in Colab): !pip install pandas
# (pandas is preinstalled in Colab and most Python environments already)
import pandas as pd

words = {words_list}
lengths = [len(w) for w in words]

df = pd.DataFrame({{"word": words, "length": lengths}})
print(df.sort_values("length", ascending=False))'''

            def show_dataframe_output():
                if words_list:
                    lengths_list = [len(w) for w in words_list]
                    df = pd.DataFrame({"word": words_list, "length": lengths_list})
                    st.dataframe(
                        df.sort_values("length", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning("Enter at least one word above.")

            code_and_output(code, show_dataframe_output, key="lesson7_dataframes")


    # -----------------------------------------------------------------------
    # GUIDED LEARNING — Module 2: Working with Text Data
    # -----------------------------------------------------------------------

        lesson_footer(track, lesson_idx, PY_TRACKS[track]["lessons"], f"nav_lesson_{track}")

    elif mode == "Guided Learning" and track == "Working with Text Data":
        st.markdown(
            "<div class='module-head'><span class='badge badge-amber'>Module 2 · Intermediate</span>"
            "<h2>Working with Text Data</h2>"
            "<p>Real text lives in files, arrives in the wrong encoding, and hides inside PDFs. "
            "This is the bridge between Fundamentals and real NLP.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'><b>The bridge between Fundamentals and real NLP.</b> "
            "Module 1 taught you Python using text that was already typed into the page. "
            "Real text arrives in files, in the wrong encoding, full of HTML and URLs, "
            "sometimes inside a PDF — and your code breaks. These seven lessons are about "
            "that gap.<br><br>"
            "<b>Prerequisites:</b> Fundamentals lessons on strings, lists/loops, functions "
            "and DataFrames. If those feel shaky, go back first — this module assumes them."
            "</div>",
            unsafe_allow_html=True,
        )


        # --- M2 L0: Reading & Writing Files ---
        if lesson_idx == 0:
            st.subheader("Reading & Writing Files")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>6 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Everything you've analysed so far was typed directly into code. Real text "
                "lives in files. **`open()`** is how Python reaches it — and the `with` "
                "statement is how you avoid leaking it."
            )
            st.info(
                "**Always use `with open(...) as f:`** — it closes the file automatically, "
                "even if your code crashes midway. Calling `open()` without `with` leaves the "
                "file handle open, which on large jobs will eventually exhaust your OS limit."
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Every corpus you'll ever work with is a file or a folder of files. "
                    "Reading them correctly — line by line for large files, all at once for "
                    "small ones — is the first step of literally every pipeline."
                )
            with st.expander("Common mistakes"):
                st.markdown(
                    "- **`\"w\"` silently erases the file.** Opening in write mode truncates "
                    "it to zero bytes before you write anything. Use `\"a\"` to append.\n"
                    "- **Forgetting `encoding=`.** Python uses your OS default, which differs "
                    "between Windows and Linux — so code that works on your laptop breaks on "
                    "a server. Always state `encoding=\"utf-8\"` explicitly.\n"
                    "- **`.read()` on a huge file** loads it entirely into memory. Iterate "
                    "line by line instead."
                )

            code = '''# Setup: none needed — open() is built into Python
import tempfile, os

folder = tempfile.mkdtemp()
path = os.path.join(folder, "notes.txt")

# WRITE — "w" creates the file (and erases it if it already exists)
with open(path, "w", encoding="utf-8") as f:
    f.write("First line\\n")
    f.write("Second line\\n")
    f.write("Third line\\n")

# READ — the whole file as one string
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
print(repr(content))

# READ — line by line (memory-friendly for big files)
with open(path, encoding="utf-8") as f:
    lines = [line.strip() for line in f]
print(lines)
print("Number of lines:", len(lines))'''

            def show_m2_files():
                import tempfile, os
                folder = tempfile.mkdtemp()
                path = os.path.join(folder, "notes.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("First line\n"); f.write("Second line\n"); f.write("Third line\n")
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                st.code(repr(content), language="text")
                with open(path, encoding="utf-8") as f:
                    lines = [line.strip() for line in f]
                st.write(lines)
                st.caption(f"Number of lines: **{len(lines)}**")
                st.caption(
                    "Note `repr()` shows the `\\n` newline characters explicitly — "
                    "`print()` would hide them. Useful when debugging whitespace."
                )

            code_and_output(code, show_m2_files, key="m2_l0_files")
            st.warning(
                "**On this app, files you write disappear.** Streamlit Cloud gives each "
                "session a temporary filesystem that's wiped when the app restarts. The code "
                "above is real and correct — run it in Colab or locally to keep the output."
            )

        # --- M2 L1: Encodings ---
        if lesson_idx == 1:
            st.subheader("Encodings — why text breaks")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>7 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Computers store **bytes**, not letters. An *encoding* is the agreed mapping "
                "between the two. Use the wrong one and you get `Ã©` where you wanted `é`, "
                "or an outright crash. This is the single most common wall beginners hit on "
                "their first real dataset."
            )
            st.info(
                "**UTF-8 is the answer almost always.** It handles every language plus "
                "emoji. When a file refuses to load, it's usually older Latin-1 (Windows-1252) "
                "data — and the fix is to say so explicitly, not to guess."
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Indian-language text, accented European names, currency symbols like ₹, "
                    "and emoji in social media data **all** depend on correct encoding. A "
                    "mis-decoded corpus silently corrupts every downstream count and token."
                )
            with st.expander("Common mistakes"):
                st.markdown(
                    "- **Assuming `len()` counts bytes.** It counts *characters*. "
                    "`\"₹\"` is 1 character but 3 bytes in UTF-8.\n"
                    "- **Using `errors='ignore'`** to make an error go away. It silently "
                    "*deletes* the characters it can't handle — you lose data and never know.\n"
                    "- `errors='replace'` is more honest: you see `�` where loss happened."
                )

            code = '''# Setup: none needed — encoding is built into Python
s = "Café — naïve résumé ₹500"

print("Characters:", len(s))
print("Bytes in UTF-8:", len(s.encode("utf-8")))

# A file saved as Latin-1, read as UTF-8 -> crash
latin_bytes = "Café".encode("latin-1")
print("Latin-1 bytes:", latin_bytes)

try:
    latin_bytes.decode("utf-8")
except UnicodeDecodeError as e:
    print("UnicodeDecodeError:", e)

# Decoding with the RIGHT encoding works
print("Correct:", latin_bytes.decode("latin-1"))

# errors="replace" shows you exactly where data was lost
print("Replaced:", latin_bytes.decode("utf-8", errors="replace"))'''

            def show_m2_encoding():
                s = "Café — naïve résumé ₹500"
                c1, c2 = st.columns(2)
                c1.metric("Characters", len(s))
                c2.metric("Bytes (UTF-8)", len(s.encode("utf-8")))
                latin_bytes = "Café".encode("latin-1")
                st.write("Latin-1 bytes:", latin_bytes)
                try:
                    latin_bytes.decode("utf-8")
                except UnicodeDecodeError as e:
                    st.error(f"UnicodeDecodeError: {e}")
                st.success(f'Correct (latin-1): {latin_bytes.decode("latin-1")}')
                st.warning(f'errors="replace": {latin_bytes.decode("utf-8", errors="replace")}')
                st.caption(
                    "24 characters but 32 bytes — the accented letters and ₹ take "
                    "more than one byte each. That gap is why encoding exists."
                )

            code_and_output(code, show_m2_encoding, key="m2_l1_encoding")
            st.caption(
                "This is exactly the fallback the **Quick Tools** file uploader uses: "
                "try UTF-8, fall back to Latin-1, and *tell you* it happened."
            )

        # --- M2 L2: Regex ---
        if lesson_idx == 2:
            st.subheader("Regular Expressions")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>10 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "A **regex** is a pattern for finding structure inside text — emails, phone "
                "numbers, dates, IDs. It's the sharpest tool in text processing and the "
                "easiest to cut yourself on."
            )
            st.info(
                "**The building blocks:** `\\d` a digit · `\\w` a word character · "
                "`\\s` whitespace · `+` one or more · `*` zero or more · `{2,}` at least two · "
                "`[abc]` any of these · `[^>]` anything except `>`"
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Regex does the work *before* the linguistics: stripping HTML, pulling out "
                    "identifiers, redacting personal data, splitting on custom boundaries. "
                    "Tokenizers themselves are built out of regexes."
                )
            with st.expander("Common mistakes"):
                st.markdown(
                    "- **Over-capturing.** The classic: a lazy email pattern swallows the "
                    "sentence's full stop. There's a live demonstration of exactly this below.\n"
                    "- **Forgetting the raw string `r\"...\"`.** Without it, `\\d` is "
                    "interpreted by Python before regex ever sees it.\n"
                    "- **Trying to parse HTML with regex.** Use a parser. Regex is for "
                    "*patterns*, not nested structure."
                )

            regex_text = st.text_input(
                "Text to search:",
                "Contact: priya@example.com or ravi@example.org. Call +91-98765-43210. Order #A1234 on 2026-08-02.",
                key="m2_regex_input",
            )

            code = '''# Setup: none needed — re is in Python's standard library
import re

text = "Contact: priya@example.com or ravi@example.org. Call +91-98765-43210."

# A NAIVE email pattern — note what it does to the second address
naive = r"[\\w.+-]+@[\\w-]+\\.[\\w.]+"
print("naive:", re.findall(naive, text))

# FIXED: require letters at the end, so the sentence's full stop is excluded
fixed = r"[\\w.+-]+@[\\w-]+\\.[a-zA-Z]{2,}"
print("fixed:", re.findall(fixed, text))

print("dates:", re.findall(r"\\d{4}-\\d{2}-\\d{2}", text))
print("redacted:", re.sub(fixed, "[EMAIL]", text))'''

            def show_m2_regex():
                import re as _re
                t = regex_text
                naive = r"[\w.+-]+@[\w-]+\.[\w.]+"
                fixed = r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"
                n_res, f_res = _re.findall(naive, t), _re.findall(fixed, t)
                st.markdown("**Naive email pattern**")
                st.write(n_res)
                st.markdown("**Fixed email pattern**")
                st.write(f_res)
                if n_res != f_res:
                    st.error(
                        "**Look closely — they differ.** The naive pattern ends with "
                        "`[\\w.]+`, and `.` is in that set, so it swallows the full stop "
                        "at the end of the sentence. This is a real bug I hit writing this "
                        "very lesson, not a hypothetical one."
                    )
                st.markdown("**Other patterns**")
                st.write({
                    "dates (YYYY-MM-DD)": _re.findall(r"\d{4}-\d{2}-\d{2}", t),
                    "order IDs (#A1234)": _re.findall(r"#[A-Z]\d+", t),
                    "phone (+91-xxxxx-xxxxx)": _re.findall(r"\+\d{2}-\d{5}-\d{5}", t),
                })
                st.markdown("**Redacting emails with `re.sub`**")
                st.code(_re.sub(fixed, "[EMAIL]", t), language="text")

            code_and_output(code, show_m2_regex, key="m2_l2_regex")

        # --- M2 L3: Cleaning Text ---
        if lesson_idx == 3:
            st.subheader("Cleaning Text")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>7 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Real text is filthy: HTML tags, URLs, runaway whitespace, inconsistent case. "
                "Cleaning is a **sequence of small, ordered transformations** — and, as in the "
                "Preprocessing Pipeline tool, the order changes the result."
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Skip cleaning and `<b>great</b>` and `great` count as different words, "
                    "URLs dominate your keyword list, and `The` and `the` split every count "
                    "in two. Cleaning is what makes the numbers mean anything."
                )
            with st.expander("Common mistakes"):
                st.markdown(
                    "- **Cleaning too aggressively.** Stripping all punctuation destroys "
                    "sentence boundaries — so `sent_tokenize` stops working afterwards.\n"
                    "- **Lowercasing before NER.** Capitalisation is a *signal* for entity "
                    "recognition. Lowercase first and you damage the thing you're about to run.\n"
                    "- **Assuming one cleaning recipe fits all tasks.** Sentiment needs "
                    "punctuation and negation kept; keyword extraction doesn't."
                )

            messy_default = "  The   PRODUCT is <b>GREAT</b>!!! Visit https://example.com NOW!!!  "
            messy = st.text_area("Messy text to clean:", messy_default, height=80, key="m2_clean_input")

            code = '''# Setup: none needed — re is in Python's standard library
import re

messy = "  The   PRODUCT is <b>GREAT</b>!!! Visit https://example.com NOW!!!  "

step1 = messy.strip()                        # trim the ends
step2 = re.sub(r"<[^>]+>", "", step1)        # remove HTML tags
step3 = re.sub(r"https?://\\S+", "", step2)   # remove URLs
step4 = re.sub(r"\\s+", " ", step3)           # collapse runs of whitespace
step5 = step4.lower().strip()                # normalise case

for name, val in [("strip", step1), ("de-HTML", step2), ("de-URL", step3),
                  ("collapse", step4), ("lower", step5)]:
    print(f"{name:10s} {val!r}")'''

            def show_m2_clean():
                import re as _re
                s1 = messy.strip()
                s2 = _re.sub(r"<[^>]+>", "", s1)
                s3 = _re.sub(r"https?://\S+", "", s2)
                s4 = _re.sub(r"\s+", " ", s3)
                s5 = s4.lower().strip()
                rows = [("0. original", messy), ("1. strip", s1), ("2. de-HTML", s2),
                        ("3. de-URL", s3), ("4. collapse whitespace", s4), ("5. lowercase", s5)]
                st.dataframe(
                    pd.DataFrame([(n, repr(v), len(v)) for n, v in rows],
                                 columns=["Step", "Result", "Length"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    f"Length went from **{len(messy)}** to **{len(s5)}** characters. "
                    "`repr()` is used so you can see the whitespace."
                )

            code_and_output(code, show_m2_clean, key="m2_l3_clean")

        # --- M2 L4: Text in pandas ---
        if lesson_idx == 4:
            st.subheader("Text in pandas — the `.str` accessor")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>7 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "One document is a string. A thousand documents is a **column**. pandas' "
                "`.str` accessor applies string operations down an entire column at once — "
                "no loop required."
            )
            st.info(
                "`.str` methods **chain**: `df[\"col\"].str.strip().str.lower()` — each "
                "returns a new Series, so you build the pipeline left to right."
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "This is the bridge from single-text demos to real corpora. Every dataset "
                    "you meet — reviews, tweets, filings, support tickets — arrives as a "
                    "table with a text column."
                )
            with st.expander("Common mistakes"):
                st.markdown(
                    "- **Forgetting missing values.** `.str` methods return `NaN` for `None`, "
                    "and `.str.contains()` will raise unless you pass `na=False`.\n"
                    "- **Integer columns turning into floats.** Watch the output below: "
                    "`length` shows `14.0`, not `14`, purely because one row is `NaN` — "
                    "pandas has no integer type that supports missing values by default.\n"
                    "- **Looping with `.apply()` when a `.str` method exists.** `.str` is "
                    "clearer and usually faster."
                )

            code = '''# Setup (run once, e.g. in Colab): !pip install pandas
import pandas as pd

df = pd.DataFrame({"review": [
    "  GREAT product!  ",
    "terrible, would not buy",
    "Okay I guess...",
    None,                       # a missing value, on purpose
]})

df["clean"]     = df["review"].str.strip().str.lower()
df["length"]    = df["clean"].str.len()
df["has_great"] = df["clean"].str.contains("great", na=False)
df["words"]     = df["clean"].str.split().str.len()

print(df)'''

            def show_m2_pandas():
                df_demo = pd.DataFrame({"review": [
                    "  GREAT product!  ", "terrible, would not buy", "Okay I guess...", None,
                ]})
                df_demo["clean"] = df_demo["review"].str.strip().str.lower()
                df_demo["length"] = df_demo["clean"].str.len()
                df_demo["has_great"] = df_demo["clean"].str.contains("great", na=False)
                df_demo["words"] = df_demo["clean"].str.split().str.len()
                st.dataframe(df_demo, use_container_width=True)
                st.caption(
                    "Row 3 is `None` and stays `NaN` throughout — pandas propagates "
                    "missing values instead of guessing. Note `length` is **14.0**, not 14: "
                    "one missing value forced the whole column to float."
                )

            code_and_output(code, show_m2_pandas, key="m2_l4_pandas")

        # --- M2 L5: Reading PDFs ---
        if lesson_idx == 5:
            st.subheader("Reading PDFs")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-green'>Practitioner</span>"
                "<span class='badge badge-blue'>8 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Most business and academic text arrives as PDF — annual reports, regulatory "
                "filings, research papers, policy documents. Getting text *out* is often the "
                "real first step of a project."
            )
            st.info(
                "A PDF describes **where marks go on a page**, not what the words are. "
                "There's no guaranteed reading order, no reliable paragraph structure. "
                "Extraction is therefore always approximate — expect to clean up afterwards."
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "An analyst asked to 'analyse sentiment in these 200 annual reports' "
                    "spends most of the effort here, not on the sentiment model. PDF handling "
                    "is one of the most practically valuable skills in this whole app."
                )
            with st.expander("Common mistakes"):
                st.markdown(
                    "- **Scanned PDFs contain no text at all** — just images of text. "
                    "`extract_text()` returns empty. Those need OCR (e.g. Tesseract), which "
                    "is a different and much heavier tool.\n"
                    "- **Tables come out scrambled.** Column layout is visual, not structural. "
                    "`pdfplumber` handles tables better than `pypdf`.\n"
                    "- **Assuming page order is reading order.** Multi-column academic papers "
                    "often extract as interleaved nonsense."
                )

            code = '''# Setup (run once, e.g. in Colab): !pip install pypdf
from pypdf import PdfReader

reader = PdfReader("annual_report.pdf")
print("Pages:", len(reader.pages))

# Extract one page
first = reader.pages[0].extract_text()
print(first[:300])

# Extract the whole document
full_text = "\\n".join(page.extract_text() or "" for page in reader.pages)
print("Total characters:", len(full_text))

# `or ""` matters: extract_text() returns None on pages with no
# extractable text (e.g. a scanned image), which would crash join().'''

            def show_m2_pdf():
                st.markdown("**Upload a PDF to try it on your own document**")
                up = st.file_uploader("PDF file", type=["pdf"], key="m2_pdf_upload")
                if up is not None:
                    try:
                        from pypdf import PdfReader
                        import io as _io
                        reader = PdfReader(_io.BytesIO(up.getvalue()))
                        pages = len(reader.pages)
                        full = "\n".join(p.extract_text() or "" for p in reader.pages)
                        c1, c2 = st.columns(2)
                        c1.metric("Pages", pages)
                        c2.metric("Characters extracted", f"{len(full):,}")
                        if not full.strip():
                            st.error(
                                "**No text extracted.** This is almost certainly a scanned "
                                "PDF — images of text rather than text. You'd need OCR here."
                            )
                        else:
                            st.text_area("Extracted text (preview)", full[:3000], height=200, key="m2_pdf_out")
                            if st.button("Send this to Quick Tools", key="m2_pdf_send"):
                                st.session_state.corpus_text = full
                                st.success("Loaded into the shared corpus — open Quick Tools to analyse it.")
                    except Exception as e:
                        st.error(f"Couldn't read that PDF: {e}")
                else:
                    st.info("Upload a PDF to see live extraction. No file is stored.")

            code_and_output(code, show_m2_pdf, key="m2_l5_pdf")

        # --- M2 L6: Errors & Debugging ---
        if lesson_idx == 6:
            st.subheader("Errors & Debugging")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>8 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Almost no course teaches this, and it's the single skill that separates "
                "someone who gets unstuck alone from someone who doesn't. **An error message "
                "is information, not failure.**"
            )
            st.info(
                "**Read a traceback bottom-up.** The last line names the error type and "
                "what went wrong. Work upward to find the line in *your* code — usually "
                "the lowest line mentioning a file you actually wrote."
            )
            with st.expander("Why this matters"):
                st.markdown(
                    "Every error below is one you *will* hit working with text: a token index "
                    "past the end of a list, a number that won't parse, a dictionary key that "
                    "isn't there, a string added to an integer."
                )

            code = '''# Setup: none needed — these are Python's built-in exceptions
# Each block deliberately triggers a real error, then catches it.

try:
    tokens = ["the", "cat", "sat"]
    print(tokens[5])                 # only indexes 0-2 exist
except IndexError as e:
    print("IndexError:", e)

try:
    count = int("abc")               # not a number
except ValueError as e:
    print("ValueError:", e)

try:
    counts = {"the": 5}
    print(counts["cat"])             # key was never added
except KeyError as e:
    print("KeyError:", e)

try:
    total = "5" + 5                  # str + int
except TypeError as e:
    print("TypeError:", e)'''

            def show_m2_errors():
                rows = []
                try:
                    ["the", "cat", "sat"][5]
                except IndexError as e:
                    rows.append(("IndexError", str(e), "Index past the end of the list"))
                try:
                    int("abc")
                except ValueError as e:
                    rows.append(("ValueError", str(e), "Right type, impossible value"))
                try:
                    {"the": 5}["cat"]
                except KeyError as e:
                    rows.append(("KeyError", str(e), "Key not in the dictionary"))
                try:
                    "5" + 5
                except TypeError as e:
                    rows.append(("TypeError", str(e), "Incompatible types"))
                st.dataframe(
                    pd.DataFrame(rows, columns=["Error", "Message", "What it means"]),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "These are real exceptions raised and caught live, not text I typed in."
                )

            code_and_output(code, show_m2_errors, key="m2_l6_errors")

            st.markdown("**Practise: what error does this raise?**")
            guess_snippet = st.selectbox(
                "Pick a snippet:",
                [
                    'int("12.5")',
                    '["a", "b"][2]',
                    '{"x": 1}["y"]',
                    '"total: " + 42',
                ],
                key="m2_err_guess",
            )
            guess = st.radio(
                "Your prediction:",
                ["IndexError", "ValueError", "KeyError", "TypeError"],
                horizontal=True,
                key="m2_err_choice",
            )
            if st.button("Check my answer", key="m2_err_check"):
                try:
                    eval(guess_snippet)
                    st.warning("That didn't raise an error at all.")
                except Exception as e:
                    actual = type(e).__name__
                    if actual == guess:
                        st.success(f"Correct — `{guess_snippet}` raises **{actual}**: {e}")
                    else:
                        st.error(f"Not quite. `{guess_snippet}` raises **{actual}**, not {guess}. ({e})")

            if st.button("Module 2 complete", key="m2_done"):
                st.success("You can now get real text out of real files. On to the NLTK and spaCy tracks →")

    # -----------------------------------------------------------------------
    # GUIDED LEARNING — NLTK Track
    # -----------------------------------------------------------------------

        lesson_footer(track, lesson_idx, PY_TRACKS[track]["lessons"], f"nav_lesson_{track}")

    elif mode == "Guided Learning" and track == "NLTK":
        st.markdown(
            "<div class='module-head'><span class='badge badge-nltk'>Module 3 · NLP</span>"
            "<h2>NLTK</h2>"
            "<p>The classic, rule-based toolkit — best for understanding how NLP works "
            "under the hood.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'><b>NLTK</b> (Natural Language Toolkit) is the classic, "
            "lexicon- and rule-based Python NLP library — great for understanding "
            "<i>how</i> NLP tasks work under the hood. Lessons 0, 1, 3 and 4 use the exact "
            "same example sentences as the <b>spaCy</b> track, so you can switch tracks "
            "and compare the two libraries directly on identical input.</div>",
            unsafe_allow_html=True,
        )

        SHARED_TEXT_TOKENS, SHARED_TEXT_ENTITIES = render_shared_text_editor()


        # --- NLTK Lesson 0: Tokenization ---
        if lesson_idx == 0:
            st.subheader("Tokenization")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "Tokenization splits text into individual pieces — words, punctuation, "
                "contractions — that downstream tasks (POS tagging, NER, etc.) operate on. "
                "NLTK's `word_tokenize` uses the Penn Treebank tokenization rules."
            )
            st.info(f"Shared example sentence: *\"{SHARED_TEXT_TOKENS}\"*")
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Every NLP pipeline starts here. Get tokenization wrong and everything "
                    "downstream — counts, tags, entities — is wrong too. Notice how `don't` "
                    "splits into `do` + `n't`, and `Corp's` splits into `Corp` + `'s`."
                )

            code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("punkt"); nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize

text = "{SHARED_TEXT_TOKENS}"
tokens = word_tokenize(text)
print(tokens)'''

            def show_nltk_tok():
                toks = word_tokenize(SHARED_TEXT_TOKENS)
                st.write(toks)
                st.caption(f"**{len(toks)} tokens.** Notice `n't` and `'s` are split off as their own tokens.")

            code_and_output(code, show_nltk_tok, key="nltk_lesson0_tokenize")

        # --- NLTK Lesson 1: Stopwords ---
        if lesson_idx == 1:
            st.subheader("Stopwords")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "**Stopwords** are common words (*the, is, at, and...*) that carry little "
                "meaning on their own and are often filtered out before analysis like "
                "keyword extraction."
            )
            st.info(f"Shared example sentence: *\"{SHARED_TEXT_TOKENS}\"*")
            with st.expander("Common mistake"):
                st.markdown(
                    "Don't always remove stopwords blindly — for sentiment analysis, words "
                    "like *not* or *no* are technically stopwords in some lists but change "
                    "meaning entirely (\"good\" vs \"not good\")."
                )

            code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "{SHARED_TEXT_TOKENS}"
stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text)
filtered = [t for t in tokens if t.lower() not in stop_words and t.isalpha()]
print(filtered)'''

            def show_nltk_stopwords():
                stop_words = set(stopwords.words("english"))
                toks = word_tokenize(SHARED_TEXT_TOKENS)
                filtered = [t for t in toks if t.lower() not in stop_words and t.isalpha()]
                st.write(filtered)
                st.caption(f"NLTK's default English stopword list has **{len(stop_words)} words**.")

            code_and_output(code, show_nltk_stopwords, key="nltk_lesson1_stopwords")

        # --- NLTK Lesson 2: Stemming ---
        if lesson_idx == 2:
            st.subheader("Stemming")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "**Stemming** chops words down to a crude root form using fixed rules — "
                "fast, but the result isn't always a real word. NLTK's classic algorithm "
                "is the **Porter Stemmer**."
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Stemming groups related words (*run, running, runs* → `run`) so they "
                    "count as \"the same word\" in tasks like keyword frequency. The tradeoff: "
                    "speed and simplicity over accuracy — compare this lesson's output to "
                    "the **spaCy → Lemmatization** lesson on the exact same words."
                )

            stem_words = ["running", "flies", "easily", "fairly", "studies", "argued", "universities"]
            code = f'''# Setup (run once, e.g. in Colab): none — PorterStemmer ships with NLTK
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = {stem_words}
print([(w, ps.stem(w)) for w in words])'''

            def show_nltk_stem():
                ps = PorterStemmer()
                rows = [(w, ps.stem(w)) for w in stem_words]
                st.dataframe(pd.DataFrame(rows, columns=["Word", "Stem"]), use_container_width=True, hide_index=True)
                st.caption(
                    "Notice `studies` → `studi` and `easily` → `easili` — not real words. "
                    "That's expected: stemming trades correctness for speed via fixed rules."
                )

            code_and_output(code, show_nltk_stem, key="nltk_lesson2_stem")

            st.markdown("<div class='lib-section'></div>", unsafe_allow_html=True)
            st.markdown("#### Now on the shared text — does vocabulary actually shrink?")
            st.info(f"Shared example sentence: *\"{SHARED_TEXT_TOKENS}\"*")
            st.caption(
                "The 7-word list above is a clean, curated example. Real text is messier — "
                "here's the same PorterStemmer applied to every word in the shared text, "
                "with the vocabulary size before and after so you can see stemming's actual "
                "effect: does it really reduce the number of distinct word-forms?"
            )

            code_shared_stem = f'''from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

text = "{SHARED_TEXT_TOKENS}"
ps = PorterStemmer()
words = [t for t in word_tokenize(text.lower()) if t.isalpha()]
stems = [ps.stem(w) for w in words]

print("Unique words:", len(set(words)))
print("Unique stems:", len(set(stems)))'''

            def show_shared_stem():
                ps = PorterStemmer()
                words = [t for t in word_tokenize(SHARED_TEXT_TOKENS.lower()) if t.isalpha()]
                stems = [ps.stem(w) for w in words]
                rows = list(zip(words, stems))
                st.dataframe(pd.DataFrame(rows, columns=["Word", "Stem"]), use_container_width=True, hide_index=True)

                n_unique_words = len(set(words))
                n_unique_stems = len(set(stems))
                c1, c2 = st.columns(2)
                c1.metric("Unique words before", n_unique_words)
                c2.metric("Unique stems after", n_unique_stems)
                if n_unique_stems < n_unique_words:
                    st.caption(
                        f"Stemming collapsed {n_unique_words - n_unique_stems} word-form(s) "
                        "together — e.g. plural/singular or verb tense variants that share a "
                        "root now count as the same term. On short text this reduction is "
                        "often small or zero; it becomes more visible on longer documents "
                        "with more repeated word families."
                    )
                else:
                    st.caption(
                        "No reduction here — this text happens to have few words sharing "
                        "a common root. Try a longer or more repetitive passage (or the "
                        "Classification & Clustering module's document boxes) to see stemming "
                        "actually shrink the vocabulary."
                    )

            code_and_output(code_shared_stem, show_shared_stem, key="nltk_lesson2_shared_stem")

        # --- NLTK Lesson 3: POS Tagging ---
        if lesson_idx == 3:
            st.subheader("Part-of-Speech (POS) Tagging")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "POS tagging labels each token with its grammatical role — noun, verb, "
                "adjective, etc. NLTK uses **Penn Treebank tags** (e.g. `NNP` = proper noun, "
                "`VBD` = past-tense verb)."
            )
            st.info(f"Shared example sentence: *\"{SHARED_TEXT_ENTITIES}\"*")

            code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
#     nltk.download(pkg)

from nltk import word_tokenize, pos_tag

text = "{SHARED_TEXT_ENTITIES}"
tagged = pos_tag(word_tokenize(text))
print(tagged)'''

            def show_nltk_pos():
                tagged = pos_tag(word_tokenize(SHARED_TEXT_ENTITIES))
                st.dataframe(pd.DataFrame(tagged, columns=["Token", "Tag"]), use_container_width=True, hide_index=True, height=250)
                st.caption("Full Penn Treebank tag reference: run `nltk.help.upenn_tagset()` locally to see what each code means.")

            code_and_output(code, show_nltk_pos, key="nltk_lesson3_pos")

        # --- NLTK Lesson 4: NER ---
        if lesson_idx == 4:
            st.subheader("Named Entity Recognition")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "NLTK's built-in NER (`ne_chunk`) is a rule-based chunker layered on top of "
                "POS tags — lightweight, but noticeably less accurate than model-based "
                "approaches (see the **spaCy → NER** lesson on this exact sentence)."
            )
            st.info(f"Shared example sentence: *\"{SHARED_TEXT_ENTITIES}\"*")
            with st.expander("Common mistake"):
                st.markdown(
                    "Don't assume `ne_chunk` output is production-accurate. On this exact "
                    "sentence NLTK makes **three** verifiable errors: it tags **\"CFO\"** as an "
                    "ORGANIZATION, **\"Example Corp\"** as a PERSON, and **\"Bengaluru\"** as a "
                    "PERSON. It also detects no dates and no monetary amounts at all — those "
                    "categories aren't in its label set."
                )

            code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
#             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
#             "maxent_ne_chunker_tab", "words"]:
#     nltk.download(pkg)

from nltk import word_tokenize, pos_tag, ne_chunk

text = "{SHARED_TEXT_ENTITIES}"
tree = ne_chunk(pos_tag(word_tokenize(text)))
for subtree in tree:
    if hasattr(subtree, "label"):
        print(subtree.label(), "->", " ".join(w for w, t in subtree.leaves()))'''

            def show_nltk_ner():
                tree = ne_chunk(pos_tag(word_tokenize(SHARED_TEXT_ENTITIES)))
                ents = []
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        ents.append((" ".join(w for w, t in subtree.leaves()), subtree.label()))
                if ents:
                    st.dataframe(pd.DataFrame(ents, columns=["Entity", "Type"]), use_container_width=True, hide_index=True)
                else:
                    st.info("No entities detected.")

            code_and_output(code, show_nltk_ner, key="nltk_lesson4_ner")

        # --- NLTK Lesson 5: Sentiment (VADER) ---
        if lesson_idx == 5:
            st.subheader("Sentiment Analysis (VADER)")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a "
                "lexicon-based sentiment tool bundled with NLTK, tuned for short, informal "
                "text like reviews and social media posts."
            )
            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "spaCy has **no built-in equivalent** to VADER — that's a real, "
                    "structural difference between the two libraries, not a gap in this app. "
                    "Real spaCy projects add sentiment via a separate package or a trained model."
                )

            sample_text = st.text_input(
                "Try your own sentence:",
                "I absolutely loved how smooth the checkout process was, though the delivery was a bit slow.",
                key="nltk_sentiment_input",
            )

            code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = "{sample_text}"
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(text))'''

            def show_nltk_vader():
                if sample_text.strip():
                    sia = SentimentIntensityAnalyzer()
                    scores = sia.polarity_scores(sample_text)
                    st.write(scores)
                    compound = scores["compound"]
                    label = "Positive " if compound > 0.05 else "Negative " if compound < -0.05 else "Neutral "
                    st.metric("Overall", label)
                else:
                    st.warning("Enter a sentence above.")

            code_and_output(code, show_nltk_vader, key="nltk_lesson5_vader")

        # --- NLTK Lesson 6: Keyword Extraction (TF) ---
        # Promoted from Quick Tools into a taught lesson — this was previously
        # only runnable, never explained step by step in Guided Learning.
        if lesson_idx == 6:
            st.subheader("Keyword Extraction")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "The simplest form of **keyword extraction** is counting word frequency "
                "after removing stopwords — this is **Term Frequency (TF)**, the same "
                "concept taught more formally in the Classification & Clustering module. "
                "Here's the NLTK version on our shared example sentence."
            )
            st.info(f"Shared example sentence: *\"{SHARED_TEXT_ENTITIES}\"*")

            code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

text = "{SHARED_TEXT_ENTITIES}"
stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text.lower())
keywords = [t for t in tokens if t.isalpha() and t not in stop_words]

print(Counter(keywords).most_common(10))'''

            def show_nltk_keywords():
                stop_words = set(stopwords.words("english"))
                tokens = word_tokenize(SHARED_TEXT_ENTITIES.lower())
                keywords = [t for t in tokens if t.isalpha() and t not in stop_words]
                freq = Counter(keywords).most_common(10)
                if freq:
                    kw_df = pd.DataFrame(freq, columns=["Keyword", "Frequency"])
                    st.dataframe(kw_df, use_container_width=True, hide_index=True)
                    st.bar_chart(kw_df.set_index("Keyword"))
                else:
                    st.warning("No keywords found.")

            code_and_output(code, show_nltk_keywords, key="nltk_lesson6_keywords")

            st.caption(
                "Try the **Quick Tools → Keyword Extraction** tool to run this on "
                "your own text, with NLTK and spaCy shown side by side."
            )

        # --- NLTK Lesson 7: Word Cloud ---
        if lesson_idx == 7:
            st.subheader("Word Cloud")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "A word cloud is a **visualization** of exactly the keyword frequencies "
                "from the previous lesson — same numbers, sized by frequency instead of "
                "listed in a table. It adds no new NLP technique; it's a picture of one "
                "you've already learned."
            )

            code = f'''# Setup: pip install wordcloud matplotlib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "{SHARED_TEXT_ENTITIES}"
stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text.lower())
words = [t for t in tokens if t.isalpha() and t not in stop_words]
freq = dict(Counter(words))

wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()'''

            def show_nltk_wordcloud():
                stop_words = set(stopwords.words("english"))
                tokens = word_tokenize(SHARED_TEXT_ENTITIES.lower())
                words = [t for t in tokens if t.isalpha() and t not in stop_words]
                freq = dict(Counter(words))
                if not freq:
                    st.warning("No words left to plot.")
                    return
                wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate_from_frequencies(freq)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

            code_and_output(code, show_nltk_wordcloud, key="nltk_lesson7_wordcloud")

        # --- NLTK Lesson 8: Frequency Distributions ---
        if lesson_idx == 8:
            st.subheader("Frequency Distributions")
            st.markdown("<span class='badge badge-nltk'>NLTK</span>", unsafe_allow_html=True)
            st.markdown(
                "NLTK has a purpose-built object for exactly the counting you've been doing "
                "informally since **Keyword Extraction**: `FreqDist`. It's a specialized "
                "dictionary of counts with convenience methods built in — `most_common()`, "
                "plotting, and `hapaxes()` (words that appear **exactly once**, often the "
                "most distinctive ones in a document)."
            )
            st.info(f"Shared example sentence: *\"{SHARED_TEXT_ENTITIES}\"*")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "This is descriptive statistics for text — the same instinct as `.describe()` "
                    "on a numeric DataFrame column, applied to word counts instead. Before "
                    "reaching for TF-IDF, classification, or clustering, a `FreqDist` is "
                    "usually the first thing worth looking at: how many distinct words are "
                    "there, how skewed is the distribution (a handful of words dominating, "
                    "or fairly even), and are there hapaxes worth investigating."
                )

            code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

text = "{SHARED_TEXT_ENTITIES}"
stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text.lower())
words = [t for t in tokens if t.isalpha() and t not in stop_words]

fd = FreqDist(words)
print("Distinct words:", fd.B())          # B() = number of distinct word "bins"
print("Total word count:", fd.N())        # N() = total token count
print("Most common:", fd.most_common(10))
print("Hapaxes (appear once):", fd.hapaxes())'''

            def show_freqdist():
                stop_words = set(stopwords.words("english"))
                tokens = word_tokenize(SHARED_TEXT_ENTITIES.lower())
                words = [t for t in tokens if t.isalpha() and t not in stop_words]
                if not words:
                    st.warning("No words left after removing stopwords — try a longer or different shared text.")
                    return

                from nltk import FreqDist
                fd = FreqDist(words)

                c1, c2, c3 = st.columns(3)
                c1.metric("Distinct words", fd.B())
                c2.metric("Total word count", fd.N())
                c3.metric("Hapaxes (appear once)", len(fd.hapaxes()))

                top = fd.most_common(10)
                top_df = pd.DataFrame(top, columns=["Word", "Count"])
                st.dataframe(top_df, use_container_width=True, hide_index=True)
                st.bar_chart(top_df.set_index("Word"))

                hapaxes = fd.hapaxes()
                if hapaxes:
                    shown = ", ".join(f"`{w}`" for w in hapaxes[:15])
                    more = f" (+{len(hapaxes) - 15} more)" if len(hapaxes) > 15 else ""
                    st.caption(
                        f"**{len(hapaxes)} hapax(es)** — words appearing exactly once: "
                        f"{shown}{more}. On short text almost every word is a hapax (nothing "
                        "repeats yet); on longer documents, a high hapax count relative to "
                        "total words is itself a descriptive statistic — it tells you the "
                        "vocabulary is diverse rather than repetitive."
                    )
                else:
                    st.caption("No hapaxes — every word here repeats at least once.")

            code_and_output(code, show_freqdist, key="nltk_lesson8_freqdist")

            if st.button("NLTK track complete", key="nltk_done"):
                st.success("Nice work! Try the spaCy track next to see the same tasks with a model-based library →")

    # -----------------------------------------------------------------------
    # GUIDED LEARNING — spaCy Track
    # -----------------------------------------------------------------------

        lesson_footer(track, lesson_idx, PY_TRACKS[track]["lessons"], f"nav_lesson_{track}")

    elif mode == "Guided Learning" and track == "spaCy":
        st.markdown(
            "<div class='module-head'><span class='badge badge-spacy'>Module 4 · NLP</span>"
            "<h2>spaCy</h2>"
            "<p>Modern and model-based. The same tasks as the NLTK module, on the same "
            "sentences, so the differences are real rather than incidental.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'><b>spaCy</b> is a modern, model-based NLP library — "
            "faster to get accurate results from, but more of a \"black box\" than NLTK's "
            "explicit rules. Lessons 0, 1, 3 and 4 reuse the exact same example sentences "
            "as the <b>NLTK</b> track for direct comparison.</div>",
            unsafe_allow_html=True,
        )

        SHARED_TEXT_TOKENS, SHARED_TEXT_ENTITIES = render_shared_text_editor()

        nlp_spacy, spacy_ok = get_spacy_model()
        if not spacy_ok:
            st.error("spaCy couldn't load right now. Refresh and try again in a moment.")
        else:

            # --- spaCy Lesson 0: Tokenization ---
            if lesson_idx == 0:
                st.subheader("Tokenization")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "spaCy tokenizes text using a combination of rules and exceptions "
                    "(not a statistical model, for tokenization specifically)."
                )
                st.info(f"Shared example sentence: *\"{SHARED_TEXT_TOKENS}\"*")

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = "{SHARED_TEXT_TOKENS}"
doc = nlp(text)
print([t.text for t in doc])'''

                def show_spacy_tok():
                    doc = nlp_spacy(SHARED_TEXT_TOKENS)
                    toks = [t.text for t in doc]
                    st.write(toks)
                    st.caption(f"**{{len(toks)}} tokens.**")

                code_and_output(code, show_spacy_tok, key="spacy_lesson0_tokenize")
                st.markdown(
                    "<div class='lib-section'></div>"
                    "**NLTK vs spaCy here:** on this sentence, both split `don't` → "
                    "`do` + `n't` and `Corp's` → `Corp` + `'s` identically — for common "
                    "contractions, the two tokenizers largely agree. Differences show up more "
                    "on messier real-world text (URLs, emojis, unusual punctuation).",
                    unsafe_allow_html=True,
                )

            # --- spaCy Lesson 1: Stopwords ---
            if lesson_idx == 1:
                st.subheader("Stopwords")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "spaCy flags stopwords via a per-token boolean attribute, `token.is_stop`, "
                    "rather than a separate list you filter against manually."
                )
                st.info(f"Shared example sentence: *\"{SHARED_TEXT_TOKENS}\"*")

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = "{SHARED_TEXT_TOKENS}"
doc = nlp(text)
filtered = [t.text for t in doc if not t.is_stop and t.is_alpha]
print(filtered)'''

                def show_spacy_stopwords():
                    doc = nlp_spacy(SHARED_TEXT_TOKENS)
                    filtered = [t.text for t in doc if not t.is_stop and t.is_alpha]
                    st.write(filtered)

                code_and_output(code, show_spacy_stopwords, key="spacy_lesson1_stopwords")
                st.markdown(
                    "<div class='lib-section'></div>"
                    "**NLTK vs spaCy here:** run the same sentence through the "
                    "**NLTK → Stopwords** lesson and compare — NLTK's default list keeps "
                    "\"everyone\" while spaCy's marks it as a stopword. The two libraries "
                    "ship *different* default stopword lists, so results genuinely diverge.",
                    unsafe_allow_html=True,
                )

            # --- spaCy Lesson 2: Lemmatization ---
            if lesson_idx == 2:
                st.subheader("Lemmatization")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "**Lemmatization** reduces a word to its real dictionary base form "
                    "(the *lemma*), using vocabulary and grammar rules — slower than stemming, "
                    "but the output is always a real word."
                )
                with st.expander("Why this matters for NLP"):
                    st.markdown(
                        "Compare this directly to the **NLTK → Stemming** lesson, run on "
                        "the identical word list. `studies` → `study` (correct) here, vs "
                        "`studi` (truncated) there."
                    )

                stem_words = ["running", "flies", "easily", "fairly", "studies", "argued", "universities"]
                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

words = {stem_words}
doc = nlp(" ".join(words))
print([(t.text, t.lemma_) for t in doc])'''

                def show_spacy_lemma():
                    doc = nlp_spacy(" ".join(stem_words))
                    rows = [(t.text, t.lemma_) for t in doc]
                    st.dataframe(pd.DataFrame(rows, columns=["Word", "Lemma"]), use_container_width=True, hide_index=True)

                code_and_output(code, show_spacy_lemma, key="spacy_lesson2_lemma")

                st.markdown("<div class='lib-section'></div>", unsafe_allow_html=True)
                st.markdown("#### Now on the shared text — does vocabulary actually shrink?")
                st.info(f"Shared example sentence: *\"{SHARED_TEXT_TOKENS}\"*")
                st.caption(
                    "The 7-word list above is a clean, curated example. Real text is "
                    "messier — here's spaCy's lemmatizer applied to every word in the "
                    "shared text, with the vocabulary size before and after, for a direct "
                    "comparison with the **NLTK → Stemming** lesson's same measurement."
                )

                code_shared_lemma = f'''import spacy
nlp = spacy.load("en_core_web_sm")

text = "{SHARED_TEXT_TOKENS}"
doc = nlp(text)
words = [t.text.lower() for t in doc if t.is_alpha]
lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]

print("Unique words:", len(set(words)))
print("Unique lemmas:", len(set(lemmas)))'''

                def show_shared_lemma():
                    doc = nlp_spacy(SHARED_TEXT_TOKENS)
                    words = [t.text.lower() for t in doc if t.is_alpha]
                    lemmas = [t.lemma_.lower() for t in doc if t.is_alpha]
                    rows = [(t.text, t.lemma_) for t in doc if t.is_alpha]
                    st.dataframe(pd.DataFrame(rows, columns=["Word", "Lemma"]), use_container_width=True, hide_index=True)

                    n_unique_words = len(set(words))
                    n_unique_lemmas = len(set(lemmas))
                    c1, c2 = st.columns(2)
                    c1.metric("Unique words before", n_unique_words)
                    c2.metric("Unique lemmas after", n_unique_lemmas)
                    if n_unique_lemmas < n_unique_words:
                        st.caption(
                            f"Lemmatization collapsed {n_unique_words - n_unique_lemmas} "
                            "word-form(s) together, and — unlike stemming — every collapsed "
                            "form is still a real dictionary word. Compare this reduction "
                            "count to the NLTK Stemming lesson's on the same text: they "
                            "often differ, since the two use completely different rules."
                        )
                    else:
                        st.caption(
                            "No reduction here — try a longer or more repetitive passage "
                            "to see lemmatization actually shrink the vocabulary."
                        )

                code_and_output(code_shared_lemma, show_shared_lemma, key="spacy_lesson2_shared_lemma")

                st.markdown(
                    "<div class='lib-section'></div>"
                    "**NLTK vs spaCy here:** every lemma above is a real word — the "
                    "concrete, verifiable payoff of a dictionary/model-based approach over "
                    "the Porter Stemmer's fixed truncation rules.",
                    unsafe_allow_html=True,
                )

            # --- spaCy Lesson 3: POS Tagging ---
            if lesson_idx == 3:
                st.subheader("Part-of-Speech (POS) Tagging")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "spaCy gives you **two** POS layers per token: a coarse universal tag "
                    "(`token.pos_`, e.g. `PROPN`) and a fine-grained Penn-Treebank-style tag "
                    "(`token.tag_`, e.g. `NNP`) — the second is directly comparable to NLTK's output."
                )
                st.info(f"Shared example sentence: *\"{SHARED_TEXT_ENTITIES}\"*")

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = "{SHARED_TEXT_ENTITIES}"
doc = nlp(text)
print([(t.text, t.pos_, t.tag_) for t in doc])'''

                def show_spacy_pos():
                    doc = nlp_spacy(SHARED_TEXT_ENTITIES)
                    rows = [(t.text, t.pos_, t.tag_) for t in doc]
                    st.dataframe(pd.DataFrame(rows, columns=["Token", "Universal POS", "Fine-grained Tag"]), use_container_width=True, hide_index=True, height=250)

                code_and_output(code, show_spacy_pos, key="spacy_lesson3_pos")
                st.markdown(
                    "<div class='lib-section'></div>"
                    "**NLTK vs spaCy here:** compare `Fine-grained Tag` to the "
                    "**NLTK → POS Tagging** output on this same sentence — both use the "
                    "same Penn Treebank tag set (`NNP`, `DT`, `IN`...) and largely agree; "
                    "spaCy just adds the extra universal-tag layer on top.",
                    unsafe_allow_html=True,
                )

            # --- spaCy Lesson 4: NER ---
            if lesson_idx == 4:
                st.subheader("Named Entity Recognition")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "spaCy's NER is a trained statistical model — it recognizes a wider range "
                    "of entity types (money, dates, GPEs) than NLTK's rule-based chunker, and "
                    "is generally more accurate on real-world text."
                )
                st.info(f"Shared example sentence: *\"{SHARED_TEXT_ENTITIES}\"*")

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = "{SHARED_TEXT_ENTITIES}"
doc = nlp(text)
for ent in doc.ents:
    print(ent.label_, "->", ent.text)'''

                def show_spacy_ner():
                    doc = nlp_spacy(SHARED_TEXT_ENTITIES)
                    ents = [(ent.text, ent.label_) for ent in doc.ents]
                    if ents:
                        st.dataframe(pd.DataFrame(ents, columns=["Entity", "Type"]), use_container_width=True, hide_index=True)
                    else:
                        st.info("No entities detected.")

                code_and_output(code, show_spacy_ner, key="spacy_lesson4_ner")

                st.markdown("**Entities highlighted in context**")
                st.caption(
                    "Rendered with `displacy.render(doc, style='ent')` — spaCy's built-in "
                    "visualiser. Seeing entities *in the sentence* makes the misses obvious "
                    "in a way a table doesn't."
                )
                render_entities(nlp_spacy(SHARED_TEXT_ENTITIES), key="viz_lesson_ner")
                st.markdown(
                    "<div class='lib-section'></div>"
                    "**NLTK vs spaCy here — and neither is perfect:** on this exact sentence "
                    "NLTK mislabels \"CFO\" and \"Example Corp\" and \"Bengaluru\" all as the wrong "
                    "types. spaCy correctly tags Bengaluru as `GPE` and additionally catches "
                    "`$2.5 million` as MONEY and both dates as DATE — categories NLTK's label "
                    "set doesn't even include.",
                    unsafe_allow_html=True,
                )
                st.warning(
                    "**But look at what spaCy misses:** it doesn't tag \"Example Corp\" as an "
                    "organisation *at all* — it returns no ORG entity here. That's not a bug we "
                    "introduced; it's the central limitation of model-based NER. spaCy learned "
                    "what companies look like from real-world training text, and the deliberately "
                    "fake placeholder name \"Example Corp\" doesn't resemble it. **Model-based NLP "
                    "degrades on text unlike its training data** — which is exactly why you "
                    "evaluate on your own domain instead of trusting a leaderboard number."
                )

            # --- spaCy Lesson 5: Dependency Parsing ---
            if lesson_idx == 5:
                st.subheader("Dependency Parsing")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "Dependency parsing shows the **grammatical relationship** between "
                    "every pair of words in a sentence (subject, object, modifier...) — "
                    "who did what to whom, structurally."
                )
                with st.expander("Why this matters for NLP"):
                    st.markdown(
                        "NLTK doesn't include an easy, ready-to-use dependency parser out of "
                        "the box (it needs external tools like the Stanford Parser) — this is "
                        "a genuine capability spaCy has that plain NLTK doesn't, not something "
                        "we're choosing to skip in the NLTK track."
                    )
                st.info(f"Shared example sentence (truncated): *\"{SHARED_TEXT_ENTITIES[:70]}...\"*")

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = "{SHARED_TEXT_ENTITIES}"
doc = nlp(text)
for token in doc:
    print(token.text, "--", token.dep_, "-->", token.head.text)'''

                def show_spacy_dep():
                    doc = nlp_spacy(SHARED_TEXT_ENTITIES)
                    rows = [(t.text, t.dep_, t.head.text) for t in doc]
                    st.dataframe(pd.DataFrame(rows, columns=["Token", "Relation", "Head word"]), use_container_width=True, hide_index=True, height=250)
                    st.caption("`ROOT` marks the main verb of the sentence; every other token's `Head word` points toward it.")

                code_and_output(code, show_spacy_dep, key="spacy_lesson5_dep")

                st.markdown("**The parse as a diagram**")
                st.caption(
                    "The table above and this diagram are the *same data*. Grammatical "
                    "structure is a tree, so the arrows make relationships visible that "
                    "rows of text genuinely obscure — follow any arrow back to `ROOT`."
                )
                render_dependency_tree(nlp_spacy(SHARED_TEXT_ENTITIES), key="viz_lesson_dep")

            # --- spaCy Lesson 6: Sentiment (lexicon-based) ---
            # NLTK's track has a Sentiment lesson (VADER); spaCy has none built in.
            # Added here so both tracks stay symmetric for direct comparison —
            # same honest caveat already established in the Quick Tools version.
            if lesson_idx == 6:
                st.subheader("Sentiment (Lexicon-based)")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.caption(
                    "**Note:** spaCy has **no built-in sentiment analysis** — that's a "
                    "genuine, structural difference from NLTK (which ships VADER), not "
                    "something skipped here. This lesson uses spaCy for tokenization and "
                    "lemmatization plus a small illustrative lexicon, so every number is "
                    "verifiable — not a spaCy built-in feature."
                )

                sample_text_sp = st.text_input(
                    "Try your own sentence:",
                    "The customer service team really impressed me, though the checkout page was confusing.",
                    key="spacy_sentiment_input",
                )

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = "{sample_text_sp}"
doc = nlp(text)

positive_words = {{"love", "great", "excellent", "clean", "easy", "amazing", "best", "impressed"}}
negative_words = {{"terrible", "bad", "worst", "hate", "confusing", "boring"}}

pos_count = sum(1 for t in doc if t.lemma_.lower() in positive_words)
neg_count = sum(1 for t in doc if t.lemma_.lower() in negative_words)
print("Positive:", pos_count, "| Negative:", neg_count)'''

                def show_spacy_sentiment_lesson():
                    if not sample_text_sp.strip():
                        st.warning("Enter a sentence above.")
                        return
                    doc = nlp_spacy(sample_text_sp)
                    positive_words = {"love", "great", "excellent", "clean", "easy", "amazing", "best", "impressed"}
                    negative_words = {"terrible", "bad", "worst", "hate", "confusing", "boring"}
                    pos_count = sum(1 for t in doc if t.lemma_.lower() in positive_words)
                    neg_count = sum(1 for t in doc if t.lemma_.lower() in negative_words)
                    label = "Positive " if pos_count > neg_count else "Negative " if neg_count > pos_count else "Neutral "
                    st.metric("Overall", label)
                    c1, c2 = st.columns(2)
                    c1.metric("Positive words", pos_count)
                    c2.metric("Negative words", neg_count)
                    st.caption("spaCy's `.lemma_` reduces words to their base form before matching — \"impressed\" and \"impressing\" would match the same lexicon entry.")

                code_and_output(code, show_spacy_sentiment_lesson, key="spacy_lesson6_sentiment")

            # --- spaCy Lesson 7: Keyword Extraction ---
            if lesson_idx == 7:
                st.subheader("Keyword Extraction")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "Same **Term Frequency** idea as the NLTK track's Keyword Extraction "
                    "lesson, but using spaCy's `.lemma_` (base form) instead of NLTK's raw "
                    "tokens — \"runs\" and \"running\" collapse into one count here."
                )
                st.info(f"Shared example sentence: *\"{SHARED_TEXT_ENTITIES}\"*")

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")
text = "{SHARED_TEXT_ENTITIES}"
doc = nlp(text)

keywords = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
print(Counter(keywords).most_common(10))'''

                def show_spacy_keywords():
                    doc = nlp_spacy(SHARED_TEXT_ENTITIES)
                    keywords = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
                    freq = Counter(keywords).most_common(10)
                    if freq:
                        kw_df = pd.DataFrame(freq, columns=["Keyword (lemma)", "Frequency"])
                        st.dataframe(kw_df, use_container_width=True, hide_index=True)
                        st.bar_chart(kw_df.set_index("Keyword (lemma)"))
                    else:
                        st.warning("No keywords found.")

                code_and_output(code, show_spacy_keywords, key="spacy_lesson7_keywords")

                st.caption(
                    "Try the **Quick Tools → Keyword Extraction** tool to run this on "
                    "your own text, with NLTK and spaCy shown side by side."
                )

            # --- spaCy Lesson 8: Word Cloud ---
            if lesson_idx == 8:
                st.subheader("Word Cloud")
                st.markdown("<span class='badge badge-spacy'>spaCy</span>", unsafe_allow_html=True)
                st.markdown(
                    "A visualization of the previous lesson's lemma frequencies — same "
                    "numbers, sized by frequency instead of listed in a table."
                )

                code = f'''# Setup: pip install wordcloud matplotlib
import spacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
text = "{SHARED_TEXT_ENTITIES}"
doc = nlp(text)
words = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
freq = dict(Counter(words))

wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()'''

                def show_spacy_wordcloud():
                    doc = nlp_spacy(SHARED_TEXT_ENTITIES)
                    words = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
                    freq = dict(Counter(words))
                    if not freq:
                        st.warning("No words left to plot.")
                        return
                    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate_from_frequencies(freq)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

                code_and_output(code, show_spacy_wordcloud, key="spacy_lesson8_wordcloud")

                if st.button("spaCy track complete", key="spacy_done"):
                    st.success("Great work — you've now seen the same NLP tasks through both a classic (NLTK) and modern (spaCy) lens.")

            lesson_footer(track, lesson_idx, PY_TRACKS[track]["lessons"], f"nav_lesson_{track}")

    # -----------------------------------------------------------------------
    # CLASSIFICATION & CLUSTERING — added after users pointed out these
    # concepts existed only as Quick Tools with no lesson explaining them, and
    # that the Quick Tools' 10 short sample reviews were too small/thin for
    # real insight. This module uses NLTK's inaugural-address corpus (60 real
    # speeches, 1789-2025, genuinely public domain — US government works) so
    # TF-IDF, classification, and clustering all have a real multi-document
    # corpus to work with, and a REAL classification target: the era, computed
    # mechanically from the year in each filename, not an invented label.
    # -----------------------------------------------------------------------
    elif mode == "Guided Learning" and track == "Classification & Clustering":
        st.markdown(
            "<div class='module-head'><span class='badge badge-amber'>Module 5 · Machine Learning</span>"
            "<h2>Classification &amp; Clustering</h2>"
            "<p>From counting words to predicting labels and finding groups — TF, TF-IDF, "
            "Naive Bayes, and K-Means, each demonstrated on a real 60-document corpus "
            "instead of a toy sentence.</p></div>",
            unsafe_allow_html=True,
        )

        # --- Lesson 0: Term Frequency (TF) ---
        if lesson_idx == 0:
            st.subheader("Term Frequency (TF)")
            st.markdown(
                "<span class='badge badge-amber'>Feature Extraction</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "**Term Frequency** is the simplest way to turn text into numbers: "
                "just count how often each word appears. This is exactly what the "
                "**Keyword Extraction** Quick Tool already does — TF is the formal name "
                "for that count. Everything in this module builds on it."
            )

            docs_result = render_document_boxes(key_prefix="cc_docbox")

            if not docs_result:
                st.info("Fill in the documents above (or keep the defaults) and press Analyze to see Term Frequency per document.")
            else:
                code = '''from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text.lower())
words = [t for t in tokens if t.isalpha() and t not in stop_words]

tf = Counter(words)
print(tf.most_common(10))'''

                def show_tf():
                    stop_words = set(stopwords.words("english"))
                    pick = st.selectbox(
                        "Show Term Frequency for:", [label for label, _ in docs_result],
                        key="cc_tf_pick",
                    )
                    pick_text = dict(docs_result)[pick]
                    tokens = word_tokenize(pick_text.lower())
                    words = [t for t in tokens if t.isalpha() and t not in stop_words]
                    tf = Counter(words).most_common(10)
                    if tf:
                        tf_df = pd.DataFrame(tf, columns=["Term", "Term Frequency (count)"])
                        st.dataframe(tf_df, use_container_width=True, hide_index=True)
                        st.bar_chart(tf_df.set_index("Term"))
                        top_term, top_count = tf[0]
                        st.caption(
                            f"In plain terms: the word **\"{top_term}\"** appears "
                            f"**{top_count} time(s)** in this document, after lowercasing, "
                            "removing punctuation, and dropping common stopwords (\"the\", "
                            "\"and\", etc.) — more than any other word, so it tops the list. "
                            "That's all TF is: a count, nothing more."
                        )
                    else:
                        st.warning("No words left after removing stopwords — try a longer document.")

                code_and_output(code, show_tf, key="cc_tf")

                st.info(
                    "**The limitation TF-IDF fixes:** words like \"the\" or \"movie\" "
                    "might top this list for *every* document in your set — high TF doesn't "
                    "mean *distinctive*. The next lesson shows the fix, using these same "
                    f"{len(docs_result)} documents."
                )

        # --- Lesson 1: TF-IDF (needs a corpus) ---
        elif lesson_idx == 1:
            st.subheader("TF-IDF — Term Frequency × Inverse Document Frequency")
            st.markdown(
                "<span class='badge badge-amber'>Feature Extraction</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "TF-IDF downweights words that are common **across every document** and "
                "upweights words that are frequent in *this* document but rare elsewhere. "
                "It genuinely needs a corpus of multiple documents — computed on a single "
                "document, the \"IDF\" half is constant and TF-IDF collapses to plain TF."
            )

            docs_result = render_document_boxes(key_prefix="cc_docbox")

            if not docs_result:
                st.info("Fill in the documents above (or keep the defaults) and press Analyze to see TF-IDF across your set.")
            elif len(docs_result) < 2:
                st.warning(
                    "TF-IDF needs **at least 2 documents** to mean anything — only "
                    f"{len(docs_result)} document has text. Fill in another box and re-Analyze."
                )
            else:
                labels = [label for label, _ in docs_result]
                pick = st.selectbox("Show TF-IDF for:", labels, key="cc_tfidf_pick")
                pick_idx = labels.index(pick)

                code = '''from sklearn.feature_extraction.text import TfidfVectorizer

# `documents` = your list of document texts
vec = TfidfVectorizer(stop_words="english", max_df=0.8)
matrix = vec.fit_transform(documents)

terms = vec.get_feature_names_out()
row = matrix[pick_idx].toarray().ravel()
top10 = sorted(zip(terms, row), key=lambda x: -x[1])[:10]
print(top10)'''

                def show_tfidf():
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    texts = [t for _, t in docs_result]
                    vec = TfidfVectorizer(stop_words="english", max_df=0.9)
                    matrix = vec.fit_transform(texts)
                    terms = vec.get_feature_names_out()
                    row = matrix[pick_idx].toarray().ravel()
                    top_idx = row.argsort()[::-1][:10]
                    top = [(terms[i], round(float(row[i]), 3)) for i in top_idx if row[i] > 0]
                    if top:
                        df = pd.DataFrame(top, columns=["Term", "TF-IDF weight"])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.bar_chart(df.set_index("Term"))
                        top_term, top_weight = top[0]
                        st.caption(
                            f"In plain terms: **\"{top_term}\"** has the highest TF-IDF "
                            f"weight ({top_weight}) in **{pick}** — meaning it appears "
                            "often *here* but rarely or never in your other documents. A "
                            "word that's frequent in every document (like \"movie\" across "
                            "10 movie reviews) gets pushed *down* by TF-IDF even if its raw "
                            "count (TF) was high — that's the down-weighting in action. "
                            "Weights are unitless (not a percentage or a count) — only their "
                            "*relative order* matters."
                        )
                    else:
                        st.info(
                            "Every word in this document either doesn't appear elsewhere "
                            "in the vocabulary or was filtered — try more varied documents."
                        )

                code_and_output(code, show_tfidf, key="cc_tfidf")

        # --- Lesson 2: Naive Bayes Classification ---
        elif lesson_idx == 2:
            st.subheader("Naive Bayes Classification")
            st.markdown(
                "<span class='badge badge-amber'>Classification</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "**Classification** predicts a label for new examples, learned from past "
                "examples with known labels. That idea has nothing to do with text — it "
                "works on plain numbers first. Part 1 below classifies numbers; Part 2 "
                "applies the exact same idea to text, where the only extra step is turning "
                "words into numbers first (TF-IDF, from the previous lesson)."
            )

            st.markdown("#### Part 1 — Classification on plain numbers")
            st.caption(
                "Synthetic teaching data — 80 fictional students' hours studied and "
                "practice tests taken, predicting pass/fail. Not real student records: "
                "generated from a simple rule (more study + more practice raises pass "
                "chance) plus random noise, so outcomes aren't a clean straight line — "
                "closer to how messy real data actually looks."
            )

            import numpy as np
            _rng_study = np.random.RandomState(7)
            _n_students = 80
            _hours = _rng_study.uniform(0, 10, _n_students)
            _tests = _rng_study.uniform(0, 5, _n_students)
            _score = 0.55 * _hours + 1.1 * _tests + _rng_study.normal(0, 1.8, _n_students)
            _result = np.where(_score > 6.5, "Pass", "Fail")
            study_data = pd.DataFrame({
                "hours_studied": _hours.round(1),
                "practice_tests": _tests.round(1),
                "result": _result,
            })
            with st.expander(f"All {len(study_data)} rows of training data"):
                st.dataframe(study_data, use_container_width=True, hide_index=True)
            st.caption(
                f"Class balance: {(study_data['result'] == 'Pass').sum()} Pass, "
                f"{(study_data['result'] == 'Fail').sum()} Fail — genuinely mixed, not "
                "cherry-picked to look clean."
            )

            code_numeric = '''from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Numeric features -> use GaussianNB, not MultinomialNB (that's for counts/text, Part 2)
X = data[["hours_studied", "practice_tests"]]
y = data["result"]

clf = GaussianNB()
clf.fit(X, y)

new_student = [[5, 1]]  # 5 hours studied, 1 practice test
print(clf.predict(new_student))
print(clf.predict_proba(new_student))'''

            def show_numeric_classification():
                from sklearn.naive_bayes import GaussianNB
                X = study_data[["hours_studied", "practice_tests"]]
                y = study_data["result"]
                clf = GaussianNB()
                clf.fit(X, y)

                c1, c2 = st.columns(2)
                new_hours = c1.slider("New student: hours studied", 0, 10, 5, key="cc_new_hours")
                new_tests = c2.slider("New student: practice tests taken", 0, 5, 1, key="cc_new_tests")
                new_student = pd.DataFrame(
                    {"hours_studied": [new_hours], "practice_tests": [new_tests]}
                )
                pred = clf.predict(new_student)[0]
                proba = clf.predict_proba(new_student)[0]

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = study_data["result"].map({"Pass": "tab:green", "Fail": "tab:red"})
                ax.scatter(study_data["hours_studied"], study_data["practice_tests"], c=colors, s=100, edgecolors="black", label="Training data")
                ax.scatter([new_hours], [new_tests], c="blue", s=200, marker="*", edgecolors="black", label="New student")
                ax.set_xlabel("Hours studied")
                ax.set_ylabel("Practice tests taken")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                st.metric("Predicted result", pred)
                proba_dict = dict(zip(clf.classes_, proba.round(3)))
                pred_conf = proba_dict[pred]
                st.caption(
                    f"In plain terms: given **{new_hours} hours studied** and "
                    f"**{new_tests} practice tests**, the model predicts **{pred}** with "
                    f"**{pred_conf:.0%} confidence** (full breakdown: {proba_dict}). This "
                    "confidence comes from how close the blue star sits to the green "
                    "(Pass) vs. red (Fail) points above — Naive Bayes fits a probability "
                    "distribution to each class and compares how likely the new point is "
                    "under each one. Move the sliders toward the edge between the two "
                    "colors and watch the confidence drop toward 50%."
                )

            code_and_output(code_numeric, show_numeric_classification, key="cc_numeric_classification")

            st.markdown("<div class='lib-section'></div>", unsafe_allow_html=True)
            st.markdown("#### Part 2 — The same idea, applied to text")
            st.markdown(
                "**Classification** predicts a label for new text, learned from labeled "
                "examples. By default, this classifies all **60 inaugural addresses** by "
                "**era** (19th / 20th / 21st century) — a real label taken mechanically "
                "from each speech's year, split into genuine train and test sets. Or upload "
                "your own labeled CSV to run the exact same pipeline on your own data."
            )

            cc_clf_source = st.radio(
                "Text corpus",
                [
                    "60 US Inaugural Addresses, by era (built-in)",
                    "3,300+ passages, by author (built-in, bigger)",
                    "Upload your own labeled CSV",
                ],
                key="cc_clf_source_choice",
                horizontal=True,
            )

            if cc_clf_source.startswith("60 US Inaugural"):
                docs = load_inaugural_corpus()
                texts = [d["text"] for d in docs]
                eras = [d["era"] for d in docs]
                era_counts = pd.Series(eras).value_counts()
                st.caption(
                    "Class sizes: " + ", ".join(f"**{k}**: {v}" for k, v in era_counts.items())
                    + " — the 21st-century class is small because NLTK's corpus only goes "
                    "up to the most recent inauguration it was built with."
                )
            elif cc_clf_source.startswith("3,300+ passages"):
                docs = load_author_passages_corpus()
                texts = [d["text"] for d in docs]
                eras = [d["author"] for d in docs]
                era_counts = pd.Series(eras).value_counts()
                st.caption(
                    "Each row is a ~10-sentence passage from one of 9 public-domain books "
                    "by 3 authors (Austen, Chesterton, Shakespeare) — genuinely more rows "
                    "than the inaugural corpus, so the train/test split has much more to "
                    "work with. Label = which book NLTK filed the passage under, a "
                    "mechanical fact, not a judgment call. Class sizes: "
                    + ", ".join(f"**{k}**: {v}" for k, v in era_counts.items())
                )
            else:
                st.caption(
                    "Upload a CSV with **one text column and one label column**. Classification "
                    "needs real labels to learn from and check accuracy against — a plain "
                    "PDF/text upload has no labels, so it can't be used to train or honestly "
                    "evaluate a classifier (that's exactly what Clustering, below, is for)."
                )
                cc_clf_csv = st.file_uploader(
                    "CSV file", type=["csv"], key="cc_clf_csv_upload",
                    help="Nothing is stored — processed in memory only.",
                )
                texts, eras = [], []
                if cc_clf_csv is not None:
                    try:
                        df_cc = pd.read_csv(cc_clf_csv)
                        if df_cc.empty:
                            st.warning("That CSV appears to be empty.")
                        else:
                            all_cols = df_cc.columns.tolist()
                            c1, c2 = st.columns(2)
                            text_col = c1.selectbox("Text column", all_cols, key="cc_clf_text_col")
                            label_col = c2.selectbox(
                                "Label column", all_cols,
                                index=min(1, len(all_cols) - 1), key="cc_clf_label_col",
                            )
                            clean = df_cc[[text_col, label_col]].dropna()
                            texts = clean[text_col].astype(str).tolist()
                            eras = clean[label_col].astype(str).tolist()
                            class_counts = pd.Series(eras).value_counts()
                            if len(texts) < 10 or len(class_counts) < 2 or class_counts.min() < 2:
                                st.warning(
                                    f"Found {len(texts)} labeled row(s) across {len(class_counts)} "
                                    "class(es). Need at least ~10 rows across 2+ classes, with at "
                                    "least 2 examples per class, for a train/test split to be "
                                    "meaningful."
                                )
                                texts, eras = [], []
                            else:
                                st.success(
                                    f"Loaded **{len(texts)} labeled rows**, "
                                    f"**{len(class_counts)} classes**: "
                                    + ", ".join(f"{k} ({v})" for k, v in class_counts.items())
                                )
                    except Exception as e:
                        st.error(f"Couldn't read that CSV: {e}")

            if len(texts) < 4 or len(set(eras)) < 2:
                st.info("Load a labeled corpus above to run classification.")
            else:
                test_size = st.slider("Test set fraction", 0.15, 0.4, 0.25, 0.05, key="cc_test_size")

                code = f'''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# X = document texts, y = labels
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size={test_size}, random_state=0, stratify=labels
)

vec = TfidfVectorizer(stop_words="english", max_df=0.8)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)
preds = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))'''

                def show_classification():
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.naive_bayes import MultinomialNB
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score

                    X_train, X_test, y_train, y_test = train_test_split(
                        texts, eras, test_size=test_size, random_state=0, stratify=eras
                    )
                    vec = TfidfVectorizer(stop_words="english", max_df=0.8)
                    X_train_vec = vec.fit_transform(X_train)
                    X_test_vec = vec.transform(X_test)
                    clf = MultinomialNB()
                    clf.fit(X_train_vec, y_train)
                    preds = clf.predict(X_test_vec)
                    acc = accuracy_score(y_test, preds)

                    st.metric("Test accuracy", f"{acc:.0%}", help=f"Computed on {len(y_test)} held-out documents")
                    result_df = pd.DataFrame({"Actual label": y_test, "Predicted label": preds})
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.caption(
                        f"This is a **real, honestly computed** number on held-out data — "
                        "not a claimed or expected result. Re-run with a different test "
                        "fraction above and watch it change; with a small corpus, accuracy "
                        "is noisy from run to run."
                    )

                code_and_output(code, show_classification, key="cc_classification")

        # --- Lesson 3: K-Means Clustering ---
        elif lesson_idx == 3:
            st.subheader("K-Means Clustering")
            st.markdown(
                "<span class='badge badge-amber'>Clustering</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "**Clustering** groups things by similarity **without using labels** — "
                "the opposite setup from classification. Like Classification, this idea "
                "has nothing to do with text at first. Part 1 clusters plain 2D points; "
                "Part 2 applies the same K-Means algorithm to text, where the only extra "
                "step is TF-IDF turning words into numbers first."
            )

            st.markdown("#### Part 1 — Clustering plain numbers")
            st.caption(
                "Synthetic 2D points — 90 fictional points across 3 groups, deliberately "
                "spread out (not tiny/tidy toy clusters) so the shapes are visually "
                "obvious but the groups still overlap a little at the edges, like real data."
            )

            import numpy as np
            rng = np.random.RandomState(42)
            group_a = rng.normal(loc=[2, 2], scale=1.0, size=(30, 2))
            group_b = rng.normal(loc=[8, 3], scale=1.0, size=(30, 2))
            group_c = rng.normal(loc=[5, 8], scale=1.0, size=(30, 2))
            points = np.vstack([group_a, group_b, group_c])
            points_df = pd.DataFrame(points, columns=["x", "y"])

            st.markdown("##### How do you choose k? The elbow method")
            st.markdown(
                "K-Means needs to be **told** how many clusters to look for — it won't "
                "discover that number on its own. The **elbow method** helps pick a "
                "reasonable k: run K-Means for a range of k values, plot each one's "
                "**inertia** (total squared distance from every point to its own cluster "
                "center — lower is tighter, more compact clusters), and look for the point "
                "where adding more clusters stops helping much. That bend in the curve "
                "looks like an elbow, hence the name."
            )

            from sklearn.cluster import KMeans as _KMeansElbow
            elbow_ks = list(range(1, 8))
            inertias = []
            for _k in elbow_ks:
                _km = _KMeansElbow(n_clusters=_k, n_init=10, random_state=0)
                _km.fit(points_df[["x", "y"]])
                inertias.append(_km.inertia_)

            fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
            ax_elbow.plot(elbow_ks, inertias, marker="o")
            ax_elbow.set_xlabel("k (number of clusters)")
            ax_elbow.set_ylabel("Inertia (lower = tighter clusters)")
            ax_elbow.set_title("Elbow plot — where does adding more k stop helping?")
            st.pyplot(fig_elbow)
            plt.close(fig_elbow)
            st.caption(
                "Inertia always decreases as k increases (more clusters can only fit "
                "the data at least as well) — the question is where it stops decreasing "
                "*a lot*. Here the curve should bend noticeably around **k=3**, which "
                "matches how this synthetic data was actually generated (3 groups) — a "
                "genuine check, not a number chosen to make the demo look right."
            )

            k_numeric = st.slider("Number of clusters (k)", 2, 6, 3, key="cc_numeric_k")

            code_numeric_cl = f'''from sklearn.cluster import KMeans

# Elbow method: try a range of k, plot inertia, look for the bend
inertias = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(points)
    inertias.append(km.inertia_)
# plot `inertias` against range(1, 8) and look for where it stops dropping fast

# Then fit with the chosen k
km = KMeans(n_clusters={k_numeric}, n_init=10, random_state=0)
cluster_labels = km.fit_predict(points)
print(cluster_labels)'''

            def show_numeric_clustering():
                from sklearn.cluster import KMeans
                km_num = KMeans(n_clusters=k_numeric, n_init=10, random_state=0)
                labels_num = km_num.fit_predict(points_df[["x", "y"]])

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(points_df["x"], points_df["y"], c=labels_num, cmap="tab10", s=110, edgecolors="black")
                centers = km_num.cluster_centers_
                ax.scatter(centers[:, 0], centers[:, 1], c="black", s=200, marker="X", label="Cluster centers")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                sizes = pd.Series(labels_num).value_counts().sort_index()
                centers_txt = ", ".join(
                    f"Cluster {i}: center at ({c[0]:.1f}, {c[1]:.1f}), {sizes.get(i, 0)} points"
                    for i, c in enumerate(centers)
                )
                st.caption(
                    f"In plain terms: K-Means picked {k_numeric} center points (the black "
                    f"X's) and assigned every point to whichever center is closest — "
                    f"{centers_txt}. With no labels at all, K-Means found these groups "
                    "purely from how close points are to each other. This is exactly what "
                    "happens to TF-IDF vectors in Part 2 — just in many more than 2 "
                    "dimensions, where 'closest' can't be drawn on a page."
                )

            code_and_output(code_numeric_cl, show_numeric_clustering, key="cc_numeric_clustering")

            st.markdown("<div class='lib-section'></div>", unsafe_allow_html=True)
            st.markdown("#### Part 2 — The same idea, applied to text")
            st.markdown(
                "By default, this clusters the same 60 inaugural addresses and then checks, "
                "honestly, how well the *unsupervised* clusters line up with the *real* eras "
                "from the previous lesson. Clustering needs no labels at all, so you can also "
                "point it at your own documents — built-in corpora or an upload, same as "
                "the Preprocessing Pipeline's corpus picker."
            )

            cc_clust_source = st.radio(
                "Text corpus",
                [
                    "60 US Inaugural Addresses (built-in, has real era labels to check against)",
                    "Your own documents (built-in corpus or upload)",
                ],
                key="cc_clust_source_choice",
                horizontal=True,
            )

            has_eras = cc_clust_source.startswith("60 US Inaugural")
            if has_eras:
                docs = load_inaugural_corpus()
                texts = [d["text"] for d in docs]
                eras = [d["era"] for d in docs]
                labels_disp = [d["label"] for d in docs]
            else:
                picked = render_multi_doc_picker(key_prefix="cc_clust_tool")
                texts = [d["text"] for d in picked]
                labels_disp = [d["label"] for d in picked]
                eras = None

            if len(texts) < 2:
                st.info("Need at least 2 documents to cluster. Pick a source above with 2+ documents.")
            else:
                max_k = min(6, len(texts))

                with st.expander("Not sure what k to pick? Check the elbow plot"):
                    st.caption(
                        "Same idea as Part 1's elbow method, just run on TF-IDF vectors "
                        "instead of 2D points — this takes a few seconds since it fits "
                        "K-Means several times."
                    )
                    if st.button("Compute elbow plot", key="cc_text_elbow_btn"):
                        from sklearn.feature_extraction.text import TfidfVectorizer as _TV
                        from sklearn.cluster import KMeans as _KM
                        _vec_elbow = _TV(stop_words="english", max_df=0.8)
                        _X_elbow = _vec_elbow.fit_transform(texts)
                        _elbow_ks = list(range(1, max(2, max_k) + 1))
                        _inertias = []
                        for _k in _elbow_ks:
                            _km = _KM(n_clusters=_k, n_init=10, random_state=0)
                            _km.fit(_X_elbow)
                            _inertias.append(_km.inertia_)
                        _fig, _ax = plt.subplots(figsize=(6, 4))
                        _ax.plot(_elbow_ks, _inertias, marker="o")
                        _ax.set_xlabel("k (number of clusters)")
                        _ax.set_ylabel("Inertia (lower = tighter clusters)")
                        _ax.set_title("Elbow plot on your TF-IDF vectors")
                        st.pyplot(_fig)
                        plt.close(_fig)
                        st.caption(
                            "Look for where the curve stops dropping steeply. Real text "
                            "data rarely has as clean a bend as the synthetic 2D points in "
                            "Part 1 — that's honest; text clustering is genuinely fuzzier."
                        )

                k = st.slider(
                    "Number of clusters (k)", min_value=2, max_value=max(2, max_k),
                    value=min(3, max_k), key="cc_cluster_k",
                )

                code = f'''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

vec = TfidfVectorizer(stop_words="english", max_df=0.8)
X = vec.fit_transform(documents)                 # N documents -> TF-IDF vectors
km = KMeans(n_clusters={k}, n_init=10, random_state=0)
cluster_labels = km.fit_predict(X)

coords = PCA(n_components=2).fit_transform(X.toarray())  # for plotting only'''

                def show_clustering_cc():
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.cluster import KMeans
                    from sklearn.decomposition import PCA

                    vec = TfidfVectorizer(stop_words="english", max_df=0.8)
                    X = vec.fit_transform(texts)
                    km = KMeans(n_clusters=k, n_init=10, random_state=0)
                    cluster_labels = km.fit_predict(X)
                    coords = PCA(n_components=2, random_state=0).fit_transform(X.toarray())

                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="tab10", s=90, edgecolors="black")
                    ax.set_xlabel("PCA component 1")
                    ax.set_ylabel("PCA component 2")
                    ax.set_title(f"K-Means clusters of {len(texts)} document(s) (k={k})")
                    st.pyplot(fig)
                    plt.close(fig)

                    if has_eras:
                        st.markdown("**Does clustering (unsupervised) recover era (the real label)?**")
                        crosstab = pd.crosstab(
                            pd.Series(cluster_labels, name="Cluster"),
                            pd.Series(eras, name="Actual era"),
                        )
                        st.dataframe(crosstab, use_container_width=True)
                        st.caption(
                            "Read this like a confusion matrix. If a cluster's row is concentrated "
                            "in one era column, K-Means found a grouping that lines up with real "
                            "history using **only word similarity, no labels**. If it's spread across "
                            "columns, clustering found some *other* structure (e.g. topic or length) "
                            "that doesn't match era — an honest, possible outcome, not a failure to hide."
                        )
                    else:
                        result_df = pd.DataFrame({
                            "Document": [
                                l if len(l) <= 60 else l[:60] + "…" for l in labels_disp
                            ],
                            "Cluster": cluster_labels,
                        })
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        st.caption(
                            "No ground-truth labels here, so there's nothing to check clusters "
                            "against — this just shows which of your documents K-Means grouped "
                            "together based on word similarity alone."
                        )

                code_and_output(code, show_clustering_cc, key="cc_clustering")

        lesson_footer(track, lesson_idx, PY_TRACKS[track]["lessons"], f"nav_lesson_{track}")

    # -----------------------------------------------------------------------
    # QUICK TOOLS (Python)
    # -----------------------------------------------------------------------
    elif mode == "Quick Tools":
        st.header("Quick Tools")
        st.caption(
            "**Load your text once — then run every tool on it.** Your text stays loaded "
            "as you switch between tools below."
        )

        # ==================== CORPUS LOADER ====================
        # Inspired by Orange's Corpus widget: text is loaded ONCE into session
        # state and every downstream tool reads from it, instead of each tool
        # asking for its own input.
        with st.expander("**Your text** — paste, pick a sample, or upload a file", expanded=True):
            input_choice = st.radio(
                "Where should the text come from?",
                [
                    "Paste my own text",
                    "Use a sample review",
                    "Public-domain classic",
                    "Upload a file",
                ],
                horizontal=True,
                key="corpus_source",
            )

            if input_choice == "Use a sample review":
                chosen = st.selectbox("Pick a sample:", SAMPLE_REVIEWS["review"].tolist(), key="corpus_sample")
                st.session_state.corpus_text = chosen

            elif input_choice == "Public-domain classic":
                st.caption(
                    "Every text here is **public domain** — free of copyright restrictions, "
                    "so you can analyse, copy and reuse the output without limitation. "
                    "Loaded via NLTK's corpora; see CONTENT-LICENSES.md in the repo for provenance."
                )
                pd_choice = st.selectbox(
                    "Pick a public-domain text:",
                    list(PUBLIC_DOMAIN_CORPORA.keys()),
                    key="corpus_pd",
                )
                pd_corpus, pd_file = PUBLIC_DOMAIN_CORPORA[pd_choice]
                try:
                    pd_text = load_public_domain_text(pd_corpus, pd_file)
                    st.session_state.corpus_text = pd_text
                    st.success(
                        f"Loaded **{pd_choice}** — first {len(pd_text):,} characters "
                        "(truncated so the tools stay responsive)."
                    )
                except Exception as e:
                    st.error(f"Couldn't load that text: {e}")

            elif input_choice == "Upload a file":
                uploaded = st.file_uploader(
                    "Upload a .txt, .csv or .pdf file",
                    type=["txt", "csv", "pdf"],
                    key="corpus_upload",
                    help="Max ~200MB. Nothing is stored — the file is processed in memory only.",
                )
                if uploaded is not None:
                    try:
                        if uploaded.name.lower().endswith(".pdf"):
                            from pypdf import PdfReader
                            import io as _io
                            reader = PdfReader(_io.BytesIO(uploaded.getvalue()))
                            extracted = "\n".join(p.extract_text() or "" for p in reader.pages)
                            if not extracted.strip():
                                st.error(
                                    "**No text could be extracted.** This is almost certainly a "
                                    "scanned PDF — images of text rather than real text. "
                                    "Extracting it needs OCR, which this app doesn't include."
                                )
                            else:
                                st.session_state.corpus_text = extracted
                                st.success(
                                    f"Extracted **{len(extracted):,} characters** from "
                                    f"{len(reader.pages)} page(s) of **{uploaded.name}**."
                                )
                                st.caption(
                                    "PDF extraction is approximate — a PDF stores the position "
                                    "of marks on a page, not clean text. Check the preview below."
                                )
                        elif uploaded.name.lower().endswith(".csv"):
                            df_up = pd.read_csv(uploaded)
                            if df_up.empty:
                                st.warning("That CSV appears to be empty.")
                            else:
                                text_cols = [c for c in df_up.columns if df_up[c].dtype == object]
                                if not text_cols:
                                    st.warning("No text columns found in that CSV. Pick a file with at least one text column.")
                                else:
                                    col = st.selectbox("Which column holds the text?", text_cols, key="corpus_csv_col")
                                    n_rows = st.slider(
                                        "How many rows to combine?", 1,
                                        min(50, len(df_up)), min(10, len(df_up)),
                                        key="corpus_csv_rows",
                                    )
                                    st.session_state.corpus_text = "\n".join(
                                        df_up[col].dropna().astype(str).head(n_rows).tolist()
                                    )
                                    st.success(f"Loaded {n_rows} row(s) from column **{col}**.")
                        else:
                            raw = uploaded.getvalue()
                            try:
                                st.session_state.corpus_text = raw.decode("utf-8")
                            except UnicodeDecodeError:
                                # Real-world files aren't always UTF-8. Fall back rather
                                # than crash, but tell the user what happened.
                                st.session_state.corpus_text = raw.decode("latin-1")
                                st.warning(
                                    "This file isn't valid UTF-8, so it was read as Latin-1 instead. "
                                    "Some characters may look wrong — that's a real encoding issue, "
                                    "not an app bug."
                                )
                            st.success(f"Loaded **{uploaded.name}** ({len(st.session_state.corpus_text):,} characters).")
                    except Exception as e:
                        st.error(f"Couldn't read that file: {e}")

            else:
                st.session_state.corpus_text = st.text_area(
                    "Paste text here:",
                    st.session_state.get(
                        "corpus_text",
                        "I absolutely loved the new design of this app, it's clean and easy to use!",
                    ),
                    height=120,
                    key="corpus_paste",
                )

        text = st.session_state.get("corpus_text", "")

        if text.strip():
            n_chars = len(text)
            n_words = len(text.split())
            c1, c2, c3 = st.columns(3)
            c1.metric("Characters", f"{n_chars:,}")
            c2.metric("Words (rough)", f"{n_words:,}")
            c3.metric("Sentences", f"{len(sent_tokenize(text)):,}")
            with st.expander("Preview loaded text"):
                st.text(text[:2000] + ("\n\n...(truncated preview)" if len(text) > 2000 else ""))
        else:
            st.info("Load some text above to get started.")

        st.markdown("---")

        tool = st.selectbox("Choose a tool", QUICK_TOOLS)

        st.markdown("---")

        # ===================== PREPROCESSING PIPELINE =====================
        # Modelled on Orange's "Preprocess Text" widget: the point is to make
        # preprocessing VISIBLE and ordered, showing what each stage removes,
        # rather than hiding it inside one opaque clean_text() call.
        if tool == "Preprocessing Pipeline":
            st.markdown(
                "<span class='badge badge-amber'>Foundational</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Preprocessing is applied **in order**, each stage feeding the next. "
                "Toggle stages on and off and watch how many tokens survive each step — "
                "this is the part of NLP most often treated as a black box."
            )

            p1, p2 = st.columns(2)
            with p1:
                do_lower = st.checkbox("1. Lowercase", value=True, key="pp_lower")
                do_punct = st.checkbox("2. Remove punctuation", value=True, key="pp_punct")
                do_stop = st.checkbox("3. Remove stopwords", value=True, key="pp_stop")
            with p2:
                norm_choice = st.radio(
                    "4. Normalization",
                    ["None", "Stemming (NLTK)", "Lemmatization (spaCy)"],
                    key="pp_norm",
                )

            if text.strip():
                stages = []
                working = text

                if do_lower:
                    working = working.lower()
                tokens = word_tokenize(working)
                stages.append(("Tokenized" + (" + lowercased" if do_lower else ""), len(tokens), list(tokens)))

                if do_punct:
                    tokens = [t for t in tokens if t.isalpha()]
                    stages.append(("After removing punctuation/non-alphabetic", len(tokens), list(tokens)))

                if do_stop:
                    sw = set(stopwords.words("english"))
                    tokens = [t for t in tokens if t.lower() not in sw]
                    stages.append(("After removing stopwords", len(tokens), list(tokens)))

                if norm_choice == "Stemming (NLTK)":
                    ps = PorterStemmer()
                    tokens = [ps.stem(t) for t in tokens]
                    stages.append(("After stemming", len(tokens), list(tokens)))
                elif norm_choice == "Lemmatization (spaCy)":
                    nlp_pp, ok_pp = get_spacy_model()
                    if ok_pp:
                        doc_pp = nlp_pp(" ".join(tokens))
                        tokens = [t.lemma_ for t in doc_pp]
                        stages.append(("After lemmatization", len(tokens), list(tokens)))
                    else:
                        st.error("spaCy couldn't load — skipping the lemmatization stage.")

                st.subheader("Pipeline stages")
                stage_df = pd.DataFrame(
                    [(i + 1, name, count) for i, (name, count, _) in enumerate(stages)],
                    columns=["Step", "Stage", "Tokens remaining"],
                )
                st.dataframe(stage_df, use_container_width=True, hide_index=True)
                st.bar_chart(stage_df.set_index("Stage")["Tokens remaining"])

                first_count = stages[0][1]
                last_count = stages[-1][1]
                if first_count:
                    pct = 100 * (first_count - last_count) / first_count
                    st.caption(
                        f"Started with **{first_count}** tokens, ended with **{last_count}** — "
                        f"preprocessing removed **{pct:.0f}%** of them."
                    )

                with st.expander("See the actual tokens at each stage"):
                    for name, count, toks in stages:
                        st.markdown(f"**{name}** ({count} tokens)")
                        st.write(toks[:60] + (["...(truncated)"] if len(toks) > 60 else []))

                pipeline_lines = ["import nltk", "from nltk.tokenize import word_tokenize"]
                if do_stop:
                    pipeline_lines.append("from nltk.corpus import stopwords")
                if norm_choice == "Stemming (NLTK)":
                    pipeline_lines.append("from nltk.stem import PorterStemmer")
                if norm_choice == "Lemmatization (spaCy)":
                    pipeline_lines.append("import spacy")

                body = ["", "text = your_text_here", ""]
                if do_lower:
                    body.append("text = text.lower()")
                body.append("tokens = word_tokenize(text)")
                if do_punct:
                    body.append("tokens = [t for t in tokens if t.isalpha()]")
                if do_stop:
                    body.append('sw = set(stopwords.words("english"))')
                    body.append("tokens = [t for t in tokens if t.lower() not in sw]")
                if norm_choice == "Stemming (NLTK)":
                    body.append("ps = PorterStemmer()")
                    body.append("tokens = [ps.stem(t) for t in tokens]")
                if norm_choice == "Lemmatization (spaCy)":
                    body.append('nlp = spacy.load("en_core_web_sm")')
                    body.append('tokens = [t.lemma_ for t in nlp(" ".join(tokens))]')
                body.append("print(tokens)")

                pipeline_code = (
                    "# Setup (run once, e.g. in Colab):\n"
                    "# import nltk\n"
                    '# nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("stopwords")\n'
                    + ("# !pip install spacy && !python -m spacy download en_core_web_sm\n"
                       if norm_choice == "Lemmatization (spaCy)" else "")
                    + "\n" + "\n".join(pipeline_lines) + "\n" + "\n".join(body)
                )

                st.subheader("The exact code for the pipeline you just built")
                st.code(pipeline_code, language="python")
                st.download_button(
                    "Download this pipeline",
                    data=pipeline_code,
                    file_name="preprocessing_pipeline.py",
                    mime="text/x-python",
                    key="dl_pipeline",
                )

                st.info(
                    "**Order matters.** Removing stopwords *before* lowercasing misses "
                    "capitalized stopwords like \"The\". Stemming *before* stopword removal "
                    "can turn a word into something the stopword list no longer matches. "
                    "Try reordering mentally and predict what changes."
                )

                # --- Optional bonus stage: TF-IDF ---
                # Deliberately NOT wired into the token-count chain above: TF-IDF's "IDF"
                # half is only meaningful across a CORPUS of documents. Computed on a
                # single document it collapses to something proportional to plain term
                # frequency — a common student confusion this tool makes explicit rather
                # than hiding.
                st.markdown("<div class='lib-section'></div>", unsafe_allow_html=True)
                show_tfidf = st.checkbox(
                    "Bonus: show TF-IDF weighting (scikit-learn)", key="pp_tfidf"
                )
                if show_tfidf:
                    st.caption(
                        "TF-IDF needs **more than one document** to mean anything — IDF "
                        "measures how rare a word is *across documents*. Your text is scored "
                        "here against our 10-review reference corpus, so a word gets a high "
                        "score only if it's frequent in your text **and** uncommon in the "
                        "reference set."
                    )
                    from sklearn.feature_extraction.text import TfidfVectorizer

                    corpus_for_tfidf = SAMPLE_REVIEWS["review"].tolist() + [text]
                    vec = TfidfVectorizer(stop_words="english")
                    tfidf_matrix = vec.fit_transform(corpus_for_tfidf)
                    feature_names = vec.get_feature_names_out()
                    your_row = tfidf_matrix[-1].toarray().ravel()
                    top_idx = your_row.argsort()[::-1][:10]
                    top_terms = [(feature_names[i], round(float(your_row[i]), 3))
                                 for i in top_idx if your_row[i] > 0]

                    if top_terms:
                        tfidf_df = pd.DataFrame(top_terms, columns=["Term", "TF-IDF weight"])
                        st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
                        st.bar_chart(tfidf_df.set_index("Term"))
                    else:
                        st.info(
                            "Every word in your text either doesn't appear in the reference "
                            "corpus's vocabulary or was filtered as a stopword — try a longer, "
                            "more topical piece of text."
                        )

                    tfidf_code = '''# Setup: pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF needs a CORPUS — here, your text plus a small reference corpus
corpus = reference_reviews + [your_text]
vec = TfidfVectorizer(stop_words="english")
matrix = vec.fit_transform(corpus)

# Your text is the last row
your_scores = matrix[-1].toarray().ravel()
terms = vec.get_feature_names_out()
top10 = sorted(zip(terms, your_scores), key=lambda x: -x[1])[:10]
print(top10)'''
                    st.code(tfidf_code, language="python")

        # ===================== SENTIMENT ANALYSIS =====================
        elif tool == "Sentiment Analysis":
            st.caption(
                "New to sentiment analysis? See **Guided Learning → NLTK / spaCy → "
                "Sentiment** for a step-by-step explanation first."
            )
            st.markdown(
                "<span class='badge badge-nltk'>NLTK path</span>",
                unsafe_allow_html=True,
            )
            if text.strip():
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(text)
                compound = scores["compound"]
                label = "Positive " if compound > 0.05 else "Negative " if compound < -0.05 else "Neutral "

                code = f'''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = """{text[:80]}..."""
sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores(text)
print(scores)'''

                def show_sentiment_output_nltk():
                    st.metric("Sentiment", label)
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Positive", f"{scores['pos']:.2f}")
                    col_b.metric("Negative", f"{scores['neg']:.2f}")
                    col_c.metric("Compound", f"{compound:.2f}", help="Overall score from -1 (very negative) to +1 (very positive)")
                    st.caption(
                        "**VADER** (part of NLTK) is a lexicon-based sentiment tool tuned for "
                        "short, informal text like reviews and social media."
                    )

                code_and_output(code, show_sentiment_output_nltk, key="tool_sentiment_nltk")

            st.markdown(
                "<div class='lib-section'></div>"
                "<span class='badge badge-spacy'>spaCy path</span>",
                unsafe_allow_html=True,
            )
            nlp_spacy, spacy_ok = get_spacy_model()
            if not spacy_ok:
                st.error("spaCy couldn't load right now. Refresh and try again in a moment.")
            elif text.strip():
                st.caption(
                    "**Note:** spaCy doesn't ship built-in sentiment analysis. Real spaCy "
                    "projects typically add an extension like `spacytextblob`. This example "
                    "uses spaCy for tokenization/lemmatization plus a small illustrative "
                    "lexicon, so every number here is verifiable — not a spaCy built-in feature."
                )
                doc = nlp_spacy(text)
                positive_words = {"love", "great", "excellent", "clean", "easy", "amazing", "best"}
                negative_words = {"terrible", "bad", "worst", "hate", "confusing", "boring"}
                pos_count = sum(1 for t in doc if t.lemma_.lower() in positive_words)
                neg_count = sum(1 for t in doc if t.lemma_.lower() in negative_words)
                label_spacy = "Positive " if pos_count > neg_count else "Negative " if neg_count > pos_count else "Neutral "

                code = f'''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = """{text[:80]}..."""
doc = nlp(text)

positive_words = {{"love", "great", "excellent", "clean", "easy", "amazing", "best"}}
negative_words = {{"terrible", "bad", "worst", "hate", "confusing", "boring"}}

pos_count = sum(1 for t in doc if t.lemma_.lower() in positive_words)
neg_count = sum(1 for t in doc if t.lemma_.lower() in negative_words)
print("Positive:", pos_count, "| Negative:", neg_count)'''

                def show_sentiment_output_spacy():
                    st.metric("Sentiment", label_spacy)
                    col_a, col_b = st.columns(2)
                    col_a.metric("Positive words", pos_count)
                    col_b.metric("Negative words", neg_count)
                    st.caption(
                        "spaCy's `.lemma_` reduces words to their base form (e.g. "
                        "\"loved\" → \"love\") before matching against the lexicon."
                    )

                code_and_output(code, show_sentiment_output_spacy, key="tool_sentiment_spacy")

        # ===================== KEYWORD EXTRACTION =====================
        elif tool == "Keyword Extraction":
            st.caption(
                "New to keyword extraction? See **Guided Learning → NLTK / spaCy → "
                "Keyword Extraction** for a step-by-step explanation first."
            )
            st.markdown(
                "<span class='badge badge-nltk'>NLTK path</span>",
                unsafe_allow_html=True,
            )
            if text.strip():
                stop_words = set(stopwords.words("english"))
                tokens = word_tokenize(clean_text(text))
                keywords = [t for t in tokens if t.isalpha() and t not in stop_words]
                freq = Counter(keywords).most_common(10)

                code = '''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text.lower())
keywords = [t for t in tokens if t.isalpha() and t not in stop_words]

print(Counter(keywords).most_common(10))'''

                def show_keyword_output_nltk():
                    if freq:
                        kw_df = pd.DataFrame(freq, columns=["Keyword", "Frequency"])
                        st.dataframe(kw_df, use_container_width=True, hide_index=True)
                        st.bar_chart(kw_df.set_index("Keyword"))
                    else:
                        st.warning("No keywords found — try a longer sentence.")

                code_and_output(code, show_keyword_output_nltk, key="tool_keywords_nltk")

            st.markdown(
                "<div class='lib-section'></div>"
                "<span class='badge badge-spacy'>spaCy path</span>",
                unsafe_allow_html=True,
            )
            nlp_spacy, spacy_ok = get_spacy_model()
            if not spacy_ok:
                st.error("spaCy couldn't load right now. Refresh and try again in a moment.")
            elif text.strip():
                doc = nlp_spacy(text)
                keywords_spacy = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
                freq_spacy = Counter(keywords_spacy).most_common(10)

                code = '''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

keywords = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
print(Counter(keywords).most_common(10))'''

                def show_keyword_output_spacy():
                    if freq_spacy:
                        kw_df = pd.DataFrame(freq_spacy, columns=["Keyword (lemma)", "Frequency"])
                        st.dataframe(kw_df, use_container_width=True, hide_index=True)
                        st.bar_chart(kw_df.set_index("Keyword (lemma)"))
                        st.caption(
                            "spaCy returns **lemmas** (base forms), e.g. \"loved\" → \"love\" — "
                            "slightly different from NLTK's raw-token approach."
                        )
                    else:
                        st.warning("No keywords found — try a longer sentence.")

                code_and_output(code, show_keyword_output_spacy, key="tool_keywords_spacy")

        # ===================== NAMED ENTITY RECOGNITION =====================
        elif tool == "Named Entity Recognition":

            st.markdown(
                "<span class='badge badge-nltk'>NLTK path</span>",
                unsafe_allow_html=True,
            )
            if text.strip():
                tokens = word_tokenize(text)
                tagged = pos_tag(tokens)
                tree = ne_chunk(tagged)

                entities = []
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        entity_text = " ".join(word for word, tag in subtree.leaves())
                        entities.append((entity_text, subtree.label()))

                code = '''# Setup (run once, e.g. in Colab):
# import nltk
# for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
#             "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
#             "maxent_ne_chunker_tab", "words"]:
#     nltk.download(pkg)

from nltk import word_tokenize, pos_tag, ne_chunk

tokens = word_tokenize(text)
tagged = pos_tag(tokens)
tree = ne_chunk(tagged)

for subtree in tree:
    if hasattr(subtree, "label"):
        print(subtree.label(), "->", " ".join(w for w, t in subtree.leaves()))'''

                def show_ner_output_nltk():
                    if entities:
                        ent_df = pd.DataFrame(entities, columns=["Entity", "Type"])
                        st.dataframe(ent_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No named entities detected in this text. Try text with names of people, places, or organizations.")
                    st.caption("NLTK's built-in chunker is lightweight — good for teaching, but less accurate than model-based approaches like the spaCy path.")

                code_and_output(code, show_ner_output_nltk, key="tool_ner_nltk")

            st.markdown(
                "<div class='lib-section'></div>"
                "<span class='badge badge-spacy'>spaCy path</span>",
                unsafe_allow_html=True,
            )
            nlp_spacy, spacy_ok = get_spacy_model()
            if not spacy_ok:
                st.error("spaCy couldn't load right now. Refresh and try again in a moment.")
            elif text.strip():
                doc = nlp_spacy(text)
                entities_spacy = [(ent.text, ent.label_) for ent in doc.ents]

                code = '''# Setup (run once, e.g. in Colab):
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)
for ent in doc.ents:
    print(ent.label_, "->", ent.text)'''

                def show_ner_output_spacy():
                    if entities_spacy:
                        ent_df = pd.DataFrame(entities_spacy, columns=["Entity", "Type"])
                        st.dataframe(ent_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No named entities detected in this text. Try text with names of people, places, or organizations.")
                    st.caption("spaCy uses a pretrained statistical model — generally more accurate than NLTK's rule-based chunker, especially on real-world text.")

                code_and_output(code, show_ner_output_spacy, key="tool_ner_spacy")

                st.markdown("**Entities highlighted in your text**")
                render_entities(doc, key="viz_tool_ner")

        # ===================== WORD CLOUD =====================
        elif tool == "Word Cloud":
            st.caption(
                "See **Guided Learning → NLTK / spaCy → Word Cloud** for the same "
                "tool explained as a taught lesson first."
            )
            st.markdown(
                "A word cloud sizes each word by how often it appears — a quick visual gut-check "
                "of what a text is about. It's built on the **same word-frequency counting** as "
                "Keyword Extraction, just rendered differently, using every word rather than the "
                "top 10."
            )

            def render_wordcloud(freq_dict: dict, key: str):
                if not freq_dict:
                    st.warning("No words left to plot — try a longer text or fewer filters.")
                    return
                wc = WordCloud(
                    width=800, height=400, background_color="white", colormap="viridis"
                ).generate_from_frequencies(freq_dict)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)  # free memory — Streamlit Cloud's ceiling is ~1GB total

            st.markdown(
                "<span class='badge badge-nltk'>NLTK path</span>",
                unsafe_allow_html=True,
            )
            if text.strip():
                stop_words = set(stopwords.words("english"))
                tokens_wc = word_tokenize(clean_text(text))
                words_wc = [t for t in tokens_wc if t.isalpha() and t not in stop_words]
                freq_wc_nltk = dict(Counter(words_wc))

                code = '''# Setup (run once, e.g. in Colab):
# import nltk
# nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("stopwords")
# !pip install wordcloud matplotlib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text.lower())
words = [t for t in tokens if t.isalpha() and t not in stop_words]
freq = dict(Counter(words))

wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()'''

                def show_wc_nltk():
                    render_wordcloud(freq_wc_nltk, key="wc_nltk")
                    st.caption(f"Built from **{len(freq_wc_nltk)}** unique words after stopword removal.")

                code_and_output(code, show_wc_nltk, key="tool_wc_nltk")

            st.markdown(
                "<div class='lib-section'></div>"
                "<span class='badge badge-spacy'>spaCy path</span>",
                unsafe_allow_html=True,
            )
            nlp_spacy, spacy_ok = get_spacy_model()
            if not spacy_ok:
                st.error("spaCy couldn't load right now. Refresh and try again in a moment.")
            elif text.strip():
                doc_wc = nlp_spacy(text)
                words_wc_spacy = [t.lemma_.lower() for t in doc_wc if not t.is_stop and t.is_alpha]
                freq_wc_spacy = dict(Counter(words_wc_spacy))

                code = '''# Setup (run once, e.g. in Colab):
# !pip install spacy wordcloud matplotlib
# !python -m spacy download en_core_web_sm

import spacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
words = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
freq = dict(Counter(words))

wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()'''

                def show_wc_spacy():
                    render_wordcloud(freq_wc_spacy, key="wc_spacy")
                    st.caption(
                        f"Built from **{len(freq_wc_spacy)}** unique lemmas — spaCy merges "
                        "inflections (e.g. \"runs\"/\"running\"→\"run\") before counting, so this "
                        "cloud can look noticeably different from the NLTK one above."
                    )

                code_and_output(code, show_wc_spacy, key="tool_wc_spacy")

        # ===================== CLASSIFICATION =====================
        elif tool == "Classification":
            st.markdown(
                "<span class='badge badge-amber'>Machine Learning</span>",
                unsafe_allow_html=True,
            )

            clf_source = st.radio(
                "Training data",
                [
                    "10 sample movie reviews (built-in)",
                    "Upload your own labeled CSV",
                ],
                key="clf_source_choice",
                horizontal=True,
            )

            if clf_source.startswith("10 sample movie reviews"):
                train_texts = SAMPLE_REVIEWS["review"].tolist()
                train_labels = SAMPLE_REVIEWS["sentiment"].tolist()
                st.warning(
                    "**Honesty check:** this trains on just **10 hand-labeled reviews** — "
                    "enough to show *how* a classifier learns, nowhere near enough to be a "
                    "reliable real classifier. Production systems train on thousands to "
                    "millions of labeled examples. Treat every prediction below as a "
                    "demonstration, not a verdict. For a version trained on a real 60-document "
                    "corpus with a genuine train/test split and honest accuracy, see "
                    "**Guided Learning → Classification & Clustering**."
                )
                with st.expander("The training data (10 reviews, hand-labeled)"):
                    st.dataframe(SAMPLE_REVIEWS, use_container_width=True, hide_index=True)
            else:
                st.caption(
                    "Upload a CSV with **one text column and one label column** — classification "
                    "needs real labels to learn from, so a plain PDF/text upload can't work here "
                    "(there'd be nothing to learn against). This is different from Clustering, "
                    "which needs no labels at all."
                )
                clf_csv = st.file_uploader(
                    "CSV file", type=["csv"], key="clf_csv_upload",
                    help="Nothing is stored — processed in memory only.",
                )
                train_texts, train_labels = [], []
                if clf_csv is not None:
                    try:
                        df_clf = pd.read_csv(clf_csv)
                        if df_clf.empty:
                            st.warning("That CSV appears to be empty.")
                        else:
                            all_cols = df_clf.columns.tolist()
                            c1, c2 = st.columns(2)
                            text_col = c1.selectbox("Text column", all_cols, key="clf_csv_text_col")
                            label_col = c2.selectbox(
                                "Label column", all_cols,
                                index=min(1, len(all_cols) - 1), key="clf_csv_label_col",
                            )
                            clean = df_clf[[text_col, label_col]].dropna()
                            train_texts = clean[text_col].astype(str).tolist()
                            train_labels = clean[label_col].astype(str).tolist()
                            n_classes = len(set(train_labels))
                            if len(train_texts) < 4 or n_classes < 2:
                                st.warning(
                                    f"Found {len(train_texts)} labeled row(s) across {n_classes} "
                                    "class(es) — need at least a few rows across 2+ classes to "
                                    "train anything meaningful."
                                )
                                train_texts, train_labels = [], []
                            else:
                                st.success(
                                    f"Loaded **{len(train_texts)} labeled rows**, "
                                    f"**{n_classes} classes**: "
                                    + ", ".join(f"{k} ({v})" for k, v in
                                                pd.Series(train_labels).value_counts().items())
                                )
                    except Exception as e:
                        st.error(f"Couldn't read that CSV: {e}")

            classify_input = st.text_input(
                "Type a sentence to classify:",
                value="This movie was absolutely wonderful, I loved it!",
                key="clf_input",
            )

            if not train_texts:
                st.info("Load training data above to enable classification.")
            else:
                st.markdown(
                    "<span class='badge badge-nltk'>NLTK path — Naive Bayes</span>",
                    unsafe_allow_html=True,
                )
                if classify_input.strip():
                    from nltk.classify import NaiveBayesClassifier

                    def review_features(words):
                        return {f"contains({w})": True for w in set(words)}

                    train_set_nltk = [
                        (review_features(word_tokenize(str(review).lower())), label)
                        for review, label in zip(train_texts, train_labels)
                    ]
                    clf_nltk = NaiveBayesClassifier.train(train_set_nltk)
                    feats_input = review_features(word_tokenize(classify_input.lower()))
                    pred_nltk = clf_nltk.classify(feats_input)
                    probs_nltk = clf_nltk.prob_classify(feats_input)

                    code = '''# Setup: nltk.download("punkt"); nltk.download("punkt_tab")
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

def review_features(words):
    return {f"contains({w})": True for w in set(words)}

train_set = [(review_features(word_tokenize(r.lower())), label)
             for r, label in zip(reviews, labels)]
classifier = NaiveBayesClassifier.train(train_set)

new_text = "This movie was absolutely wonderful, I loved it!"
print(classifier.classify(review_features(word_tokenize(new_text.lower()))))'''

                    def show_clf_nltk():
                        st.metric("Predicted sentiment", pred_nltk)
                        prob_df = pd.DataFrame(
                            [(label, probs_nltk.prob(label)) for label in clf_nltk.labels()],
                            columns=["Label", "Probability"],
                        ).sort_values("Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Label"))
                        top_conf = prob_df.iloc[0]
                        st.caption(
                            f"In plain terms: the model is **{top_conf['Probability']:.0%} "
                            f"confident** this text is **{top_conf['Label']}**, based purely "
                            f"on which of its words showed up anywhere in the "
                            f"{len(train_texts)} training examples for each label. NLTK's "
                            "Naive Bayes uses **word presence** (is this word in the text, "
                            "yes/no) rather than frequency or TF-IDF weighting — try the "
                            "scikit-learn path below to see how that changes the result."
                        )

                    code_and_output(code, show_clf_nltk, key="tool_clf_nltk")

                st.markdown(
                    "<div class='lib-section'></div>"
                    "<span class='badge badge-spacy'>scikit-learn path — TF-IDF + Naive Bayes</span>",
                    unsafe_allow_html=True,
                )
                if classify_input.strip():
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.naive_bayes import MultinomialNB

                    vec_clf = TfidfVectorizer()
                    X_train = vec_clf.fit_transform(train_texts)
                    y_train = train_labels
                    clf_sk = MultinomialNB()
                    clf_sk.fit(X_train, y_train)
                    X_input = vec_clf.transform([classify_input])
                    pred_sk = clf_sk.predict(X_input)[0]
                    proba_sk = clf_sk.predict_proba(X_input)[0]

                    code = '''# Setup: pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = TfidfVectorizer()
X_train = vec.fit_transform(reviews)
clf = MultinomialNB()
clf.fit(X_train, labels)

new_text = "This movie was absolutely wonderful, I loved it!"
X_new = vec.transform([new_text])
print(clf.predict(X_new)[0])
print(dict(zip(clf.classes_, clf.predict_proba(X_new)[0])))'''

                    def show_clf_sklearn():
                        st.metric("Predicted sentiment", pred_sk)
                        prob_df = pd.DataFrame(
                            {"Label": clf_sk.classes_, "Probability": proba_sk}
                        ).sort_values("Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Label"))
                        top_conf_sk = prob_df.iloc[0]
                        st.caption(
                            f"In plain terms: **{top_conf_sk['Probability']:.0%} "
                            f"confident** this text is **{top_conf_sk['Label']}**. This path "
                            "weights words by **TF-IDF** rather than plain presence — the "
                            "more standard real-world approach, and the same TfidfVectorizer "
                            "used in the Preprocessing Pipeline's bonus TF-IDF stage. If this "
                            "prediction disagrees with the NLTK path above, that's a genuine "
                            "and informative disagreement — it means the two weighting "
                            "schemes are picking up on different signals in your input."
                        )

                    code_and_output(code, show_clf_sklearn, key="tool_clf_sklearn")

        # ===================== CLUSTERING =====================
        elif tool == "Clustering":
            st.markdown(
                "<span class='badge badge-amber'>Machine Learning</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Clustering groups similar texts **without labels** — the opposite setup from "
                "Classification. Pick a data source below, then K-Means groups it by TF-IDF "
                "similarity and PCA flattens the vectors to 2D so the groups can be plotted."
            )

            clust_source = st.radio(
                "Data source",
                [
                    "10 sample movie reviews (built-in)",
                    "Your own documents (built-in corpus or upload)",
                ],
                key="clust_source_choice",
                horizontal=True,
            )

            if clust_source.startswith("10 sample movie reviews"):
                docs = SAMPLE_REVIEWS["review"].tolist()
                doc_labels = [f"Review {i}" for i in range(len(docs))]
            else:
                picked = render_multi_doc_picker(key_prefix="clust_tool")
                docs = [d["text"] for d in picked]
                doc_labels = [d["label"] for d in picked]

            if len(docs) < 2:
                st.info("Need at least 2 documents to cluster. Pick a source above with 2+ documents.")
            else:
                max_k = min(6, len(docs))
                k = st.slider(
                    "Number of clusters (k)", min_value=2, max_value=max(2, max_k),
                    value=min(2, max_k), key="clust_k",
                )

                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA

                vec_cl = TfidfVectorizer(stop_words="english")
                X_cl = vec_cl.fit_transform(docs)
                km = KMeans(n_clusters=k, n_init=10, random_state=0)
                cluster_labels = km.fit_predict(X_cl)

                code = f'''# Setup: pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(documents)          # TF-IDF vectors, one row per document
km = KMeans(n_clusters={k}, n_init=10, random_state=0)
labels = km.fit_predict(X)                # which cluster each document landed in

# Flatten to 2D purely for plotting — clustering itself uses the full vectors
coords = PCA(n_components=2).fit_transform(X.toarray())'''

                def show_clustering():
                    coords = PCA(n_components=2, random_state=0).fit_transform(X_cl.toarray())
                    fig, ax = plt.subplots(figsize=(7, 5))
                    scatter = ax.scatter(
                        coords[:, 0], coords[:, 1], c=cluster_labels, cmap="tab10", s=120,
                        edgecolors="black",
                    )
                    for i, (x, y) in enumerate(coords):
                        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(6, 6))
                    ax.set_xlabel("PCA component 1")
                    ax.set_ylabel("PCA component 2")
                    ax.set_title(f"K-Means clusters (k={k}), reduced to 2D via PCA")
                    st.pyplot(fig)
                    plt.close(fig)

                    result_df = pd.DataFrame({
                        "#": range(len(docs)),
                        "Document": [
                            l if len(l) <= 60 else l[:60] + "…" for l in doc_labels
                        ],
                        "Cluster": cluster_labels,
                    })
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    # Top terms per cluster centroid — what the cluster is actually "about"
                    terms = vec_cl.get_feature_names_out()
                    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
                    st.markdown(
                        "**What each cluster is about.** A centroid is the average TF-IDF "
                        "vector of every document K-Means put in that cluster — the words "
                        "below are simply the ones with the highest average weight in that "
                        "cluster, i.e. the words that best characterize it:"
                    )
                    for c in range(k):
                        centroid = km.cluster_centers_[c]
                        top_terms_idx = centroid.argsort()[::-1][:6]
                        top_terms = ", ".join(terms[i] for i in top_terms_idx if centroid[i] > 0)
                        n_docs_c = cluster_sizes.get(c, 0)
                        st.caption(
                            f"**Cluster {c}** ({n_docs_c} document(s)): "
                            f"{top_terms or '(no distinctive terms)'}"
                        )

                code_and_output(code, show_clustering, key="tool_clustering")

            st.info(
                "**Why PCA?** TF-IDF vectors have one dimension per vocabulary word — "
                "far more than 2 or 3, impossible to plot directly. PCA finds the two "
                "directions of greatest variance and projects onto those, losing information "
                "but making the clusters visually inspectable. K-Means itself runs on the "
                "full, un-reduced vectors."
            )

    # -----------------------------------------------------------------------
    # ABOUT
    # -----------------------------------------------------------------------
    else:
        st.header("About NLPPlayground")
        st.markdown(
            """
            **NLPPlayground** is an interactive platform for learning Python, R, and Natural
            Language Processing by *seeing real code and real (or worked-example) output side
            by side* — built for students, business professionals, and practitioners alike.

            **This is an early Phase 1 preview.** Word clouds, Naive Bayes classification, and
            K-Means clustering (with TF-IDF) have joined Quick Tools. More lessons, datasets,
            and tools (topic modeling, AI-powered explanations, and side-by-side text
            comparison) are on the way.

            ---
            **A note on privacy:** this demo processes text in-memory only and does not store or
            share what you paste. Even so, please avoid pasting confidential or sensitive
            information into any online demo tool.

            **A note on the R track:** this app runs on Python (Streamlit), so R code shown is
            **not executed live** — outputs are worked examples, clearly labeled as such. Copy
            the code and run it yourself in RStudio or Posit Cloud to experiment with your own text.
            """
        )

# ===========================================================================
# R TRACK
# ===========================================================================
else:

    # -----------------------------------------------------------------------
    # GUIDED LEARNING — Module 1: R Fundamentals (static examples)
    # -----------------------------------------------------------------------
    if mode == "Guided Learning":
        st.markdown(
            "<div class='module-head'><span class='badge badge-green'>R · Reference track</span>"
            "<h2>R Fundamentals</h2>"
            "<p>Real, correct R code with precomputed output — this app runs on Python.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card'><b>R reference track.</b> Eight short lessons mirroring "
            "the Python Fundamentals track. The code is real, standard R — but since this app "
            "runs on Python, output here is a precomputed, hand-verified worked example rather "
            "than live-executed. Copy the code into RStudio or Posit Cloud to experiment.<br><br>"
            "<b>Scope note, so you can plan:</b> R deliberately covers Fundamentals only. "
            "The NLTK and spaCy tracks are Python libraries with no direct R equivalent "
            "(R's counterparts would be <code>tidytext</code> and <code>quanteda</code>), and "
            "we'd rather be upfront about that than ship a shallow imitation.</div>",
            unsafe_allow_html=True,
        )


        # --- Lesson 0: What is R? ---
        if lesson_idx == 0:
            st.subheader("What is R?")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>3 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "**R is a programming language built specifically for statistics and data "
                "analysis.** It's widely used in academic research, social sciences, and "
                "increasingly for text analytics through packages like `tidytext` and `quanteda`. "
                "If Python is the generalist, R is the specialist statistician."
            )
            st.info(
                "R runs code the same way any programming language does — line by line, "
                "in order. Its syntax differs from Python in small but important ways, like "
                "using `<-` for assignment instead of `=`."
            )

            with st.expander("Why R for NLP specifically?"):
                st.markdown(
                    "R has a strong tradition in academic text analysis, especially through "
                    "packages like `tidytext` (tidyverse-style text mining), `quanteda` "
                    "(quantitative text analysis), and `tm` (an older but still-used text "
                    "mining framework). If your field leans toward social science or applied "
                    "statistics research, you'll likely encounter R-based text analytics."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- R traditionally uses `<-` for assignment, not `=` (though `=` also "
                    "works in most contexts — `<-` is the convention).\n"
                    "- R is also case-sensitive, just like Python: `Text` and `text` are different.\n"
                    "- `cat()` prints without quotes around strings; `print()` includes them."
                )

            code = '''# Setup: none needed — cat() is part of base R, no packages required.
name <- "Learner"
cat("Hello,", name, "- welcome to NLPPlayground!\\n")'''

            def show_r_intro():
                st.code("Hello, Learner - welcome to NLPPlayground!", language="text")
                st.caption("`cat()` concatenates and prints text — one of the most common R output functions.")

            code_and_output_r(code, show_r_intro, key="r_lesson0_intro")
            render_r_practice_box(
                key="r_practice_0",
                placeholder='name <- "Your name here"\ncat("Hello,", name, "\\n")',
            )


        # --- Lesson 1: Variables & Data Types ---
        if lesson_idx == 1:
            st.subheader("Variables & Data Types")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Just like Python, R stores values in named variables. R's core types are "
                "`numeric` (numbers), `character` (text), and `logical` (TRUE/FALSE)."
            )
            st.info(
                "R uses `<-` to assign values to variables. `class()` tells you the data "
                "type — R's equivalent of Python's `type()`."
            )

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Text in R is stored as `character` type. Sentiment scores from packages "
                    "like `tidytext` typically come back as `numeric`. Knowing which type "
                    "you're working with determines which functions you can apply to it."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- R's TRUE/FALSE must be written in ALL CAPS (or as `T`/`F`), unlike "
                    "Python's `True`/`False`.\n"
                    "- `paste()` joins strings with a space by default; use `paste0()` to "
                    "join with no space.\n"
                    "- Numbers in R default to `numeric` (double), not a separate `int` type "
                    "like Python."
                )

            code = '''# Setup: none needed — class(), print() are part of base R, no packages required.
age <- 7
word <- "sentiment"
is_learning <- TRUE

print(class(age))          # "numeric"
print(class(word))         # "character"
print(class(is_learning))  # "logical"
print(age * 2)              # 14
print(paste(word, word))    # "sentiment sentiment"'''

            def show_r_types():
                st.code(
                    '[1] "numeric"\n[1] "character"\n[1] "logical"\n[1] 14\n[1] "sentiment sentiment"',
                    language="text",
                )

            code_and_output_r(code, show_r_types, key="r_lesson1_types")
            render_r_practice_box(
                key="r_practice_1",
                placeholder='my_number <- 42\nmy_word <- "hello"\nprint(class(my_number))\nprint(class(my_word))',
            )


        # --- Lesson 2: Strings & Text ---
        if lesson_idx == 2:
            st.subheader("Strings & Text")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "R has built-in functions for inspecting and transforming text, similar in "
                "spirit to Python's string methods but with different function names."
            )
            st.info("`nchar()` counts characters, `toupper()`/`tolower()` change case, `strsplit()` splits text into pieces.")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Just like in Python, string operations are the entry point for every "
                    "R-based text analysis. Packages like `tidytext` build on top of these "
                    "fundamentals — `unnest_tokens()`, for example, relies on the same "
                    "splitting logic as `strsplit()` underneath."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- `strsplit()` returns a **list**, so you usually need `[[1]]` to get "
                    "the actual vector of pieces out.\n"
                    "- R has no simple built-in string-reverse function — it takes combining "
                    "`strsplit()`, `rev()`, and `paste()` together, shown below.\n"
                    "- `nchar()` counts characters, not words — use `strsplit()` + `length()` for word count."
                )

            code = '''# Setup: none needed — nchar(), toupper(), strsplit() are part of base R, no packages required.
s <- "NLPPlayground makes learning NLP fun"

nchar(s)                                        # number of characters
toupper(s)                                       # all uppercase
strsplit(s, " ")[[1]]                            # split into words
paste(rev(strsplit(s, "")[[1]]), collapse = "")  # reverse the string'''

            def show_r_strings():
                st.code(
                    '[1] 36\n'
                    '[1] "NLPPLAYGROUND MAKES LEARNING NLP FUN"\n'
                    '[1] "NLPPlayground" "makes"         "learning"      "NLP"           "fun"\n'
                    '[1] "nuf PLN gninrael sekam dnuorgyalPPLN"',
                    language="text",
                )

            code_and_output_r(code, show_r_strings, key="r_lesson2_strings")
            render_r_practice_box(
                key="r_practice_2",
                placeholder='s <- "Type your own sentence here"\nprint(nchar(s))\nprint(toupper(s))',
            )


        # --- Lesson 3: Vectors & Loops ---
        if lesson_idx == 3:
            st.subheader("Vectors & Loops")
            st.markdown(
                "<span class='badge badge-green'>Beginner</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "R's equivalent of a Python list is a **vector**, created with `c()` "
                "(short for \"combine\"). Loops in R work much like Python's, using `for`."
            )
            st.info("A **vector** holds multiple values of the same type. `for` loops let you process each element in turn.")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "R's `tidytext` and `quanteda` packages are optimized to work on entire "
                    "vectors of text at once (a whole document collection), often avoiding "
                    "explicit loops in favor of vectorized operations — but understanding the "
                    "loop first makes the vectorized shortcuts easier to understand later."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- R vectors are **1-indexed** — the first item is `words[1]`, not `words[0]` "
                    "like Python. This trips up almost everyone coming from Python.\n"
                    "- `c(\"a\", \"b\", \"c\")` — don't forget the `c()` wrapper when creating a vector.\n"
                    "- `which.max(nchar(words))` gives the *position* of the longest word, not the word itself."
                )

            code = '''# Setup: none needed — c(), for loops are part of base R, no packages required.
words <- c("python", "nlp", "data", "ai", "learning")

for (word in words) {
  cat(word, "->", nchar(word), "characters\\n")
}

longest <- words[which.max(nchar(words))]
cat("Longest word:", longest, "\\n")'''

            def show_r_vectors():
                st.code(
                    'python -> 6 characters\n'
                    'nlp -> 3 characters\n'
                    'data -> 4 characters\n'
                    'ai -> 2 characters\n'
                    'learning -> 8 characters\n'
                    'Longest word: learning',
                    language="text",
                )

            code_and_output_r(code, show_r_vectors, key="r_lesson3_vectors")
            render_r_practice_box(
                key="r_practice_3",
                placeholder='words <- c("python", "nlp", "data")\nfor (w in words) {\n  print(w)\n}',
            )


        # --- Lesson 4: Conditionals ---
        if lesson_idx == 4:
            st.subheader("Conditionals (if / else if / else)")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>5 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "R's conditionals work the same way as Python's — the syntax just looks a "
                "little different, using curly braces `{}` instead of indentation."
            )
            st.info("R uses `else if` (two words) where Python uses `elif` (one word) — a common trip-up when switching between the two.")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Same as the Python track: turning a raw sentiment score into a readable "
                    "label like \"Positive\" or \"Negative\" is a conditional at its core, "
                    "whether you write it in R or Python."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- The `else` must go on the **same line** as the closing `}` of the "
                    "previous block, or R throws an error in scripts (this differs from "
                    "some other languages).\n"
                    "- R uses `{}` to group code, not indentation — though indenting is still "
                    "good practice for readability.\n"
                    "- `else if`, not `elseif` or `elif`."
                )

            code = '''# Setup: none needed — if/else if/else is part of base R, no packages required.
score <- 0.3

if (score > 0.1) {
  label <- "Positive"
} else if (score < -0.1) {
  label <- "Negative"
} else {
  label <- "Neutral"
}

print(label)'''

            def show_r_conditional():
                st.code('[1] "Positive"', language="text")
                st.caption("Example uses a fixed score of 0.3 — copy the code and change the value to see other labels.")

            code_and_output_r(code, show_r_conditional, key="r_lesson4_conditionals")
            render_r_practice_box(
                key="r_practice_4",
                placeholder='score <- 0.3\nif (score > 0.1) {\n  label <- "Positive"\n} else {\n  label <- "Neutral"\n}\nprint(label)',
            )


        # --- Lesson 5: Functions ---
        if lesson_idx == 5:
            st.subheader("Functions")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>7 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "R functions are defined with the `function()` keyword and use `return()` "
                "(or simply the last evaluated line) to send back a value."
            )
            st.info("A **function** packages reusable logic — exactly like Python. Real R text-cleaning pipelines are built the same way.")

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "R-based NLP workflows (especially with `tidytext`) chain small functions "
                    "together using the pipe operator `%>%` or `|>`: clean text, tokenize, "
                    "remove stopwords, count. Writing your own `clean_text()` function here is "
                    "the same building block used throughout real R text-mining pipelines."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- Unlike Python, R functions **don't require** an explicit `return()` — "
                    "the last evaluated expression is returned automatically. Relying on this "
                    "can be confusing, so many style guides recommend using `return()` explicitly anyway.\n"
                    "- `gsub()` (used for removing punctuation) takes a regex pattern — "
                    "special regex characters need escaping.\n"
                    "- Function arguments in R can have default values, e.g. `function(text, lower = TRUE)`."
                )

            code = '''# Setup: none needed — function(), gsub(), tolower() are part of base R, no packages required.
clean_text <- function(text) {
  text <- tolower(text)
  text <- gsub("[[:punct:]]", "", text)
  return(text)
}

result <- clean_text("WOW!!! This Product is AMAZING... totally worth it!!")
print(result)'''

            def show_r_functions():
                st.write("**Before:**")
                st.code("WOW!!! This Product is AMAZING... totally worth it!!", language="text")
                st.write("**After `clean_text()`:**")
                st.code("wow this product is amazing totally worth it", language="text")

            code_and_output_r(code, show_r_functions, key="r_lesson5_functions")
            render_r_practice_box(
                key="r_practice_5",
                placeholder='square <- function(x) {\n  return(x * x)\n}\nprint(square(5))',
            )


        # --- Lesson 6: Named Lists ---
        if lesson_idx == 6:
            st.subheader("Named Lists")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>6 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "R's rough equivalent of Python's dictionary is a **named list** — or, for "
                "quick frequency counting specifically, R's built-in `table()` function, "
                "which behaves like a phonebook: look up a word, get its count."
            )
            st.info(
                "`table()` counts occurrences of each unique value in a vector — exactly "
                "what you need for word-frequency counting, R's version of a dictionary lookup."
            )

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "Word frequency counting — what the Python **Keyword Extraction** tool "
                    "does with a dictionary — is done in R with `table()`. Sentiment lexicons "
                    "(word → score) and stopword lookups follow the same key-value pattern, "
                    "usually implemented as named lists or named vectors in R."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- `table()` returns a special `table` object, not a plain named list — "
                    "use `as.list(table(...))` if you need list behavior specifically.\n"
                    "- Accessing a count for a word not present returns nothing (length-zero), "
                    "not zero — check with `\"word\" %in% names(word_counts)` first.\n"
                    "- `table()` sorts alphabetically by default, not by frequency — use "
                    "`sort(table(...), decreasing = TRUE)` to sort by count."
                )

            code = '''# Setup: none needed — table(), strsplit() are part of base R, no packages required.
text <- "the cat sat on the mat"
words <- strsplit(text, " ")[[1]]

word_counts <- table(words)
print(word_counts)
cat("Count of the:", word_counts[["the"]], "\\n")'''

            def show_r_namedlist():
                st.code(
                    'words\n'
                    'cat mat  on sat the \n'
                    '  1   1   1   1   2 \n'
                    'Count of the: 2',
                    language="text",
                )

            code_and_output_r(code, show_r_namedlist, key="r_lesson6_namedlists")
            render_r_practice_box(
                key="r_practice_6",
                placeholder='scores <- list(good = 1, bad = -1, neutral = 0)\nprint(scores$good)',
            )


        # --- Lesson 7: Data Frames ---
        if lesson_idx == 7:
            st.subheader("Data Frames")
            st.markdown(
                "<span class='badge badge-amber'>Intermediate</span>"
                "<span class='badge badge-blue'>6 min</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "A **data frame** is R's built-in table structure — rows and named columns, "
                "just like a spreadsheet. Unlike Python (where you need the separate `pandas` "
                "library), data frames are a core part of base R itself."
            )
            st.info(
                "`data.frame()` builds a table from equal-length vectors — one vector per "
                "column, exactly like Python's `pd.DataFrame()` built from equal-length lists."
            )

            with st.expander("Why this matters for NLP"):
                st.markdown(
                    "R's `tidytext` and `quanteda` packages both return results as data "
                    "frames — a table of words, documents, and scores. Recognizing this "
                    "structure here means you'll immediately understand the output shape "
                    "of any real R text-analysis package."
                )

            with st.expander("Common beginner mistakes"):
                st.markdown(
                    "- All vectors passed into `data.frame()` must be the **same length**, "
                    "just like Python's DataFrame.\n"
                    "- By default, R may convert text columns to `factor` type in older "
                    "versions — modern R (4.0+) keeps them as `character` by default.\n"
                    "- `order(-df$length)` sorts descending by negating the column; there's "
                    "no built-in `ascending=FALSE` argument like pandas has."
                )

            code = '''# Setup: none needed — data.frame(), nchar(), order() are part of base R, no packages required.
words <- c("python", "nlp", "data", "ai", "learning")
lengths <- nchar(words)

df <- data.frame(word = words, length = lengths)
print(df[order(-df$length), ])'''

            def show_r_dataframe():
                st.code(
                    '      word length\n'
                    '5 learning      8\n'
                    '1   python      6\n'
                    '3     data      4\n'
                    '2      nlp      3\n'
                    '4       ai      2',
                    language="text",
                )

            code_and_output_r(code, show_r_dataframe, key="r_lesson7_dataframes")
            render_r_practice_box(
                key="r_practice_7",
                placeholder='df <- data.frame(word = c("cat", "dog"), count = c(3, 5))\nprint(df)',
            )


    # -----------------------------------------------------------------------
    # QUICK TOOLS (R, static worked examples)
    # -----------------------------------------------------------------------

        lesson_footer("R Fundamentals", lesson_idx, R_LESSONS, "nav_lesson_r")

    elif mode == "Quick Tools":
        st.header("Quick Tools (R)")
        st.caption(
            "Real R code with precomputed, hand-verified example output — not live-executed. "
            "Pick a preset example below."
        )
        st.info(
            "**Want to analyse your own text, upload a file, or use a public-domain "
            "corpus?** Switch the sidebar's language track to **Python** — those "
            "features need live execution, which only the Python track can do. The R code "
            "here is still fully correct and runnable in RStudio or Posit Cloud."
        )

        tool = st.selectbox(
            "Choose a tool",
            ["Sentiment Analysis", "Keyword Extraction", "Named Entity Recognition (simplified)"],
        )

        st.markdown("---")

        # --- Sentiment ---
        if tool == "Sentiment Analysis":
            st.markdown(
                "**Example text:** *\"I absolutely loved the new design of this app, it's clean and easy to use!\"*"
            )
            st.caption(
                "Production R sentiment analysis usually joins text against a lexicon like "
                "`tidytext`'s `bing` or `afinn` word lists. This example shows the same idea "
                "with a small illustrative lexicon so the output can be verified by hand."
            )

            code = '''# Setup (run once): install.packages("stringr")
library(stringr)

text <- "I absolutely loved the new design of this app, it's clean and easy to use!"
positive_words <- c("loved", "great", "excellent", "clean", "easy", "amazing", "best")
negative_words <- c("terrible", "bad", "worst", "hate", "confusing", "boring")

words <- str_split(tolower(text), "\\\\W+")[[1]]
pos_count <- sum(words %in% positive_words)
neg_count <- sum(words %in% negative_words)

cat("Positive words found:", pos_count, "\\n")
cat("Negative words found:", neg_count, "\\n")
sentiment <- ifelse(pos_count > neg_count, "Positive",
              ifelse(neg_count > pos_count, "Negative", "Neutral"))
cat("Sentiment:", sentiment, "\\n")'''

            def show_r_sentiment():
                st.metric("Sentiment", "Positive ")
                col_a, col_b = st.columns(2)
                col_a.metric("Positive words found", 3)
                col_b.metric("Negative words found", 0)
                st.caption("Matched positive words: loved, clean, easy")

            code_and_output_r(code, show_r_sentiment, key="r_tool_sentiment")

        # --- Keyword Extraction ---
        elif tool == "Keyword Extraction":
            st.markdown(
                "**Example text:** *\"I absolutely loved the new design of this app, it's clean and easy to use!\"*"
            )
            st.caption(
                "Real R keyword extraction typically uses `tidytext::unnest_tokens()` plus "
                "`anti_join()` against a stopword list. This simplified version uses base R "
                "string splitting so every step is verifiable by hand."
            )

            code = '''# Setup (run once): install.packages("stringr")
library(stringr)

text <- "I absolutely loved the new design of this app, it's clean and easy to use!"
stopwords_small <- c("i", "the", "of", "this", "it's", "and", "to", "a", "is", "new")

words <- str_split(tolower(text), "\\\\W+")[[1]]
words <- words[words != ""]
keywords <- words[!(words %in% stopwords_small) & nchar(words) > 1]

table(keywords)'''

            def show_r_keywords():
                kw_df = pd.DataFrame(
                    {"Keyword": ["absolutely", "loved", "design", "app", "it", "clean", "easy", "use"],
                     "Frequency": [1, 1, 1, 1, 1, 1, 1, 1]}
                )
                st.dataframe(kw_df, use_container_width=True, hide_index=True)

            code_and_output_r(code, show_r_keywords, key="r_tool_keywords")

        # --- NER (simplified) ---
        else:
            st.markdown(
                "**Example text:** *\"Tim Cook visited Paris and met officials from Google.\"*"
            )
            st.caption(
                "Real R NER typically uses the `spacyr` or `udpipe` packages to call proper "
                "NLP models. This simplified base-R version just detects capitalized word runs "
                "as a teaching illustration — like NLTK's chunker on the Python side, it's not "
                "production-accurate."
            )

            code = '''# Setup: none needed — grepl(), strsplit(), gsub() are part of base R, no packages required.
text <- "Tim Cook visited Paris and met officials from Google."
words <- strsplit(gsub("[.]", "", text), " ")[[1]]

is_cap <- grepl("^[A-Z]", words)
# group consecutive capitalized words into entities
entities <- c()
current <- c()
for (i in seq_along(words)) {
  if (is_cap[i]) {
    current <- c(current, words[i])
  } else if (length(current) > 0) {
    entities <- c(entities, paste(current, collapse = " "))
    current <- c()
  }
}
if (length(current) > 0) entities <- c(entities, paste(current, collapse = " "))

print(entities)'''

            def show_r_ner():
                ent_df = pd.DataFrame({"Entity": ["Tim Cook", "Paris", "Google"]})
                st.dataframe(ent_df, use_container_width=True, hide_index=True)

            code_and_output_r(code, show_r_ner, key="r_tool_ner")

    # -----------------------------------------------------------------------
    # ABOUT (R track shares the same About content)
    # -----------------------------------------------------------------------
    else:
        st.header("About NLPPlayground")
        st.markdown(
            """
            **NLPPlayground** is an interactive platform for learning Python, R, and Natural
            Language Processing by *seeing real code and real (or worked-example) output side
            by side* — built for students, business professionals, and practitioners alike.

            **This is an early Phase 1 preview.** Word clouds, Naive Bayes classification, and
            K-Means clustering (with TF-IDF) have joined Quick Tools. More lessons, datasets,
            and tools (topic modeling, AI-powered explanations, and side-by-side text
            comparison) are on the way.

            ---
            **A note on privacy:** this demo processes text in-memory only and does not store or
            share what you paste. Even so, please avoid pasting confidential or sensitive
            information into any online demo tool.

            **A note on the R track:** this app runs on Python (Streamlit), so R code shown is
            **not executed live** — outputs are worked examples, clearly labeled as such. Copy
            the code and run it yourself in RStudio or Posit Cloud to experiment with your own text.
            """
        )
