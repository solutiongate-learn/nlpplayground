"""
NLPPlayground - Phase 1 MVP
Learn Python & NLP through interactive, code-visible exercises.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (see README.md)
"""

import streamlit as st
import pandas as pd
from collections import Counter
import re
import string

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

try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except Exception:
    TEXTBLOB_OK = False

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NLPPlayground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.92;
        margin: 0;
    }

    /* Card style container */
    .card {
        background: #ffffff10;
        border: 1px solid rgba(120,120,120,0.15);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
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

    /* Make columns stack nicely + look good on mobile */
    @media (max-width: 768px) {
        .hero h1 { font-size: 1.5rem; }
        .hero p { font-size: 0.9rem; }
    }

    footer, #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample dataset (small, bundled inline so there is zero setup friction)
# ---------------------------------------------------------------------------
SAMPLE_REVIEWS = pd.DataFrame({
    "review": [
        "This movie was absolutely fantastic! The acting was superb and the story kept me hooked till the end.",
        "Waste of time. The plot made no sense and the pacing was painfully slow.",
        "It was okay, nothing special but not terrible either. Decent way to spend an evening.",
        "One of the best films I've seen this year. Brilliant direction by Christopher Nolan and outstanding cinematography.",
        "Terrible acting, weak script, and the ending felt completely rushed. Very disappointed.",
        "A heartwarming story with great performances from the entire cast. Highly recommended!",
        "The special effects were amazing but the storyline was confusing and hard to follow.",
        "I loved every minute of it. Steven Spielberg really outdid himself with this masterpiece.",
        "Boring from start to finish. I almost fell asleep twice.",
        "Solid movie overall. Great soundtrack, good performances, worth watching once.",
    ]
})

# ---------------------------------------------------------------------------
# Helper: reusable code + output panel
# ---------------------------------------------------------------------------
def code_and_output(code: str, render_output):
    """Shows Python code on the left, live output on the right."""
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("🐍 Python code")
        st.code(code, language="python")
        st.download_button(
            "📥 Download this code",
            data=code,
            file_name="nlp_snippet.py",
            mime="text/x-python",
            use_container_width=True,
        )
    with c2:
        st.caption("📊 Output")
        render_output()


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 NLPPlayground</h1>
        <p>Learn Python &amp; NLP by writing, running, and seeing real code — no setup required.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📚 Navigate")
    mode = st.radio(
        "Choose your path",
        ["🎓 Guided Learning", "⚡ Quick Tools", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 📈 Your Progress")
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    st.progress(st.session_state.progress / 100)
    st.caption(f"{st.session_state.progress}% of Module 1 complete")
    st.markdown("---")
    st.caption("⚠️ Demo tool — please don't paste confidential or sensitive text.")

# ---------------------------------------------------------------------------
# GUIDED LEARNING — Module 1: Python Basics
# ---------------------------------------------------------------------------
if mode == "🎓 Guided Learning":
    st.header("Module 1: Python Basics")
    st.caption("Three short, hands-on lessons. Type your own examples and watch real Python run.")

    lesson = st.tabs(["1️⃣ Variables & Strings", "2️⃣ Lists & Loops", "3️⃣ Functions"])

    # --- Lesson 1 ---
    with lesson[0]:
        st.subheader("Variables & Strings")
        st.markdown(
            "<span class='badge badge-green'>Beginner</span>"
            "<span class='badge badge-blue'>5 min</span>",
            unsafe_allow_html=True,
        )
        st.info("💡 A **string** is just text stored in a variable. Python gives you built-in tools to inspect and transform it.")

        user_string = st.text_input("Type any sentence:", "NLPPlayground makes learning NLP fun")

        code = f'''s = "{user_string}"

print(len(s))          # number of characters
print(s.upper())       # all uppercase
print(s.split())       # split into a list of words
print(s[::-1])         # reverse the string'''

        def show_string_output():
            st.metric("Length", len(user_string))
            st.write("**Uppercase:**", user_string.upper())
            st.write("**Split into words:**", user_string.split())
            st.write("**Reversed:**", user_string[::-1])

        code_and_output(code, show_string_output)

        if st.button("✓ Mark lesson complete", key="l1"):
            st.session_state.progress = min(100, st.session_state.progress + 34)
            st.success("Nice work! Progress updated.")

    # --- Lesson 2 ---
    with lesson[1]:
        st.subheader("Lists & Loops")
        st.markdown(
            "<span class='badge badge-green'>Beginner</span>"
            "<span class='badge badge-blue'>5 min</span>",
            unsafe_allow_html=True,
        )
        st.info("💡 A **list** stores multiple items. **Loops** let you process each item one at a time — the foundation of analyzing many texts.")

        items_raw = st.text_input("Enter a few words, comma-separated:", "python, nlp, data, ai, learning")
        items = [i.strip() for i in items_raw.split(",") if i.strip()]

        code = f'''words = {items}

for word in words:
    print(word, "->", len(word), "characters")

longest = max(words, key=len)
print("Longest word:", longest)'''

        def show_list_output():
            for w in items:
                st.write(f"`{w}` → {len(w)} characters")
            if items:
                st.success(f"Longest word: **{max(items, key=len)}**")

        code_and_output(code, show_list_output)

        if st.button("✓ Mark lesson complete", key="l2"):
            st.session_state.progress = min(100, st.session_state.progress + 33)
            st.success("Nice work! Progress updated.")

    # --- Lesson 3 ---
    with lesson[2]:
        st.subheader("Functions")
        st.markdown(
            "<span class='badge badge-amber'>Intermediate</span>"
            "<span class='badge badge-blue'>7 min</span>",
            unsafe_allow_html=True,
        )
        st.info("💡 A **function** packages reusable logic. This is exactly how real NLP pipelines clean text before analysis.")

        user_text = st.text_area(
            "Paste a messy sentence (extra punctuation, CAPS, etc.):",
            "WOW!!! This Product is AMAZING... totally worth it!!",
        )

        code = '''import string

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

        code_and_output(code, show_clean_output)

        if st.button("✓ Mark lesson complete", key="l3"):
            st.session_state.progress = min(100, st.session_state.progress + 33)
            st.success("Module 1 complete! Head to Quick Tools to try real NLP tasks →")

# ---------------------------------------------------------------------------
# QUICK TOOLS
# ---------------------------------------------------------------------------
elif mode == "⚡ Quick Tools":
    st.header("Quick Tools")
    st.caption("Jump straight to a task. Paste your own text or try the sample dataset.")

    tool = st.selectbox(
        "Choose a tool",
        ["😊 Sentiment Analysis", "🔑 Keyword Extraction", "🏷️ Named Entity Recognition"],
    )

    # Shared input pattern
    input_choice = st.radio("Input source:", ["✍️ Paste my own text", "📚 Use a sample review"], horizontal=True)

    if input_choice == "📚 Use a sample review":
        chosen = st.selectbox("Pick a sample:", SAMPLE_REVIEWS["review"].tolist())
        text = chosen
    else:
        text = st.text_area("Paste text here:", "I absolutely loved the new design of this app, it's clean and easy to use!", height=120)

    st.markdown("---")

    # --- Sentiment ---
    if tool == "😊 Sentiment Analysis":
        if not TEXTBLOB_OK:
            st.error("TextBlob isn't installed. Add `textblob` to requirements.txt.")
        elif text.strip():
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            label = "Positive 😊" if polarity > 0.1 else "Negative 😞" if polarity < -0.1 else "Neutral 😐"

            code = f'''from textblob import TextBlob

blob = TextBlob("""{text[:60]}...""")
print(blob.sentiment)'''

            def show_sentiment_output():
                st.metric("Sentiment", label)
                col_a, col_b = st.columns(2)
                col_a.metric("Polarity", f"{polarity:.2f}", help="-1 = very negative, +1 = very positive")
                col_b.metric("Subjectivity", f"{subjectivity:.2f}", help="0 = factual, 1 = opinion-based")
                st.caption(
                    "💡 **Polarity** measures positive vs negative tone. **Subjectivity** measures "
                    "how much the text is personal opinion vs objective fact."
                )

            code_and_output(code, show_sentiment_output)

    # --- Keyword Extraction ---
    elif tool == "🔑 Keyword Extraction":
        if text.strip():
            stop_words = set(stopwords.words("english"))
            tokens = word_tokenize(clean_text(text))
            keywords = [t for t in tokens if t.isalpha() and t not in stop_words]
            freq = Counter(keywords).most_common(10)

            code = '''from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text.lower())
keywords = [t for t in tokens if t.isalpha() and t not in stop_words]

print(Counter(keywords).most_common(10))'''

            def show_keyword_output():
                if freq:
                    kw_df = pd.DataFrame(freq, columns=["Keyword", "Frequency"])
                    st.dataframe(kw_df, use_container_width=True, hide_index=True)
                    st.bar_chart(kw_df.set_index("Keyword"))
                else:
                    st.warning("No keywords found — try a longer sentence.")

            code_and_output(code, show_keyword_output)

    # --- NER ---
    elif tool == "🏷️ Named Entity Recognition":
        if text.strip():
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            tree = ne_chunk(tagged)

            entities = []
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity_text = " ".join(word for word, tag in subtree.leaves())
                    entities.append((entity_text, subtree.label()))

            code = '''from nltk import word_tokenize, pos_tag, ne_chunk

tokens = word_tokenize(text)
tagged = pos_tag(tokens)
tree = ne_chunk(tagged)

for subtree in tree:
    if hasattr(subtree, "label"):
        print(subtree.label(), "->", " ".join(w for w, t in subtree.leaves()))'''

            def show_ner_output():
                if entities:
                    ent_df = pd.DataFrame(entities, columns=["Entity", "Type"])
                    st.dataframe(ent_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No named entities detected in this text. Try text with names of people, places, or organizations.")
                st.caption("💡 NLTK's built-in chunker is lightweight — it's a good teaching tool, but production systems typically use spaCy or transformer-based models for higher accuracy.")

            code_and_output(code, show_ner_output)

# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
else:
    st.header("About NLPPlayground")
    st.markdown(
        """
        **NLPPlayground** is an interactive platform for learning Python and Natural Language
        Processing by *seeing real code run on real text* — built for students, business
        professionals, and practitioners alike.

        **This is an early Phase 1 preview.** More lessons, datasets, and tools (topic modeling,
        clustering, classification, AI-powered explanations, and side-by-side text comparison)
        are on the way.

        ---
        **A note on privacy:** this demo processes text in-memory only and does not store or
        share what you paste. Even so, please avoid pasting confidential or sensitive
        information into any online demo tool.
        """
    )
