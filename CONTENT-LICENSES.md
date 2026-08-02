# Content & Dependency Licensing

This document records the provenance and licence of every piece of software and
content NLPPlayground uses, so the project's intellectual-property position is
answerable rather than assumed.

**This is not legal advice.** Licences change, and several NLTK data packages have
genuinely unclear terms (documented below). If NLPPlayground is ever monetised,
the items flagged 🚩 should be reviewed by a qualified professional.

Last audited: 2 August 2026.

---

## 1. Software dependencies

| Package | Licence | Commercial use | Notes |
|---|---|---|---|
| Streamlit | Apache 2.0 | ✅ Yes | |
| pandas | BSD 3-Clause | ✅ Yes | |
| NLTK (source code) | Apache 2.0 | ✅ Yes | The *code* is permissive; its *data* is not uniformly so — see §2 |
| spaCy (source code) | MIT | ✅ Yes | |
| `en_core_web_sm` (model) | MIT | ✅ Yes | See §3 — cleaner than it first appears |
| pypdf | BSD 3-Clause | ✅ Yes | Pure-Python, installs with no transitive dependencies |
| wordcloud | MIT | ✅ Yes | Bundles `DroidSansMono.ttf` (Apache 2.0, also permissive) as its default font |
| matplotlib | PSF-based (BSD-style) | ✅ Yes | Used only to render the word cloud image; no GUI backend (`Agg`) |
| scikit-learn | BSD 3-Clause | ✅ Yes | Used for TF-IDF, Naive Bayes classification, and K-Means clustering |

Note: NLTK's **documentation** (as distinct from its code) is licensed
CC BY-NC-ND 3.0 US. We do not reproduce NLTK documentation text in this app.

---

## 2. NLTK data packages — licences vary per package

NLTK downloads corpora separately from its source code, and **each package carries
its own licence**. NLTK publishes a per-dataset summary at
[DATASET-LICENSES.md](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md).

### Packages this app downloads

| Package | Licence | Status |
|---|---|---|
| `averaged_perceptron_tagger` / `_eng` | MIT | ✅ Clean |
| `vader_lexicon` | MIT | ✅ Clean |
| `words` | Public Domain | ✅ Clean |
| `gutenberg` | Public Domain | ✅ Clean |
| `inaugural` | Public Domain | ✅ Clean |
| `udhr` | Public Domain | ✅ Clean |
| `punkt` / `punkt_tab` | **No licence attribute** | ⚠️ Unclarified |
| `stopwords` | **Unclarified** | ⚠️ Unclarified |
| `maxent_ne_chunker` / `_tab` | **"Distributed with Permission"** | 🚩 Restricted |

### 🚩 The one real risk: `maxent_ne_chunker`

NLTK's built-in NER chunker is trained on **ACE data from the Linguistic Data
Consortium**. NLTK's own licence overview warns that "Distributed with Permission"
packages "may prohibit redistribution, modification, or commercial use."

**Current position:** acceptable. NLPPlayground is free and educational, we do not
redistribute the corpus (it is fetched to the server at runtime by
`nltk.download()`), and NLTK's overall corpora terms permit non-commercial use.

**If NLPPlayground is ever monetised**, this package must be re-evaluated. The
straightforward mitigation is to drop NLTK's NER and use spaCy's, which is
licensed for commercial use (§3). The NLTK NER lesson would then become a
*described* comparison rather than a live one.

### ⚠️ `punkt` and `stopwords`

Both lack a declared licence in NLTK's index. They are long-standing, universally
used components distributed by NLTK for redistribution, and we are not aware of any
restriction — but "no declared licence" is not the same as "permissively licensed",
and this is recorded honestly rather than glossed over.

---

## 3. spaCy `en_core_web_sm` — why it is safe

The model is MIT-licensed but trained on **OntoNotes 5**, which comes from the LDC,
whose non-member agreement forbids commercial use. This looks alarming and is
frequently misread.

The resolution: **Explosion (spaCy's publisher) holds an LDC For-Profit Membership**
that permits them to incorporate LDC data into work products and redistribute the
resulting models commercially. The MIT licence on the released model is therefore
valid.

**Practical consequence:** for any future commercial version, spaCy's NER is the
safe choice and NLTK's is the liability — the reverse of the common assumption.

---

## 4. Text content in this app

| Content | Provenance | Licence |
|---|---|---|
| `SAMPLE_REVIEWS` (10 film reviews) | **Original text written for this app** | Ours |
| `SHARED_TEXT_TOKENS` / `SHARED_TEXT_ENTITIES` | **Original text written for this app** | Ours |
| All lesson prose, analogies, explanations | **Original text written for this app** | Ours |
| Public-domain classics (Austen, Carroll, Shakespeare, inaugural addresses, UDHR) | NLTK `gutenberg` / `inaugural` / `udhr` | Public Domain |

### Naming policy for example text

All example text uses **deliberately fictional identifiers** — `Example Corp`,
`Dr. Jane Doe` — following the RFC 2606 convention of reserved example names.

This policy exists because of a real mistake caught in review: an earlier draft used
an invented company name that turned out to belong to **several real businesses**,
while attributing a fabricated CFO, a fabricated $2.5M investment and fabricated
quarterly results to it. Inventing facts adjacent to a real company name is a
genuine risk with no teaching benefit.

An earlier draft of `SAMPLE_REVIEWS` also named two real film directors in
fabricated reviews. Removed for the same reason.

**Rule going forward:** no real company, person, product or work may appear in
invented example text. Real entities may only appear in genuinely public-domain
source material (where the text is authentic, not fabricated).

---

## 5. Recommended sources for future content

Safe, unrestricted sources for expanding lesson material:

**Business / finance**
- SEC EDGAR filings — US government works, public domain (17 U.S.C. § 105)
- Federal Reserve statements and minutes
- Bureau of Labor Statistics, Census Bureau reports
- FTC / SEC press releases

**Scientific**
- NASA, NIH, USGS, NOAA publications — US government works, public domain
- ⚠️ arXiv is **not** uniformly safe: licences are per-paper and many are
  non-commercial or author-retained. Check each paper individually.

**Literary / general**
- Project Gutenberg (verify per-text; most are public domain in the US)
- Wikipedia — CC BY-SA 4.0, which is usable **but viral**: attribution required and
  derivatives must be share-alike. Prefer public-domain sources to avoid this.

---

## 6. Re-audit triggers

Review this document when:

- The app becomes paid, sponsored, or otherwise commercial → **re-examine
  `maxent_ne_chunker` before launch**
- Any new NLTK corpus is added → check its row in NLTK's DATASET-LICENSES.md first
- Any new model, dataset, or third-party text is introduced
- Any example text is written that mentions a named entity
