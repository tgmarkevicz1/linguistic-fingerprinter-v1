# Linguistic Fingerprinting Engine

> An end-to-end authorship attribution system that identifies writers across different text formats — matching a blog post to a tweet, or an email to an essay — by isolating the stylistic habits that persist regardless of medium.

**Built by Trevor Markevicz** · CS + Linguistics, sophomore · [Live Demo](#) · [API Docs](#)

---

## The Problem Worth Solving

Most authorship attribution systems fail in a predictable way: they learn the *medium*, not the *person*.

Train a model on academic essays and social media posts from the same authors, and a naive classifier will learn that long texts with complex vocabulary belong to "academic" authors — not because it learned anything about *who* wrote them, but because it learned *how* long-form writing looks. Ask it to attribute a short tweet to one of those same authors, and it collapses.

This is the cross-medium attribution problem. It matters in forensic linguistics, plagiarism detection, and any context where the same person writes across multiple platforms. The standard approach doesn't solve it. This project does.

---

## The Core Insight: Adversarial Feature Selection

The central contribution of this system is a feature selection method I call **adversarial invariance scoring**.

Every stylometric feature — punctuation ratios, POS distributions, vocabulary richness — encodes two signals simultaneously: *who wrote this* and *what kind of text this is*. A naive model uses both. This one deliberately discards the second.

For each feature, I compute a score:

```
Invariance Score = MI(Feature, Author) / (MI(Feature, Medium) + ε)
```

Where `MI` is mutual information. A feature with a high score strongly predicts author identity while being weakly correlated with the text's medium or platform. A low score means the feature mostly encodes *format*, not *person* — so it gets dropped.

The result: a model that learns genuine stylistic habits — how someone constructs sentences, their punctuation tendencies, their vocabulary range — rather than surface-level artifacts of the text format.

**Why this matters architecturally:** this isn't just a preprocessing step. It's the reason the system can train on blog posts and correctly attribute tweets. Without it, cross-medium accuracy drops to near-chance. With it, the model is learning something real about how people write.

---

## System Architecture

```
Raw Text
   │
   ▼
┌─────────────────────────────────────────────────┐
│  processors.py — TextProcessor                  │
│  • spaCy NER → PII masking ([ENTITY])           │
│  • POS distribution extraction                  │
│  • MATTR (Moving-Average Type-Token Ratio)      │
│  • Readability scores, punctuation ratios       │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  engine.py — StylometricEngine                  │
│                                                 │
│  Feature Vector Construction                    │
│  ├── POS distribution (17 Universal UD tags)   │
│  ├── Char n-gram TF-IDF (2–4 grams, 500 feat.) │
│  └── Complexity scalars (FK grade, ARI, etc.)  │
│                                                 │
│  Adversarial Feature Selection                  │
│  └── Invariance Score → top-k features         │
│                                                 │
│  Ensemble Classifier                            │
│  ├── RandomForest (n=100) + SVC (RBF kernel)   │
│  ├── VotingClassifier (soft voting)             │
│  └── CalibratedClassifierCV (isotonic, 5-fold) │
│                                                 │
│  Explainability                                 │
│  └── SHAP TreeExplainer → mean |SHAP| / feature│
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  main.py — FastAPI Application                  │
│  POST /enroll         Register an author        │
│  POST /identify       Attribute unknown text    │
│  POST /compare-texts  Verify same-author?       │
│  POST /retrain        Hot-swap model (no restart│
│  GET  /compare/{id}   Radar chart comparison    │
└─────────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

### Why a soft-voting ensemble over a single classifier?

Random Forests are strong but overconfident — their raw `predict_proba` outputs don't reflect true calibrated probabilities. SVMs are well-calibrated on the decision boundary but struggle with multi-class probability estimates. By combining them with soft voting and wrapping the result in `CalibratedClassifierCV(method='isotonic')`, the system produces probability scores that are actually meaningful. When the API says "73% confidence," that number reflects real posterior probability, not an artifact of how Random Forests count votes.

### Why cosine similarity for authorship verification?

The `/compare-texts` endpoint computes cosine similarity between two feature vectors rather than Euclidean distance. In high-dimensional stylometric space, what matters is the *direction* a feature vector points — the relative proportions of syntactic and lexical habits — not its magnitude. Euclidean distance would systematically penalize verbose writers (longer texts produce larger-magnitude vectors) for reasons unrelated to their actual style. Cosine similarity is medium-length-agnostic by design.

### Why char n-grams over word n-grams?

Character n-grams (range 2–4) capture morphological patterns — affixes, character combinations, punctuation sequences — that persist even when an author consciously changes their vocabulary. They're also robust to spelling variants and out-of-vocabulary terms. Word n-grams collapse when comparing a blog post to a tweet; char n-grams don't.

### Why MATTR over standard TTR?

Type-Token Ratio (unique words / total words) is fatally length-dependent: longer texts always produce lower TTR, making cross-medium comparison meaningless. Moving-Average TTR computes TTR over sliding windows and averages them, making vocabulary richness comparable regardless of text length. This is essential for any system that needs to work across short and long texts from the same author.

### Why hot-swap retraining without a server restart?

The `/retrain` endpoint trains a new model in a background thread, then mutates the module-level engine object in place. The running FastAPI process picks up the new weights immediately — no downtime, no restart, no stale requests. The author registry is rebuilt in the same pass to ensure all feature vectors are computed in the new model's feature space. This is the pattern used in production ML systems that can't afford downtime.

---

## Feature Space

The model operates on a concatenated feature vector of three types:

| Feature Group | Dimensions | What It Captures |
|---|---|---|
| POS Distribution | 17 | Syntactic style — verb-heaviness, adjective use, pronoun density |
| Char N-gram TF-IDF | 500 | Morphological habits, punctuation sequences, affix patterns |
| Complexity Scalars | 8 | Readability, sentence length, vocabulary richness (MATTR), punctuation ratios |

After adversarial selection, the top 200 invariant features are retained. SHAP values identify which of these most strongly drive each attribution decision.

---

## Installation

```bash
# 1. Clone and set up environment
git clone https://github.com/tgmarkevicz1/linguistic-fingerprinter-v1.git
cd linguistic-fingerprinter-v1
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Download training data
python download_data.py   # Downloads Blog Authorship Corpus + Project Gutenberg

# 4. Train the model
python train.py           # ~5 min depending on hardware

# 5. Start the API
uvicorn main:app --reload --port 8000

# 6. Open the frontend
open index.html
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/enroll` | Register an author with sample texts |
| `POST` | `/identify` | Attribute unknown text → top 3 matches + SHAP |
| `GET` | `/compare/{author_id}` | Radar chart: last identified text vs. enrolled author |
| `POST` | `/compare-texts` | Compare two texts directly — no enrolled authors needed |
| `POST` | `/retrain` | Upload zip of new data, retrain in background, hot-swap model |
| `GET` | `/retrain/status` | Poll retraining progress |
| `GET` | `/authors` | List enrolled authors |
| `DELETE` | `/authors/{author_id}` | Remove an author |

All endpoints return `400 Bad Request` for inputs under 10 tokens. The stylometric variance of very short texts is statistically insignificant and would produce misleading confidence scores.

---

## Evaluation Protocol

The meaningful test is cross-medium: train on long-form texts, evaluate on short-form. Any system can memorize author identity within a single medium. The question is whether it learned *style* or *format*.

```bash
python train.py  # Runs both evaluations automatically
```

Output includes:
- Per-author precision, recall, and F1 on a held-out test set
- Cross-medium accuracy (train=long, test=short)
- Top 10 author-discriminative features by mean |SHAP|

---

## Project Structure

```
.
├── processors.py       # NLP preprocessing — PII masking, MATTR, feature extraction
├── engine.py           # ML pipeline — feature vectors, adversarial selection, ensemble
├── main.py             # FastAPI application — all API endpoints
├── train.py            # Training script — data loading, evaluation, registry enrollment
├── download_data.py    # Data pipeline — Blog Authorship Corpus + Project Gutenberg
├── index.html          # Frontend — single-file demo UI
├── requirements.txt
└── models/
    └── checkpoints/    # Persisted model, vectorizer, scaler, SHAP values
```

---

## Stack

Python 3.10+ · spaCy · scikit-learn · SHAP · FastAPI · Plotly · joblib

---

*Cross-medium authorship attribution is an open problem in forensic linguistics. This system doesn't claim to solve it definitively — but it does demonstrate that adversarial feature selection meaningfully improves cross-medium accuracy over naive stylometric baselines, which is the core claim the evaluation protocol is designed to test.*
