# Technical Specification: Cross-Medium Linguistic Fingerprinting Engine

**Target Environment:** Python 3.10+  
**Primary Libraries:** `spacy` (en_core_web_sm), `scikit-learn`, `textstat`, `fastapi`, `plotly`, `joblib`

---

## 1. Project Overview
Build a modular authorship attribution system that identifies a unique **"Linguistic DNA."** The system must be **medium-agnostic**, meaning it should successfully match long-form academic essays to short-form social media posts by the same author by isolating invariant stylistic habits.

---

## 2. Component Architecture

### A. `processors.py` (The Linguistic Foundation)
Implement a `TextProcessor` class with the following methods:

* **`mask_pii(text: str) -> str`**: Use `spaCy` Named Entity Recognition (NER) to identify and replace `PERSON`, `ORG`, and `GPE` entities with bracketed placeholders (e.g., `[ENTITY]`).
* **`get_mattr(text: str, window: int = 50) -> float`**: Implement **Moving-Average Type-Token Ratio**.
    * *Constraint:* Use a sliding window to calculate TTR. If `len(tokens) < window`, return the standard TTR for the entire string to avoid division-by-zero or null errors.
* **`extract_base_metrics(text: str) -> dict`**: Return a dictionary containing:
    * `flesch_kincaid_grade` and `automated_readability_index` via `textstat`.
    * Average word length and average sentence length.
    * Punctuation-to-word ratio (specifically tracking semicolons, em-dashes, and ellipses).

### B. `engine.py` (The ML & Feature Vector Logic)
Implement a `StylometricEngine` class:

* **Feature Vectorization**: Create a unified pipeline that concatenates:
    1.  **Syntactic Features**: Normalized Part-of-Speech (POS) distribution (Count of each tag / Total tokens).
    2.  **Lexical Features**: `TfidfVectorizer` for character n-grams (range 2-4, max_features=500).
    3.  **Complexity Features**: Scaled results from `TextProcessor`.
* **Adversarial Feature Selection**: Implement `select_invariant_features(X, y_author, y_medium)`.
    * Use `sklearn.feature_selection.mutual_info_classif`.
    * **Logic**: Calculate the Invariance Score:
        $$Score = \frac{MI(Feature, Author)}{MI(Feature, Medium) + \epsilon}$$
    * Select the top $k$ features that maximize author-identity while minimizing platform-bias.
* **Model**: Use a `RandomForestClassifier` (n_estimators=100) for the primary classification task.

### C. `main.py` (The API Layer)
Implement a **FastAPI** application:

* **`POST /enroll`**: Accepts a JSON body with `author_id` and a list of texts. Extracts features and stores the mean "Fingerprint Vector" in a local registry.
* **`POST /identify`**: Accepts an unknown text string. Returns the top 3 potential author matches with a probability score (via `predict_proba`).
* **Visualizer**: A utility function using `Plotly` to generate a **Radar Chart** comparing the unknown text's feature vector against the predicted author's mean profile.

---

## 3. Implementation Constraints

* **Normalization**: All numerical features (grades, ratios, lengths) must be processed through a `StandardScaler` or `MinMaxScaler` before entering the classifier.
* **Input Validation**: API must return `400 Bad Request` if the input text contains fewer than 10 tokens, as stylometric variance is statistically insignificant below this threshold.
* **Persistence**: The engine must save the trained model, the vectorizer state, and the scaler to the `models/checkpoints/` directory using `joblib`.

---

## 4. Expected Deliverables

1.  `processors.py`, `engine.py`, and `main.py` following the logic above.
2.  A `requirements.txt` file with version-pinned dependencies.
3.  A brief `README.md` explaining the Adversarial Selection logic.