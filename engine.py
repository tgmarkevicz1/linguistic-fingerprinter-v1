"""
engine.py
---------
Stylometric feature extraction, adversarial feature selection,
ensemble classification, and SHAP-based explainability.

Design decisions documented inline. For the architectural rationale,
see README.md § Key Architectural Decisions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from processors import TextProcessor

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("models/checkpoints")

# Small epsilon prevents division-by-zero in the invariance score denominator.
# Value chosen to be negligible relative to typical MI magnitudes (~0.01–1.0).
_EPSILON = 1e-10

# Universal Dependencies POS tagset used by spaCy's en_core_web_sm model.
# Fixed ordering ensures the POS feature vector is consistent across calls.
_POS_TAGS = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
)


# ---------------------------------------------------------------------------
# StylometricEngine
# ---------------------------------------------------------------------------

class StylometricEngine:
    """
    End-to-end authorship attribution pipeline.

    Pipeline stages (in order):
        1. Feature extraction  — POS distributions + char n-gram TF-IDF + complexity scalars
        2. Adversarial selection — retain features that predict author, not medium
        3. Scaling             — StandardScaler applied to selected features
        4. Ensemble training   — RandomForest + SVC with calibrated soft voting
        5. Explainability      — SHAP TreeExplainer over the RF component

    Usage:
        engine = StylometricEngine()
        engine.fit(texts, authors, medium_labels=mediums)
        engine.save()

        # Later:
        engine = StylometricEngine().load()
        matches = engine.top_matches("Some anonymous text...", n=3)
    """

    def __init__(self) -> None:
        self.processor = TextProcessor()

        # Char n-grams (2–4) capture morphological habits and punctuation sequences
        # that persist across text formats. Word n-grams collapse on short texts;
        # char n-grams don't.
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=500,
            sublinear_tf=True,
        )

        self.scaler = StandardScaler()
        self.model: Optional[CalibratedClassifierCV] = None
        self.selected_indices: Optional[np.ndarray] = None
        self.feature_names: list[str] = []
        self.mean_shap_per_feature: Optional[np.ndarray] = None
        self.author_labels: list[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: list[str],
        author_labels: list[str],
        medium_labels: Optional[list[str]] = None,
        top_k_features: int = 200,
    ) -> StylometricEngine:
        """
        Train the full pipeline on labelled text samples.

        Args:
            texts:           Raw text samples (PII masking is the caller's responsibility).
            author_labels:   Author ID per sample. Must be same length as texts.
            medium_labels:   Platform/format label per sample (e.g. "blog", "tweet").
                             If provided, adversarial feature selection is applied.
                             If None, all features are retained.
            top_k_features:  Number of invariant features to keep after selection.

        Returns:
            self — supports method chaining.
        """
        print(f"[Engine] Extracting features for {len(texts)} samples...")
        X_raw = self._extract_features(texts, fit_tfidf=True)
        self._build_feature_names()

        if medium_labels is not None:
            print("[Engine] Running adversarial feature selection...")
            self.selected_indices = self._select_invariant_features(
                X_raw, author_labels, medium_labels, k=top_k_features
            )
        else:
            self.selected_indices = np.arange(X_raw.shape[1])

        X_selected = X_raw[:, self.selected_indices]
        X_scaled = self.scaler.fit_transform(X_selected)

        print("[Engine] Training ensemble classifier...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        svc = SVC(kernel="rbf", probability=True, random_state=42)

        # Soft voting averages predicted probabilities across classifiers.
        # RandomForest provides stability; SVC provides strong margins.
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("svc", svc)],
            voting="soft",
        )

        # Isotonic calibration corrects RF overconfidence so predict_proba
        # outputs reflect real posterior probabilities, not vote counts.
        self.model = CalibratedClassifierCV(ensemble, method="isotonic", cv=5)
        self.model.fit(X_scaled, author_labels)
        self.author_labels = list(self.model.classes_)

        print("[Engine] Computing SHAP values...")
        self._compute_shap(X_scaled, rf, author_labels)

        self._is_fitted = True
        print("[Engine] Training complete.")
        return self

    def predict(self, texts: list[str]) -> list[str]:
        """Return the top-1 predicted author for each text."""
        self._assert_fitted()
        return self.model.predict(self._prepare_for_inference(texts)).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Return calibrated probability matrix of shape (n_texts, n_authors).

        Probabilities are isotonic-regression calibrated — they reflect
        actual posterior estimates, not raw ensemble vote proportions.
        """
        self._assert_fitted()
        return self.model.predict_proba(self._prepare_for_inference(texts))

    def top_matches(self, text: str, n: int = 3) -> list[dict]:
        """
        Return the top-n author matches with calibrated probabilities.

        Returns:
            [{"author": str, "probability": float}, ...]  sorted by probability desc.
        """
        self._assert_fitted()
        proba = self.predict_proba([text])[0]
        top_indices = np.argsort(proba)[::-1][:n]
        return [
            {"author": self.author_labels[i], "probability": round(float(proba[i]), 4)}
            for i in top_indices
        ]

    def top_shap_features(self, n: int = 5) -> list[dict]:
        """
        Return the top-n features by mean absolute SHAP value.

        SHAP values are computed over the RandomForest component at training
        time. Mean |SHAP| across all samples and classes gives a
        model-level importance score that is comparable across features.

        Returns:
            [{"feature": str, "importance": float}, ...]  sorted by importance desc.
        """
        self._assert_fitted()
        if self.mean_shap_per_feature is None:
            return []

        selected_names = [self.feature_names[i] for i in self.selected_indices]
        top_indices = np.argsort(self.mean_shap_per_feature)[::-1][:n]
        return [
            {
                "feature": selected_names[i],
                "importance": round(float(self.mean_shap_per_feature[i]), 6),
            }
            for i in top_indices
        ]

    def get_feature_vector(self, text: str) -> np.ndarray:
        """
        Return the scaled, selected feature vector for a single text.

        Used by the API to build radar chart comparisons between
        an unknown text and an enrolled author's mean profile.
        """
        self._assert_fitted()
        return self._prepare_for_inference([text])[0]

    def select_invariant_features(
        self,
        X: np.ndarray,
        y_author: list[str],
        y_medium: list[str],
        k: int = 200,
    ) -> np.ndarray:
        """
        Public wrapper around adversarial feature selection.

        Exposed as a public method so train.py can call it directly
        for evaluation or ablation studies.

        See _select_invariant_features for implementation details.
        """
        return self._select_invariant_features(X, y_author, y_medium, k)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Persist all model state to CHECKPOINT_DIR.

        Saved artifacts:
            model.joblib          — CalibratedClassifierCV (ensemble + calibration)
            tfidf.joblib          — Fitted TfidfVectorizer
            scaler.joblib         — Fitted StandardScaler
            feature_indices.joblib — Selected feature indices (post adversarial selection)
            shap_values.joblib    — Mean |SHAP| per selected feature
            feature_names.joblib  — Human-readable names for all features (pre-selection)
            author_labels.joblib  — Ordered list of author classes
        """
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "model": self.model,
            "tfidf": self.tfidf,
            "scaler": self.scaler,
            "feature_indices": self.selected_indices,
            "shap_values": self.mean_shap_per_feature,
            "feature_names": self.feature_names,
            "author_labels": self.author_labels,
        }
        for name, obj in artifacts.items():
            joblib.dump(obj, CHECKPOINT_DIR / f"{name}.joblib")
        print(f"[Engine] Saved {len(artifacts)} checkpoints to {CHECKPOINT_DIR}/")

    def load(self) -> StylometricEngine:
        """
        Restore engine state from CHECKPOINT_DIR.

        Raises FileNotFoundError if any checkpoint is missing.
        """
        self.model = joblib.load(CHECKPOINT_DIR / "model.joblib")
        self.tfidf = joblib.load(CHECKPOINT_DIR / "tfidf.joblib")
        self.scaler = joblib.load(CHECKPOINT_DIR / "scaler.joblib")
        self.selected_indices = joblib.load(CHECKPOINT_DIR / "feature_indices.joblib")
        self.mean_shap_per_feature = joblib.load(CHECKPOINT_DIR / "shap_values.joblib")
        self.feature_names = joblib.load(CHECKPOINT_DIR / "feature_names.joblib")
        self.author_labels = joblib.load(CHECKPOINT_DIR / "author_labels.joblib")
        self._is_fitted = True
        print("[Engine] Checkpoints loaded.")
        return self

    # ------------------------------------------------------------------
    # Private: feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self, texts: list[str], fit_tfidf: bool = False
    ) -> np.ndarray:
        """
        Build the full feature matrix by concatenating three feature groups:

            [POS distribution (17) | char n-gram TF-IDF (500) | complexity scalars (8)]

        Total dimensions before selection: 525.

        Args:
            texts:      List of raw text strings.
            fit_tfidf:  If True, fits the TF-IDF vectorizer on this data (training only).
                        If False, uses the already-fitted vectorizer (inference).

        Returns:
            Unscaled feature matrix of shape (n_texts, 525).
        """
        pos_features = np.array([self._pos_vector(t) for t in texts])

        tfidf_features = (
            self.tfidf.fit_transform(texts).toarray()
            if fit_tfidf
            else self.tfidf.transform(texts).toarray()
        )

        complexity_features = np.array([
            list(self.processor.extract_base_metrics(t).values())
            for t in texts
        ])

        return np.hstack([pos_features, tfidf_features, complexity_features])

    def _prepare_for_inference(self, texts: list[str]) -> np.ndarray:
        """Extract features, apply selection mask, and scale — for inference only."""
        X_raw = self._extract_features(texts, fit_tfidf=False)
        return self.scaler.transform(X_raw[:, self.selected_indices])

    def _pos_vector(self, text: str) -> np.ndarray:
        """
        Return a fixed-length normalized POS distribution vector.

        Normalized by total token count so the vector is comparable
        across texts of different lengths.
        """
        dist = self.processor.get_pos_distribution(text)
        return np.array([dist.get(tag, 0.0) for tag in _POS_TAGS])

    def _build_feature_names(self) -> None:
        """
        Construct human-readable names for every feature dimension.

        Called once after TF-IDF is fitted so get_feature_names_out() is available.
        These names power the SHAP feature labels in the API response.
        """
        pos_names = [f"pos_{tag}" for tag in _POS_TAGS]
        tfidf_names = [f"ngram_{v}" for v in self.tfidf.get_feature_names_out()]
        complexity_names = list(self.processor.extract_base_metrics("placeholder").keys())
        self.feature_names = pos_names + tfidf_names + complexity_names

    # ------------------------------------------------------------------
    # Private: adversarial feature selection
    # ------------------------------------------------------------------

    def _select_invariant_features(
        self,
        X: np.ndarray,
        y_author: list[str],
        y_medium: list[str],
        k: int,
    ) -> np.ndarray:
        """
        Select the k features that most strongly predict author identity
        while being least correlated with text medium.

        Invariance Score = MI(Feature, Author) / (MI(Feature, Medium) + ε)

        A high score means the feature is author-discriminative but
        medium-agnostic — exactly what cross-medium attribution needs.

        A low score means the feature mostly encodes platform or format
        (e.g. vocabulary complexity in long-form academic writing), which
        would cause the model to learn medium identity, not author identity.

        Args:
            X:        Feature matrix, shape (n_samples, n_features).
            y_author: Author label per sample.
            y_medium: Medium/platform label per sample.
            k:        Number of top-scoring features to retain.

        Returns:
            Array of selected feature indices, shape (min(k, n_features),).
        """
        mi_author = mutual_info_classif(X, y_author, discrete_features=False, random_state=42)
        mi_medium = mutual_info_classif(X, y_medium, discrete_features=False, random_state=42)
        invariance_scores = mi_author / (mi_medium + _EPSILON)
        return np.argsort(invariance_scores)[::-1][: min(k, X.shape[1])]

    # ------------------------------------------------------------------
    # Private: SHAP explainability
    # ------------------------------------------------------------------

    def _compute_shap(
        self,
        X_scaled: np.ndarray,
        rf: RandomForestClassifier,
        author_labels: list[str],
    ) -> None:
        """
        Fit a SHAP TreeExplainer on the RandomForest component and store
        mean absolute SHAP values per feature.

        The RF is re-fitted here (on already-scaled data) because
        CalibratedClassifierCV wraps the estimator in a way that prevents
        direct access to the underlying tree structure that SHAP requires.
        The re-fit is fast since the data is identical.

        mean_shap_per_feature[i] answers: "On average, how much does
        feature i move any prediction away from the base rate?"
        """
        rf.fit(X_scaled, author_labels)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled)

        # shap_values is a list of (n_samples, n_features) arrays, one per class.
        # Stack, take absolute values, and mean across both samples and classes.
        if isinstance(shap_values, list):
            stacked = np.abs(np.array(shap_values))  # (n_classes, n_samples, n_features)
            self.mean_shap_per_feature = stacked.mean(axis=(0, 1))
        else:
            self.mean_shap_per_feature = np.abs(shap_values).mean(axis=0)

    # ------------------------------------------------------------------
    # Private: guards
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Engine is not fitted. Call .fit() or .load() first."
            )
