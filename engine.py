"""
engine.py — ML & Feature Vector Logic
Adversarial feature selection, ensemble classification, SHAP explainability.
"""

import os
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

# Suppress SHAP's verbose warnings during tree explainer init
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

CHECKPOINT_DIR = Path("models/checkpoints")
EPSILON = 1e-10  # Avoid division-by-zero in invariance score


class StylometricEngine:
    """
    Full stylometric pipeline:
      1. Feature extraction (syntactic + lexical + complexity)
      2. Adversarial feature selection (maximize author MI, minimize medium MI)
      3. Soft-voting ensemble (RandomForest + SVC) with calibrated probabilities
      4. SHAP-based explainability
    """

    # Expected POS tags (Universal Dependencies tagset used by spaCy)
    ALL_POS_TAGS = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
        "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
        "SCONJ", "SYM", "VERB", "X",
    ]

    def __init__(self):
        self.processor = TextProcessor()
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=500,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.model: Optional[CalibratedClassifierCV] = None
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.feature_names: list[str] = []
        self.shap_values: Optional[np.ndarray] = None          # shape: (n_features,)
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
    ) -> "StylometricEngine":
        """
        Full training pipeline.

        Args:
            texts:          Raw text samples.
            author_labels:  Corresponding author IDs.
            medium_labels:  Optional platform/medium labels for adversarial selection.
                            If None, adversarial selection is skipped.
            top_k_features: Number of invariant features to retain.

        Returns:
            self (for method chaining)
        """
        print(f"[Engine] Extracting features for {len(texts)} samples...")
        X_raw = self._extract_all_features(texts, fit_tfidf=True)
        self._build_feature_names()

        # Adversarial feature selection
        if medium_labels is not None:
            print("[Engine] Running adversarial feature selection...")
            self.selected_feature_indices = self.select_invariant_features(
                X_raw, author_labels, medium_labels, k=top_k_features
            )
            X_selected = X_raw[:, self.selected_feature_indices]
        else:
            self.selected_feature_indices = np.arange(X_raw.shape[1])
            X_selected = X_raw

        # Scale
        X_scaled = self.scaler.fit_transform(X_selected)

        # Build ensemble
        print("[Engine] Training ensemble classifier...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        svc = SVC(kernel="rbf", probability=True, random_state=42)

        voting_clf = VotingClassifier(
            estimators=[("rf", rf), ("svc", svc)],
            voting="soft",
        )

        # Calibrate probabilities (isotonic regression, 5-fold)
        self.model = CalibratedClassifierCV(voting_clf, method="isotonic", cv=5)
        self.model.fit(X_scaled, author_labels)
        self.author_labels = list(self.model.classes_)

        # SHAP explainability on the Random Forest component
        print("[Engine] Computing SHAP values...")
        self._compute_shap(X_scaled, rf, author_labels)

        self._is_fitted = True
        print("[Engine] Training complete.")
        return self

    def predict(self, texts: list[str]) -> list[str]:
        """Return the most likely author for each text."""
        self._assert_fitted()
        X = self._prepare_features(texts)
        return self.model.predict(X).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Return calibrated probability matrix.
        Shape: (n_texts, n_authors)
        """
        self._assert_fitted()
        X = self._prepare_features(texts)
        return self.model.predict_proba(X)

    def top_matches(self, text: str, n: int = 3) -> list[dict]:
        """
        Return the top-n author matches for an unknown text.

        Returns:
            List of dicts: [{"author": str, "probability": float}, ...]
        """
        self._assert_fitted()
        proba = self.predict_proba([text])[0]
        top_indices = np.argsort(proba)[::-1][:n]

        return [
            {
                "author": self.author_labels[i],
                "probability": round(float(proba[i]), 4),
            }
            for i in top_indices
        ]

    def top_shap_features(self, n: int = 5) -> list[dict]:
        """
        Return the top-n most influential features by mean absolute SHAP value.

        Returns:
            List of dicts: [{"feature": str, "importance": float}, ...]
        """
        self._assert_fitted()
        if self.mean_shap_per_feature is None:
            return []

        selected_names = [
            self.feature_names[i] for i in self.selected_feature_indices
        ]
        top_indices = np.argsort(self.mean_shap_per_feature)[::-1][:n]

        return [
            {
                "feature": selected_names[i],
                "importance": round(float(self.mean_shap_per_feature[i]), 6),
            }
            for i in top_indices
        ]

    def select_invariant_features(
        self,
        X: np.ndarray,
        y_author: list[str],
        y_medium: list[str],
        k: int = 200,
    ) -> np.ndarray:
        """
        Select features that strongly predict author identity but are
        weakly correlated with the text medium/platform.

        Invariance Score = MI(Feature, Author) / (MI(Feature, Medium) + ε)

        Args:
            X:        Feature matrix, shape (n_samples, n_features).
            y_author: Author label per sample.
            y_medium: Medium/platform label per sample.
            k:        Number of top features to return.

        Returns:
            Array of selected feature indices.
        """
        mi_author = mutual_info_classif(X, y_author, discrete_features=False, random_state=42)
        mi_medium = mutual_info_classif(X, y_medium, discrete_features=False, random_state=42)

        invariance_scores = mi_author / (mi_medium + EPSILON)

        top_k = min(k, X.shape[1])
        selected = np.argsort(invariance_scores)[::-1][:top_k]
        return selected

    def get_feature_vector(self, text: str) -> np.ndarray:
        """
        Return the scaled, selected feature vector for a single text.
        Used by the API for radar chart comparisons.
        """
        self._assert_fitted()
        return self._prepare_features([text])[0]

    def save(self) -> None:
        """Persist model, vectorizer, scaler, and SHAP values to disk."""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, CHECKPOINT_DIR / "model.joblib")
        joblib.dump(self.tfidf, CHECKPOINT_DIR / "tfidf.joblib")
        joblib.dump(self.scaler, CHECKPOINT_DIR / "scaler.joblib")
        joblib.dump(self.selected_feature_indices, CHECKPOINT_DIR / "feature_indices.joblib")
        joblib.dump(self.mean_shap_per_feature, CHECKPOINT_DIR / "shap_values.joblib")
        joblib.dump(self.feature_names, CHECKPOINT_DIR / "feature_names.joblib")
        joblib.dump(self.author_labels, CHECKPOINT_DIR / "author_labels.joblib")
        print(f"[Engine] Checkpoints saved to {CHECKPOINT_DIR}/")

    def load(self) -> "StylometricEngine":
        """Restore engine state from disk."""
        self.model = joblib.load(CHECKPOINT_DIR / "model.joblib")
        self.tfidf = joblib.load(CHECKPOINT_DIR / "tfidf.joblib")
        self.scaler = joblib.load(CHECKPOINT_DIR / "scaler.joblib")
        self.selected_feature_indices = joblib.load(CHECKPOINT_DIR / "feature_indices.joblib")
        self.mean_shap_per_feature = joblib.load(CHECKPOINT_DIR / "shap_values.joblib")
        self.feature_names = joblib.load(CHECKPOINT_DIR / "feature_names.joblib")
        self.author_labels = joblib.load(CHECKPOINT_DIR / "author_labels.joblib")
        self._is_fitted = True
        print("[Engine] Checkpoints loaded.")
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_all_features(
        self, texts: list[str], fit_tfidf: bool = False
    ) -> np.ndarray:
        """
        Concatenate:
          [POS distribution | char n-gram TF-IDF | complexity scalars]
        Returns raw (unscaled) feature matrix.
        """
        pos_features = np.array([self._pos_vector(t) for t in texts])

        if fit_tfidf:
            tfidf_features = self.tfidf.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf.transform(texts).toarray()

        complexity_features = np.array([
            list(self.processor.extract_base_metrics(t).values())
            for t in texts
        ])

        return np.hstack([pos_features, tfidf_features, complexity_features])

    def _prepare_features(self, texts: list[str]) -> np.ndarray:
        """Extract → select → scale features for inference."""
        X_raw = self._extract_all_features(texts, fit_tfidf=False)
        X_selected = X_raw[:, self.selected_feature_indices]
        return self.scaler.transform(X_selected)

    def _pos_vector(self, text: str) -> np.ndarray:
        """Return a fixed-length normalized POS distribution vector."""
        dist = self.processor.get_pos_distribution(text)
        return np.array([dist.get(tag, 0.0) for tag in self.ALL_POS_TAGS])

    def _build_feature_names(self) -> None:
        """Build human-readable names for every feature dimension."""
        pos_names = [f"pos_{tag}" for tag in self.ALL_POS_TAGS]
        tfidf_names = [f"ngram_{v}" for v in self.tfidf.get_feature_names_out()]
        complexity_names = list(self.processor.extract_base_metrics("placeholder").keys())
        self.feature_names = pos_names + tfidf_names + complexity_names

    def _compute_shap(
        self,
        X_scaled: np.ndarray,
        rf: RandomForestClassifier,
        author_labels: list[str],
    ) -> None:
        """
        Fit a TreeExplainer on the Random Forest and store mean |SHAP| per feature.
        We fit the RF separately here to get SHAP values before calibration wraps it.
        """
        # Re-fit the RF alone on the already-scaled data (lightweight — same data)
        rf.fit(X_scaled, author_labels)

        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_scaled)  # list of arrays per class

        # Stack all classes, take absolute values, mean over samples and classes
        if isinstance(shap_vals, list):
            stacked = np.abs(np.array(shap_vals))   # (n_classes, n_samples, n_features)
            self.mean_shap_per_feature = stacked.mean(axis=(0, 1))
        else:
            self.mean_shap_per_feature = np.abs(shap_vals).mean(axis=0)

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Engine is not fitted. Call .fit() or .load() first."
            )
