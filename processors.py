"""
processors.py — Linguistic Foundation
Handles PII masking, MATTR calculation, and base stylometric metrics.
"""

import re
import string
from typing import Optional

import spacy
import textstat

# Load spaCy model once at module level for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )


class TextProcessor:
    """
    Core NLP preprocessing and feature extraction.
    All methods are stateless — safe to share across threads.
    """

    # PII entity types to redact
    PII_ENTITY_TYPES = {"PERSON", "ORG", "GPE"}

    # Punctuation patterns to track for stylometric analysis
    SEMICOLON_RE = re.compile(r";")
    EM_DASH_RE = re.compile(r"—|--")
    ELLIPSIS_RE = re.compile(r"\.\.\.|…")

    def mask_pii(self, text: str) -> str:
        """
        Identify and replace PERSON, ORG, and GPE named entities
        with the [ENTITY] placeholder using spaCy NER.

        Args:
            text: Raw input string.

        Returns:
            Text with PII entities replaced by [ENTITY].
        """
        if not text or not text.strip():
            return text

        doc = nlp(text)

        # Build replacement spans in reverse order to preserve character offsets
        masked = text
        replacements = [
            (ent.start_char, ent.end_char)
            for ent in doc.ents
            if ent.label_ in self.PII_ENTITY_TYPES
        ]

        for start, end in sorted(replacements, reverse=True):
            masked = masked[:start] + "[ENTITY]" + masked[end:]

        return masked

    def get_mattr(self, text: str, window: int = 50) -> float:
        """
        Compute Moving-Average Type-Token Ratio (MATTR).

        MATTR is more robust than standard TTR for texts of varying length
        because it averages TTR over sliding windows rather than the full text.

        Args:
            text:   Input string.
            window: Sliding window size (default 50 tokens).

        Returns:
            MATTR score in [0.0, 1.0]. Returns 0.0 for empty input.
        """
        tokens = self._tokenize(text)
        n = len(tokens)

        if n == 0:
            return 0.0

        # Fallback to standard TTR when text is shorter than the window
        if n < window:
            return len(set(tokens)) / n

        # Sliding window MATTR
        ttrs = []
        for i in range(n - window + 1):
            window_tokens = tokens[i : i + window]
            ttr = len(set(window_tokens)) / window
            ttrs.append(ttr)

        return sum(ttrs) / len(ttrs)

    def extract_base_metrics(self, text: str) -> dict:
        """
        Extract a dictionary of scalar stylometric features.

        Returns:
            {
                "flesch_kincaid_grade":         float,
                "automated_readability_index":  float,
                "avg_word_length":              float,
                "avg_sentence_length":          float,
                "semicolon_ratio":              float,
                "em_dash_ratio":                float,
                "ellipsis_ratio":               float,
                "mattr":                        float,
            }
        """
        if not text or not text.strip():
            return self._zero_metrics()

        tokens = self._tokenize(text)
        word_count = len(tokens)

        if word_count == 0:
            return self._zero_metrics()

        doc = nlp(text)
        sentences = list(doc.sents)
        sentence_count = max(len(sentences), 1)

        # Readability scores
        fk_grade = textstat.flesch_kincaid_grade(text)
        ari = textstat.automated_readability_index(text)

        # Average word length (alphabetic tokens only)
        alpha_tokens = [t for t in tokens if t.isalpha()]
        avg_word_length = (
            sum(len(t) for t in alpha_tokens) / len(alpha_tokens)
            if alpha_tokens
            else 0.0
        )

        # Average sentence length in tokens
        avg_sentence_length = word_count / sentence_count

        # Punctuation ratios relative to word count
        semicolon_ratio = len(self.SEMICOLON_RE.findall(text)) / word_count
        em_dash_ratio = len(self.EM_DASH_RE.findall(text)) / word_count
        ellipsis_ratio = len(self.ELLIPSIS_RE.findall(text)) / word_count

        # MATTR
        mattr = self.get_mattr(text)

        return {
            "flesch_kincaid_grade": fk_grade,
            "automated_readability_index": ari,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "semicolon_ratio": semicolon_ratio,
            "em_dash_ratio": em_dash_ratio,
            "ellipsis_ratio": ellipsis_ratio,
            "mattr": mattr,
        }

    def get_pos_distribution(self, text: str) -> dict:
        """
        Return normalized POS tag distribution.
        Each count is divided by total tokens to give a ratio in [0, 1].

        Used by StylometricEngine for syntactic feature construction.
        """
        doc = nlp(text)
        tokens = [token for token in doc if not token.is_space]
        total = len(tokens)

        if total == 0:
            return {}

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token.pos_] = counts.get(token.pos_, 0) + 1

        return {tag: count / total for tag, count in counts.items()}

    def token_count(self, text: str) -> int:
        """Return the number of non-whitespace tokens in text."""
        return len(self._tokenize(text))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return text.split()

    def _zero_metrics(self) -> dict:
        return {
            "flesch_kincaid_grade": 0.0,
            "automated_readability_index": 0.0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "semicolon_ratio": 0.0,
            "em_dash_ratio": 0.0,
            "ellipsis_ratio": 0.0,
            "mattr": 0.0,
        }
