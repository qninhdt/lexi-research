"""Distinct-n over a batch of sentences — the measurement that verifies batching
did not collapse.

Batching K sentences into one call is what makes generation affordable, and it is
also the one thing most likely to quietly ruin the dataset: a model asked for six
learner sentences about the same word tends to write one sentence six times with a
few words swapped. Those rows cost six calls and carry one row's worth of signal.

`distinct-n` is the cheapest measurement that catches it: unique n-grams over
total n-grams, across the batch as a whole. 1.0 means every n-gram appears once;
a batch of six copies of the same sentence scores ~1/6.

Batches below threshold are flagged in the report rather than discarded. The
number's first job is to tell us whether call 1's prompt needs work — throwing the
rows away would hide the signal that produced them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: Word-ish tokens, lowercased. Punctuation is dropped: two sentences differing
#: only in a comma are not meaningfully diverse, and counting punctuation as
#: tokens would say they were.
_TOKEN_RE = re.compile(r"[a-z0-9']+")

#: Default distinct-2 floor. Below it, a batch reads as paraphrases of one
#: sentence rather than K independent attempts.
DEFAULT_THRESHOLD = 0.7


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """All n-grams of `tokens`, in order. Empty when the text is shorter than `n`."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Unique n-grams over total n-grams, pooled across `texts`.

    Pooled rather than averaged per sentence: the failure being measured is
    repetition *between* sentences, and a per-sentence average would score six
    identical sentences as perfectly diverse.

    Returns 1.0 for an empty pool — no evidence of collapse is not evidence of
    collapse, and a 0.0 would flag every batch of very short sentences.
    """
    total: list[tuple[str, ...]] = []
    for text in texts:
        total.extend(ngrams(tokenize(text), n))
    if not total:
        return 1.0
    return len(set(total)) / len(total)


@dataclass(frozen=True)
class BatchDiversity:
    """Distinct-2 and distinct-3 for one call-1 batch.

    Carries `batch_uid` so a flagged batch can be looked up and read: the number
    on its own says a batch collapsed, but only the text says why the prompt let
    it.
    """

    batch_uid: str
    texts: int
    distinct2: float
    distinct3: float
    threshold: float

    @property
    def is_collapsed(self) -> bool:
        """True when the batch reads as paraphrases of one sentence.

        An empty batch is never flagged: it failed for other reasons that are
        already counted, and double-reporting it would make the diversity number
        a proxy for the rejection rate.
        """
        return self.texts > 0 and self.distinct2 < self.threshold

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "batch_uid": self.batch_uid,
            "texts": self.texts,
            "distinct2": round(self.distinct2, 4),
            "distinct3": round(self.distinct3, 4),
            "collapsed": self.is_collapsed,
        }


def batch_diversity(
    batch_uid: str,
    texts: Sequence[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> BatchDiversity:
    """Measure one batch.

    The gate is on distinct-2 alone. Distinct-3 is reported because it is
    strictly more sensitive, but gating on both would reject batches for sharing
    ordinary English trigrams ("I want to") that say nothing about diversity.
    """
    return BatchDiversity(
        batch_uid=batch_uid,
        texts=len(texts),
        distinct2=distinct_n(texts, 2),
        distinct3=distinct_n(texts, 3),
        threshold=threshold,
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "BatchDiversity",
    "batch_diversity",
    "distinct_n",
    "ngrams",
    "tokenize",
]
