"""Stage A: convert a human-annotated learner corpus into the inline edit format.

The teacher-generated dataset is small and expensive; this stage is neither. It
reads W&I+LOCNESS — 34,308 sentences of real learner writing, annotated by
Cambridge assessors and typed by ERRANT — and re-expresses each one in the
project's own `[A>B:tag]` markup. No teacher call is made, so the whole stage
costs nothing and produces roughly twenty times the rows.

What it deliberately does *not* produce is `meaning` or `feedback`. Those depend
on a target word and a dictionary sense that a general learner corpus has no
notion of, and inventing them would be fabricating labels. Stage A therefore
teaches the correction format and the tag set; stage B teaches the rest.

Four properties do the real work:

**Round-trip or reject.** Every row is rendered with `parser.render` and then
read back with `parse_correction`; a row whose stripped correction is not
byte-identical to its text is dropped, not repaired. This is the same invariant
`validate_output` check 3 enforces on model output, applied to our own conversion
so a detokenisation bug cannot enter the dataset as training signal.

**Detokenisation of both sides.** M2 stores space-separated tokens (`stores .`),
which no learner writes. Text and replacement strings both go through `detokenise`
— the replacement side matters because 2.26% of them carry punctuation spacing
(`'. However ,'`) that would otherwise be taught to the model verbatim.

**`UNK` drops the edit, not the sentence.** 1,649 annotations are typed `UNK`,
and sampling them shows they are no-ops: `'chess' -> 'chess'`. Discarding the
sentence would lose 4.4% of the corpus for annotations that assert nothing.

**`R:OTHER` is split, not passed through.** It is 9.3% of all edits — far above
the 5% that gate G5 treats as evidence the taxonomy is missing a category — and
71% of it is multi-token rewrites (`'more memories bring me'` -> `'brought me the
most memories'`), which is `unnat` as `tags.py` defines it. The split is a
heuristic over edit length, not an annotator judgement, and is recorded as such.

`coll` never appears here: ERRANT has no collocation type and no honest heuristic
recovers one. It stays in the taxonomy and is taught in stage B, where the
teacher can see the target sense.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexi_research.format.parser import Edit, ParseError, parse_correction, render
from lexi_research.format.tags import TAGS

#: ERRANT type (minus its M/U/R operation prefix) -> one of the 16 tags.
#:
#: The prefix is dropped rather than mapped: which operation an edit performs is
#: already carried by which side of `[A>B:tag]` is empty, so folding it into the
#: tag would double the taxonomy against a closed set that must not grow.
#:
#: `OTHER` is absent on purpose — it is resolved by `_split_other`, not looked up.
ERRANT_TO_TAG: dict[str, str] = {
    "SPELL": "sp",
    "ORTH": "sp",
    "CONTR": "sp",
    "VERB:SVA": "agr",
    "VERB:TENSE": "tense",
    "VERB:FORM": "form",
    "VERB:INFL": "form",
    "ADJ:FORM": "form",
    "MORPH": "form",
    "NOUN:INFL": "form",
    "DET": "art",
    "PREP": "prep",
    "PART": "part",
    "PRON": "pron",
    "WO": "order",
    "PUNCT": "punc",
    "NOUN:NUM": "num",
    "NOUN:POSS": "poss",
    "NOUN": "word",
    "VERB": "word",
    "ADJ": "word",
    "ADV": "word",
    "CONJ": "word",
}

#: ERRANT types that carry no assertion and are skipped at the edit level.
#:
#: `noop` marks a sentence an annotator left alone. `UNK` marks a span they could
#: not type — and in this corpus its correction equals its original, so it is a
#: no-op wearing a different name. Both drop the edit and keep the sentence.
EMPTY_TYPES: frozenset[str] = frozenset({"noop", "UNK"})

#: Tokens that attach to the preceding word with no space.
_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}")

#: Tokens after which no space is emitted.
_NO_SPACE_AFTER = frozenset("([{$#")

#: Contractions and possessives: `n't`, `'s`, `'re`, `'ve`, `'ll`, `'d`, `'m`.
_CLITIC = re.compile(r"^(n't|'\w{1,2})$")

#: Buckets for the stratum key. `balance_rows` groups on `(meaning, error_spec)`,
#: which stage A rows do not have, so it needs a key of its own — and edit count
#: is the axis that matters here: 34% of the corpus is already correct, and a
#: sentence re-emitted verbatim is the cheapest thing a model can learn.
EDIT_BUCKETS: tuple[str, ...] = ("clean", "one", "few", "many")


def edit_bucket(n_edits: int) -> str:
    """Which stratum an edit count falls in."""
    if n_edits == 0:
        return "clean"
    if n_edits == 1:
        return "one"
    return "few" if n_edits <= 3 else "many"


class GecImportError(ValueError):
    """The corpus could not be read, or produced nothing usable."""


@dataclass
class ImportStats:
    """Why rows were kept or lost, for `gec-import-report.json`."""

    sentences: int = 0
    converted: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    tag_counts: dict[str, int] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    cefr_counts: dict[str, int] = field(default_factory=dict)
    skipped_edits: dict[str, int] = field(default_factory=dict)
    other_split: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def keep(self, tags: Sequence[str], bucket: str, cefr: str) -> None:
        self.converted += 1
        self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1
        self.cefr_counts[cefr] = self.cefr_counts.get(cefr, 0) + 1
        for tag in tags:
            self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1

    @property
    def conversion_rate(self) -> float:
        return self.converted / self.sentences if self.sentences else 0.0

    @property
    def other_tag_share(self) -> float:
        """Share of `other` among all tags — the same measure as pilot gate G5."""
        total = sum(self.tag_counts.values())
        return self.tag_counts.get("other", 0) / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sentences": self.sentences,
            "converted": self.converted,
            "conversion_rate": round(self.conversion_rate, 4),
            "dropped": dict(sorted(self.dropped.items())),
            "skipped_edits": dict(sorted(self.skipped_edits.items())),
            "other_split": dict(sorted(self.other_split.items())),
            "tag_counts": dict(sorted(self.tag_counts.items())),
            "bucket_counts": dict(sorted(self.bucket_counts.items())),
            "cefr_counts": dict(sorted(self.cefr_counts.items())),
            "other_tag_share": round(self.other_tag_share, 4),
        }


@dataclass(frozen=True)
class M2Sentence:
    """One `S` line and the `A` lines that follow it."""

    tokens: tuple[str, ...]
    annotations: tuple[tuple[int, int, str, str, str], ...]
    """`(start, end, errant_type, correction, coder)` per annotation."""


def detokenise(tokens: Sequence[str]) -> tuple[str, tuple[int, ...]]:
    """Join M2 tokens into natural text, with each token's start offset.

    Offsets are returned rather than recomputed by the caller because an edit's
    character span has to be derived from the *same* join that produced the text.
    Re-deriving it by searching for the token would land on the wrong occurrence
    whenever a word repeats.

    The returned tuple has one entry per token plus a final end-of-text offset,
    so `offsets[i]` is the start of token `i` and `offsets[len(tokens)]` is the
    length of the text.
    """
    parts: list[str] = []
    offsets: list[int] = []
    position = 0
    previous: str | None = None

    for token in tokens:
        space = True
        if previous is None:
            space = False
        elif token and token[0] in _NO_SPACE_BEFORE:
            space = False
        elif _CLITIC.match(token):
            space = False
        elif previous and previous[-1] in _NO_SPACE_AFTER:
            space = False
        if space:
            parts.append(" ")
            position += 1
        offsets.append(position)
        parts.append(token)
        position += len(token)
        previous = token

    offsets.append(position)
    return "".join(parts), tuple(offsets)


def detokenise_text(text: str) -> str:
    """Detokenise a space-separated string — used on M2 correction strings.

    2.26% of replacements carry token spacing (`'. However ,'`). Left alone they
    would enter the markup and teach the model to emit that spacing.
    """
    return detokenise(text.split(" "))[0] if text else text


def read_m2(path: str | Path) -> Iterator[M2Sentence]:
    """Stream an M2 file into sentences with their annotations."""
    tokens: tuple[str, ...] | None = None
    annotations: list[tuple[int, int, str, str, str]] = []

    with Path(path).open(encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if line.startswith("S "):
                if tokens is not None:
                    yield M2Sentence(tokens, tuple(annotations))
                tokens, annotations = tuple(line[2:].split(" ")), []
            elif line.startswith("A "):
                if tokens is None:
                    raise GecImportError(f"{path}:{number}: an A line precedes any S line")
                fields = line[2:].split("|||")
                if len(fields) < 6:
                    raise GecImportError(f"{path}:{number}: malformed annotation")
                span, errant_type, correction = fields[0], fields[1], fields[2]
                bounds = span.split()
                if len(bounds) != 2:
                    raise GecImportError(f"{path}:{number}: malformed span {span!r}")
                annotations.append(
                    (int(bounds[0]), int(bounds[1]), errant_type, correction, fields[5])
                )

    if tokens is not None:
        yield M2Sentence(tokens, tuple(annotations))


def _split_other(original: str, replacement: str) -> str:
    """Resolve `R:OTHER`/`M:OTHER`/`U:OTHER` into `unnat` or `word`.

    A multi-token rewrite is phrasing rather than a single wrong word, which is
    what `unnat` means. This is a length heuristic standing in for an annotator
    judgement the corpus does not record: some multi-token OTHER edits are really
    grammatical, and they will be mislabelled. The alternative — leaving all of
    `R:OTHER` as `other` — puts 13% of tags in the catch-all, which is the signal
    gate G5 uses to say the taxonomy is broken, and would be just as wrong while
    also being uninformative.
    """
    return "unnat" if len(original.split()) > 1 or len(replacement.split()) > 1 else "word"


def tag_for(errant_type: str, original: str, replacement: str) -> str | None:
    """The tag for an ERRANT type, or None if it maps to nothing.

    `None` means the caller should drop the whole sentence: an unmapped type is a
    corpus the taxonomy does not cover, and silently discarding the edit would
    present a corrected sentence as if it were already correct.
    """
    key = errant_type.split(":", 1)[1] if ":" in errant_type else errant_type
    if key == "OTHER":
        return _split_other(original, replacement)
    return ERRANT_TO_TAG.get(key)


def convert_sentence(
    sentence: M2Sentence,
    *,
    coder: str = "0",
    stats: ImportStats | None = None,
) -> tuple[str, str, list[str]] | None:
    """One M2 sentence to `(text, correction, tags)`, or None if unusable.

    `coder` selects a single annotator. W&I train has exactly one per document,
    but merging annotators would produce overlapping edits that cannot render.
    """
    tokens = sentence.tokens
    text, offsets = detokenise(tokens)

    edits: list[tuple[int, int, str, str]] = []
    for start, end, errant_type, correction, annotator in sentence.annotations:
        if annotator != coder:
            continue
        if errant_type in EMPTY_TYPES:
            if stats is not None:
                stats.skipped_edits[errant_type] = stats.skipped_edits.get(errant_type, 0) + 1
            continue
        if not 0 <= start <= end <= len(tokens):
            if stats is not None:
                stats.drop("span_out_of_range")
            return None

        original = " ".join(tokens[start:end])
        replacement = "" if correction in ("", "-NONE-") else detokenise_text(correction)
        tag = tag_for(errant_type, original, replacement)
        if tag is None:
            if stats is not None:
                stats.drop(f"unmapped_type:{errant_type}")
            return None
        if tag not in TAGS:  # pragma: no cover - the map is closed over TAGS
            raise GecImportError(f"{tag!r} is not in the taxonomy")
        if stats is not None and errant_type.endswith("OTHER"):
            stats.other_split[tag] = stats.other_split.get(tag, 0) + 1
        edits.append((start, end, replacement, tag))

    edits.sort()
    cursor = -1
    for start, end, _, _ in edits:
        if start < cursor:
            if stats is not None:
                stats.drop("overlapping_edits")
            return None
        cursor = end

    rendered: list[Edit] = []
    for start, end, replacement, tag in edits:
        if start == end:
            point = offsets[start]
            rendered.append(
                Edit(original="", replacement=replacement, tag=tag, span=(point, point))
            )
            continue
        span_start = offsets[start]
        span_end = offsets[end - 1] + len(tokens[end - 1])
        rendered.append(
            Edit(
                original=text[span_start:span_end],
                replacement=replacement,
                tag=tag,
                span=(span_start, span_end),
            )
        )

    try:
        correction_text = render(text, rendered)
    except ValueError:
        if stats is not None:
            stats.drop("render_error")
        return None

    # The invariant, checked rather than assumed: what we wrote must read back as
    # exactly the sentence we started from. A detokenisation bug shows up here.
    parsed = parse_correction(correction_text)
    if isinstance(parsed, ParseError):
        if stats is not None:
            stats.drop(f"reparse:{parsed.code}")
        return None
    if parsed.text != text:
        if stats is not None:
            stats.drop("roundtrip_mismatch")
        return None

    return text, correction_text, [tag for _, _, _, tag in edits]


def row_uid(text: str, correction: str) -> str:
    """Stable id for a converted row. Content-addressed, so reruns agree."""
    digest = hashlib.sha256(f"{text}\0{correction}".encode()).hexdigest()
    return digest[:16]


def convert_m2_file(
    path: str | Path,
    *,
    cefr: str,
    stats: ImportStats,
    coder: str = "0",
) -> list[dict[str, Any]]:
    """Convert one M2 file into stage-A rows."""
    rows: list[dict[str, Any]] = []
    for sentence in read_m2(path):
        stats.sentences += 1
        converted = convert_sentence(sentence, coder=coder, stats=stats)
        if converted is None:
            continue
        text, correction, tags = converted
        bucket = edit_bucket(len(tags))
        stats.keep(tags, bucket, cefr)
        rows.append(
            {
                "row_uid": row_uid(text, correction),
                "text": text,
                "correction": correction,
                "tags": sorted(set(tags)),
                "n_edits": len(tags),
                "n_words": len(text.split()),
                "cefr": cefr,
                "edit_bucket": bucket,
                "source": Path(path).name,
            }
        )
    return rows


def dedupe(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact duplicates by `row_uid`, keeping first occurrence.

    Learner corpora repeat short sentences (`Thank you.`), and a duplicate that
    crossed a split would leak.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row["row_uid"])
        if uid in seen:
            continue
        seen.add(uid)
        unique.append(row)
    return unique


def drop_fragments(
    rows: Sequence[dict[str, Any]], *, min_words: int
) -> list[dict[str, Any]]:
    """Drop rows too short to be sentences.

    W&I is essays split into sentences, and the split leaves behind headings,
    salutations, sign-offs, and bare names — `'Svetlana'`, `'PUBLIC TRANSPORT'`,
    `'Take care,'`. They are 2.6% of the corpus and 83% of them need no edit, so
    they land in the `clean` stratum and spend its budget teaching the model to
    echo proper nouns. The task is grading a learner's sentence; a fragment is
    not one.
    """
    return [row for row in rows if int(row["n_words"]) >= min_words]


def cap_strata(
    rows: Sequence[dict[str, Any]],
    *,
    max_stratum_share: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Cap each `edit_bucket` stratum, mirroring `balance_rows`.

    Separate from `balance_rows` because that function keys on `(meaning,
    error_spec)`; both are absent here, so every row would land in one stratum
    and the cap would do nothing. Cap-only, like the original: a rare stratum is
    never discarded, and selection is by seeded hash so a rerun picks the same
    rows.

    The key is the edit bucket alone. Adding `cefr` to it was measured and does
    not work: it splits the corpus into twelve strata, none of which reaches a
    15% cap, so 32% of rows stay `clean` and the cap removes 105 of 33,081 rows.
    Bucket-only levels the four buckets and leaves the CEFR mix intact — the
    seeded hash is uncorrelated with level, so A/B/C move by under three points.
    """
    if not 0 < max_stratum_share <= 1:
        raise ValueError("max_stratum_share must be in (0, 1]")
    cap = max(1, int(len(rows) * max_stratum_share))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["edit_bucket"]), []).append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(f"{seed}|{row['row_uid']}".encode()).hexdigest(),
        )
        selected.extend(ranked[:cap])
    return selected


def split_rows(
    rows: Sequence[dict[str, Any]], *, seed: int, val_share: float
) -> list[dict[str, Any]]:
    """Tag each row `train` or `val`, by hashed `row_uid`.

    There is no test split. Stage A's purpose is to teach the correction format;
    what the project reports is measured on the teacher-graded test set from
    stage B, whose prompt and label set are the ones served. A stage-A test split
    would invite a number that answers a question nobody asked.
    """
    if not 0 < val_share < 1:
        raise ValueError("val_share must be in (0, 1)")
    out: list[dict[str, Any]] = []
    for row in rows:
        bucket = int.from_bytes(
            hashlib.sha256(f"{seed}|{row['row_uid']}".encode()).digest()[:8], "big"
        ) % 10_000
        split = "val" if bucket / 10_000 < val_share else "train"
        out.append({**row, "split": split})
    return out


#: Which M2 files make up the training corpus, and the CEFR level each carries.
#: The dev files are deliberately excluded: they are the only human-annotated
#: real-learner data available, and spending them as training rows would forfeit
#: the one thing that could later turn a fidelity claim into an accuracy claim.
TRAIN_FILES: tuple[tuple[str, str], ...] = (
    ("A.train.gold.bea19.m2", "A"),
    ("B.train.gold.bea19.m2", "B"),
    ("C.train.gold.bea19.m2", "C"),
)


def import_corpus(
    corpus_dir: str | Path,
    *,
    seed: int,
    max_stratum_share: float,
    val_share: float,
    min_words: int = 3,
    min_conversion_rate: float = 0.95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the corpus, convert it, cap strata, and split. Returns rows + report."""
    root = Path(corpus_dir)
    m2_dir = root / "m2"
    if not m2_dir.is_dir():
        raise GecImportError(f"{m2_dir} does not exist; expected the W&I+LOCNESS layout")

    stats = ImportStats()
    raw: list[dict[str, Any]] = []
    for name, cefr in TRAIN_FILES:
        path = m2_dir / name
        if not path.is_file():
            raise GecImportError(f"{path} is missing")
        raw.extend(convert_m2_file(path, cefr=cefr, stats=stats))

    if not raw:
        raise GecImportError("the corpus converted to no rows")
    if stats.conversion_rate < min_conversion_rate:
        raise GecImportError(
            f"converted {stats.conversion_rate:.2%} of {stats.sentences} sentences, "
            f"below the {min_conversion_rate:.0%} floor; the mapping or the "
            "detokenisation regressed rather than the corpus changing"
        )

    unique = dedupe(raw)
    kept = drop_fragments(unique, min_words=min_words)
    if not kept:
        raise GecImportError(f"every row is under {min_words} words")
    capped = cap_strata(kept, max_stratum_share=max_stratum_share, seed=seed)
    split = split_rows(capped, seed=seed, val_share=val_share)

    counts = Counter(row["split"] for row in split)
    report = {
        **stats.as_dict(),
        "duplicates_removed": len(raw) - len(unique),
        "fragments_removed": len(unique) - len(kept),
        "capped_out": len(kept) - len(capped),
        "rows": len(split),
        "train_rows": counts.get("train", 0),
        "val_rows": counts.get("val", 0),
        "seed": seed,
        "max_stratum_share": max_stratum_share,
        "val_share": val_share,
        "min_words": min_words,
    }
    return split, report


__all__ = [
    "EDIT_BUCKETS",
    "EMPTY_TYPES",
    "ERRANT_TO_TAG",
    "TRAIN_FILES",
    "GecImportError",
    "ImportStats",
    "M2Sentence",
    "cap_strata",
    "convert_m2_file",
    "convert_sentence",
    "dedupe",
    "detokenise",
    "detokenise_text",
    "drop_fragments",
    "edit_bucket",
    "import_corpus",
    "read_m2",
    "row_uid",
    "split_rows",
    "tag_for",
]
