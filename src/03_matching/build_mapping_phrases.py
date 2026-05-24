"""
Build phrase-level HD -> IPA mappings for Ostschweiz dialect.

This method extends the word-level mappings to contiguous n-grams. It compares
HD phrases of length 1..3 with IPA phrases of length 1..3, but only when their
relative sentence positions are close. This is intended to surface dialect
correspondences where one HD word maps to multiple IPA tokens, or multiple HD
words map to one dialect token.

Input:  Data/transcriptions_tenses.csv
Output: Data/ostschweiz_mapping_phrases.csv
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

INPUT_CSV = DATA_DIR / "transcriptions_tenses.csv"
OUTPUT_CSV = DATA_DIR / "ostschweiz_mapping_phrases.csv"
DIALECT_REGION = "Ostschweiz"

MAX_HD_NGRAM = 3
MAX_IPA_NGRAM = 3
MAX_POSITION_DISTANCE = 0.25
MIN_SENTENCE_SUPPORT = 3
MIN_HD_RATE = 0.15
MIN_IPA_RATE = 0.15
MIN_PMI = 2.0
TOP_K_PER_HD_PHRASE = 5


@dataclass(frozen=True)
class PhraseOccurrence:
    phrase: str
    n: int
    start_index: int
    center_position: float


@dataclass(frozen=True)
class SentencePair:
    row_id: int
    sentence: str
    ipa_audio: str
    ipa_reference: str
    tense: str
    hd_tokens: list[str]
    ipa_tokens: list[str]


def clean_hd(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def tokenize_ipa(text: str) -> list[str]:
    return str(text).strip().split()


def center_position(start_index: int, ngram_len: int, sentence_len: int) -> float:
    """Map an n-gram center to a stable 0..1 relative sentence position."""
    center_index = start_index + (ngram_len - 1) / 2
    return (center_index + 1) / (sentence_len + 1)


def ngrams(tokens: list[str], max_n: int) -> list[PhraseOccurrence]:
    occurrences: list[PhraseOccurrence] = []
    sentence_len = len(tokens)

    for n in range(1, max_n + 1):
        if sentence_len < n:
            continue

        for start in range(sentence_len - n + 1):
            phrase = " ".join(tokens[start : start + n])
            occurrences.append(
                PhraseOccurrence(
                    phrase=phrase,
                    n=n,
                    start_index=start,
                    center_position=center_position(start, n, sentence_len),
                )
            )

    return occurrences


def load_sentence_pairs() -> list[SentencePair]:
    df = pd.read_csv(INPUT_CSV)
    df_ost = df[df["dialect_region"] == DIALECT_REGION].copy().reset_index(drop=True)

    pairs: list[SentencePair] = []
    for row_id, row in df_ost.iterrows():
        hd_tokens = clean_hd(row["sentence"])
        ipa_tokens = tokenize_ipa(row["ipa_audio"])
        if not hd_tokens or not ipa_tokens:
            continue

        pairs.append(
            SentencePair(
                row_id=row_id,
                sentence=str(row["sentence"]),
                ipa_audio=str(row["ipa_audio"]),
                ipa_reference=str(row["ipa_reference"]),
                tense=str(row.get("tense", "")),
                hd_tokens=hd_tokens,
                ipa_tokens=ipa_tokens,
            )
        )

    print(f"Saetze ({DIALECT_REGION}): {len(df_ost):,}")
    print(f"Verwendete Satzpaare:     {len(pairs):,}")
    return pairs


def collect_phrase_stats(
    pairs: list[SentencePair],
) -> tuple[
    Counter[str],
    Counter[str],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
    dict[tuple[str, str], float],
    dict[tuple[str, str], SentencePair],
    dict[str, int],
    dict[str, int],
]:
    hd_sentence_freq: Counter[str] = Counter()
    ipa_sentence_freq: Counter[str] = Counter()
    pair_sentence_support: Counter[tuple[str, str]] = Counter()
    pair_occurrence_count: Counter[tuple[str, str]] = Counter()
    pair_distance_sum: dict[tuple[str, str], float] = defaultdict(float)
    pair_example: dict[tuple[str, str], SentencePair] = {}
    hd_ngram_len: dict[str, int] = {}
    ipa_ngram_len: dict[str, int] = {}

    for pair in pairs:
        hd_occurrences = ngrams(pair.hd_tokens, MAX_HD_NGRAM)
        ipa_occurrences = ngrams(pair.ipa_tokens, MAX_IPA_NGRAM)

        for occurrence in hd_occurrences:
            hd_ngram_len[occurrence.phrase] = occurrence.n
        for occurrence in ipa_occurrences:
            ipa_ngram_len[occurrence.phrase] = occurrence.n

        hd_sentence_freq.update({occurrence.phrase for occurrence in hd_occurrences})
        ipa_sentence_freq.update({occurrence.phrase for occurrence in ipa_occurrences})

        row_pairs_seen: set[tuple[str, str]] = set()
        for hd_occurrence in hd_occurrences:
            for ipa_occurrence in ipa_occurrences:
                distance = abs(hd_occurrence.center_position - ipa_occurrence.center_position)
                if distance > MAX_POSITION_DISTANCE:
                    continue

                key = (hd_occurrence.phrase, ipa_occurrence.phrase)
                pair_occurrence_count[key] += 1
                pair_distance_sum[key] += distance
                row_pairs_seen.add(key)

                if key not in pair_example:
                    pair_example[key] = pair

        pair_sentence_support.update(row_pairs_seen)

    return (
        hd_sentence_freq,
        ipa_sentence_freq,
        pair_sentence_support,
        pair_occurrence_count,
        pair_distance_sum,
        pair_example,
        hd_ngram_len,
        ipa_ngram_len,
    )


def build_candidates(pairs: list[SentencePair]) -> pd.DataFrame:
    (
        hd_sentence_freq,
        ipa_sentence_freq,
        pair_sentence_support,
        pair_occurrence_count,
        pair_distance_sum,
        pair_example,
        hd_ngram_len,
        ipa_ngram_len,
    ) = collect_phrase_stats(pairs)

    rows = []
    n_sentences = len(pairs)
    for (hd_phrase, ipa_phrase), sentence_support in pair_sentence_support.items():
        if sentence_support < MIN_SENTENCE_SUPPORT:
            continue

        hd_freq = hd_sentence_freq[hd_phrase]
        ipa_freq = ipa_sentence_freq[ipa_phrase]
        hd_rate = sentence_support / hd_freq
        ipa_rate = sentence_support / ipa_freq
        pmi = math.log2((sentence_support * n_sentences) / (hd_freq * ipa_freq))
        occurrence_count = pair_occurrence_count[(hd_phrase, ipa_phrase)]
        avg_position_distance = pair_distance_sum[(hd_phrase, ipa_phrase)] / occurrence_count

        if hd_rate < MIN_HD_RATE:
            continue
        if ipa_rate < MIN_IPA_RATE:
            continue
        if pmi < MIN_PMI:
            continue
        if avg_position_distance > MAX_POSITION_DISTANCE:
            continue

        hd_n = hd_ngram_len[hd_phrase]
        ipa_n = ipa_ngram_len[ipa_phrase]
        position_score = 1 - avg_position_distance
        score = sentence_support * pmi * position_score
        example = pair_example[(hd_phrase, ipa_phrase)]

        rows.append(
            {
                "HD_Phrase": hd_phrase,
                "IPA_Phrase": ipa_phrase,
                "HD_N": hd_n,
                "IPA_N": ipa_n,
                "Mapping_Type": f"{hd_n}-{ipa_n}",
                "Is_Phrase_Mapping": int(hd_n != 1 or ipa_n != 1),
                "Gemeinsame_Treffer": sentence_support,
                "Occurrence_Count": occurrence_count,
                "HD_Haeufigkeit": hd_freq,
                "IPA_Haeufigkeit": ipa_freq,
                "HD_Rate": round(hd_rate, 4),
                "IPA_Rate": round(ipa_rate, 4),
                "PMI": round(pmi, 4),
                "Avg_Position_Distance": round(avg_position_distance, 4),
                "Score": round(score, 4),
                "Example_Sentence": example.sentence,
                "Example_IPA_Audio": example.ipa_audio,
                "Example_IPA_Reference": example.ipa_reference,
                "Example_Tense": example.tense,
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["HD_Phrase", "Score", "Gemeinsame_Treffer", "PMI"],
        ascending=[True, False, False, False],
    )
    candidates["Rank_For_HD_Phrase"] = candidates.groupby("HD_Phrase").cumcount() + 1
    candidates = candidates[candidates["Rank_For_HD_Phrase"] <= TOP_K_PER_HD_PHRASE]

    return candidates.sort_values(
        ["Score", "Gemeinsame_Treffer", "PMI"],
        ascending=False,
    ).reset_index(drop=True)


def print_summary(candidates: pd.DataFrame) -> None:
    if candidates.empty:
        print("Keine Phrase-Mapping-Kandidaten gefunden.")
        return

    print(f"\nKandidaten total:       {len(candidates):,}")
    print(f"HD-Phrasen eindeutig:   {candidates['HD_Phrase'].nunique():,}")
    print(f"Phrase-Mappings (!1-1): {int(candidates['Is_Phrase_Mapping'].sum()):,}")
    print("\nNach Mapping-Typ:")
    print(candidates["Mapping_Type"].value_counts().sort_index().to_string())

    print("\nTop 40 Kandidaten:")
    columns = [
        "HD_Phrase",
        "IPA_Phrase",
        "Mapping_Type",
        "Gemeinsame_Treffer",
        "HD_Rate",
        "IPA_Rate",
        "PMI",
        "Avg_Position_Distance",
        "Score",
    ]
    print(candidates[columns].head(40).to_string(index=False))

    phrase_only = candidates[candidates["Is_Phrase_Mapping"] == 1]
    if not phrase_only.empty:
        print("\nTop 40 echte Phrase-Kandidaten:")
        print(phrase_only[columns].head(40).to_string(index=False))


def main() -> None:
    pairs = load_sentence_pairs()
    candidates = build_candidates(pairs)

    candidates.to_csv(OUTPUT_CSV, index=False)
    print(f"\nGespeichert: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print_summary(candidates)


if __name__ == "__main__":
    main()
