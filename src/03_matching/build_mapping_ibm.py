"""
Build HD -> IPA word mappings with IBM Model 1 style EM alignment.

The model learns lexical probabilities P(ipa_token | hd_word) from aligned
sentence pairs without word-level supervision. A soft position prior gives
higher weight to HD and IPA tokens at similar relative sentence positions.

Input:  Data/transcriptions_tenses.csv
Output: Data/ostschweiz_mapping_ibm.csv
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
OUTPUT_CSV = DATA_DIR / "ostschweiz_mapping_ibm.csv"
DIALECT_REGION = "Ostschweiz"

ITERATIONS = 12
POSITION_SIGMA = 0.25
NULL_TOKEN = "__NULL__"
NULL_PRIOR = 0.05
EPSILON = 1e-12

MIN_EXPECTED_COUNT = 3.0
MIN_LEXICAL_PROB = 0.05
MIN_SENTENCE_SUPPORT = 4
MIN_HD_FREQUENCY = 4
TOP_K_PER_HD = 3


@dataclass(frozen=True)
class SentencePair:
    row_id: int
    hd_tokens: list[str]
    ipa_tokens: list[str]


def clean_hd(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def tokenize_ipa(text: str) -> list[str]:
    return str(text).strip().split()


def relative_position(index: int, length: int) -> float:
    """Map token index to a stable 0..1 sentence position."""
    return (index + 1) / (length + 1)


def position_prior(src_index: int | None, trg_index: int, src_len: int, trg_len: int) -> float:
    """
    Diagonal alignment prior.

    Tokens with similar relative positions get a prior near 1.0. The NULL token
    uses a small constant prior so extra IPA tokens can be ignored.
    """
    if src_index is None:
        return NULL_PRIOR

    src_pos = relative_position(src_index, src_len)
    trg_pos = relative_position(trg_index, trg_len)
    diff = src_pos - trg_pos
    return math.exp(-(diff * diff) / (2 * POSITION_SIGMA * POSITION_SIGMA))


def load_sentence_pairs() -> tuple[list[SentencePair], Counter[str], Counter[str]]:
    df = pd.read_csv(INPUT_CSV)
    df_ost = df[df["dialect_region"] == DIALECT_REGION].copy().reset_index(drop=True)

    pairs: list[SentencePair] = []
    hd_freq: Counter[str] = Counter()
    ipa_freq: Counter[str] = Counter()

    for row_id, row in df_ost.iterrows():
        hd_tokens = clean_hd(row["sentence"])
        ipa_tokens = tokenize_ipa(row["ipa_audio"])
        if not hd_tokens or not ipa_tokens:
            continue

        pairs.append(SentencePair(row_id=row_id, hd_tokens=hd_tokens, ipa_tokens=ipa_tokens))
        hd_freq.update(hd_tokens)
        ipa_freq.update(ipa_tokens)

    print(f"Saetze ({DIALECT_REGION}): {len(df_ost):,}")
    print(f"Verwendete Satzpaare:     {len(pairs):,}")
    print(f"HD-Vokabular:             {len(hd_freq):,}")
    print(f"IPA-Vokabular:            {len(ipa_freq):,}")

    return pairs, hd_freq, ipa_freq


def initialize_translation_table(pairs: list[SentencePair]) -> dict[str, dict[str, float]]:
    possible_targets: dict[str, set[str]] = defaultdict(set)

    for pair in pairs:
        ipa_types = set(pair.ipa_tokens)
        for hd_token in set(pair.hd_tokens):
            possible_targets[hd_token].update(ipa_types)
        possible_targets[NULL_TOKEN].update(ipa_types)

    table: dict[str, dict[str, float]] = {}
    for hd_token, ipa_tokens in possible_targets.items():
        initial_prob = 1.0 / len(ipa_tokens)
        table[hd_token] = {ipa_token: initial_prob for ipa_token in ipa_tokens}

    return table


def expectation_step(
    pairs: list[SentencePair],
    translation_table: dict[str, dict[str, float]],
) -> tuple[dict[str, Counter[str]], Counter[str], float]:
    expected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_by_hd: Counter[str] = Counter()
    log_likelihood = 0.0

    for pair in pairs:
        src_len = len(pair.hd_tokens)
        trg_len = len(pair.ipa_tokens)

        for trg_index, ipa_token in enumerate(pair.ipa_tokens):
            weighted_sources: list[tuple[str, float]] = []

            null_prob = translation_table[NULL_TOKEN].get(ipa_token, EPSILON)
            null_weight = null_prob * position_prior(None, trg_index, src_len, trg_len)
            weighted_sources.append((NULL_TOKEN, null_weight))

            for src_index, hd_token in enumerate(pair.hd_tokens):
                lexical_prob = translation_table.get(hd_token, {}).get(ipa_token, EPSILON)
                weight = lexical_prob * position_prior(src_index, trg_index, src_len, trg_len)
                weighted_sources.append((hd_token, weight))

            denominator = sum(weight for _, weight in weighted_sources)
            if denominator <= 0:
                continue

            log_likelihood += math.log(denominator)
            for hd_token, weight in weighted_sources:
                posterior = weight / denominator
                expected_counts[hd_token][ipa_token] += posterior
                total_by_hd[hd_token] += posterior

    return expected_counts, total_by_hd, log_likelihood


def maximization_step(
    expected_counts: dict[str, Counter[str]],
    total_by_hd: Counter[str],
) -> dict[str, dict[str, float]]:
    next_table: dict[str, dict[str, float]] = {}

    for hd_token, ipa_counts in expected_counts.items():
        total = total_by_hd[hd_token]
        if total <= 0:
            continue
        next_table[hd_token] = {
            ipa_token: count / total
            for ipa_token, count in ipa_counts.items()
            if count > 0
        }

    return next_table


def train_ibm_model(pairs: list[SentencePair]) -> dict[str, dict[str, float]]:
    translation_table = initialize_translation_table(pairs)

    for iteration in range(1, ITERATIONS + 1):
        expected_counts, total_by_hd, log_likelihood = expectation_step(pairs, translation_table)
        translation_table = maximization_step(expected_counts, total_by_hd)
        print(f"Iteration {iteration:>2}: log_likelihood={log_likelihood:.2f}")

    return translation_table


def collect_final_alignment_stats(
    pairs: list[SentencePair],
    translation_table: dict[str, dict[str, float]],
) -> tuple[
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
    dict[tuple[str, str], float],
]:
    expected_pair_counts: Counter[tuple[str, str]] = Counter()
    hard_pair_counts: Counter[tuple[str, str]] = Counter()
    hard_pair_rows: dict[tuple[str, str], set[int]] = defaultdict(set)
    weighted_position_distance: dict[tuple[str, str], float] = defaultdict(float)

    for pair in pairs:
        src_len = len(pair.hd_tokens)
        trg_len = len(pair.ipa_tokens)

        for trg_index, ipa_token in enumerate(pair.ipa_tokens):
            weighted_sources: list[tuple[str, int | None, float]] = []

            null_prob = translation_table[NULL_TOKEN].get(ipa_token, EPSILON)
            null_weight = null_prob * position_prior(None, trg_index, src_len, trg_len)
            weighted_sources.append((NULL_TOKEN, None, null_weight))

            for src_index, hd_token in enumerate(pair.hd_tokens):
                lexical_prob = translation_table.get(hd_token, {}).get(ipa_token, EPSILON)
                weight = lexical_prob * position_prior(src_index, trg_index, src_len, trg_len)
                weighted_sources.append((hd_token, src_index, weight))

            denominator = sum(weight for _, _, weight in weighted_sources)
            if denominator <= 0:
                continue

            best_hd, _, _ = max(weighted_sources, key=lambda item: item[2])
            if best_hd != NULL_TOKEN:
                hard_pair_counts[(best_hd, ipa_token)] += 1
                hard_pair_rows[(best_hd, ipa_token)].add(pair.row_id)

            for hd_token, src_index, weight in weighted_sources:
                if hd_token == NULL_TOKEN:
                    continue

                posterior = weight / denominator
                key = (hd_token, ipa_token)
                expected_pair_counts[key] += posterior

                src_pos = relative_position(src_index, src_len)
                trg_pos = relative_position(trg_index, trg_len)
                weighted_position_distance[key] += posterior * abs(src_pos - trg_pos)

    avg_position_distance = {
        key: weighted_position_distance[key] / expected_count
        for key, expected_count in expected_pair_counts.items()
        if expected_count > 0
    }

    hard_sentence_counts = Counter({
        key: len(row_ids)
        for key, row_ids in hard_pair_rows.items()
    })

    return expected_pair_counts, hard_pair_counts, hard_sentence_counts, avg_position_distance


def build_output(
    translation_table: dict[str, dict[str, float]],
    expected_pair_counts: Counter[tuple[str, str]],
    hard_pair_counts: Counter[tuple[str, str]],
    hard_sentence_counts: Counter[tuple[str, str]],
    avg_position_distance: dict[tuple[str, str], float],
    hd_freq: Counter[str],
    ipa_freq: Counter[str],
) -> pd.DataFrame:
    candidates_by_hd: dict[str, list[dict[str, object]]] = defaultdict(list)

    for hd_token, ipa_probs in translation_table.items():
        if hd_token == NULL_TOKEN:
            continue
        if hd_freq[hd_token] < MIN_HD_FREQUENCY:
            continue

        for ipa_token, lexical_prob in ipa_probs.items():
            expected_count = expected_pair_counts[(hd_token, ipa_token)]
            sentence_support = hard_sentence_counts[(hd_token, ipa_token)]
            if (
                expected_count < MIN_EXPECTED_COUNT
                or lexical_prob < MIN_LEXICAL_PROB
                or sentence_support < MIN_SENTENCE_SUPPORT
            ):
                continue

            score = lexical_prob * math.log1p(expected_count)
            candidates_by_hd[hd_token].append(
                {
                    "Hochdeutsch": hd_token,
                    "IPA_Dialekt": ipa_token,
                    "IBM_Prob": round(lexical_prob, 6),
                    "Expected_Count": round(expected_count, 4),
                    "Hard_Count": hard_pair_counts[(hd_token, ipa_token)],
                    "Sentence_Support": sentence_support,
                    "Score": round(score, 6),
                    "Avg_Position_Distance": round(avg_position_distance.get((hd_token, ipa_token), 0.0), 4),
                    "HD_Gesamt_Haeufigkeit": hd_freq[hd_token],
                    "IPA_Gesamt_Haeufigkeit": ipa_freq[ipa_token],
                }
            )

    output_rows: list[dict[str, object]] = []
    for hd_token, candidates in candidates_by_hd.items():
        candidates.sort(key=lambda row: (row["Score"], row["Expected_Count"]), reverse=True)
        for rank, row in enumerate(candidates[:TOP_K_PER_HD], start=1):
            row["Rang"] = rank
            output_rows.append(row)

    df_out = pd.DataFrame(output_rows)
    if df_out.empty:
        return df_out

    columns = [
        "Hochdeutsch",
        "IPA_Dialekt",
        "Rang",
        "IBM_Prob",
        "Expected_Count",
        "Hard_Count",
        "Sentence_Support",
        "Score",
        "Avg_Position_Distance",
        "HD_Gesamt_Haeufigkeit",
        "IPA_Gesamt_Haeufigkeit",
    ]
    return df_out[columns].sort_values(["Score", "Expected_Count"], ascending=[False, False])


def main() -> None:
    pairs, hd_freq, ipa_freq = load_sentence_pairs()
    translation_table = train_ibm_model(pairs)

    expected_counts, hard_counts, sentence_counts, avg_pos_dist = collect_final_alignment_stats(pairs, translation_table)
    df_out = build_output(
        translation_table,
        expected_counts,
        hard_counts,
        sentence_counts,
        avg_pos_dist,
        hd_freq,
        ipa_freq,
    )

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{len(df_out)} Kandidaten gespeichert: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    if not df_out.empty:
        print(df_out.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
