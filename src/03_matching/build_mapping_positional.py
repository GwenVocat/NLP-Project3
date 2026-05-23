"""
Positionales HD → IPA Mapping für Ostschweiz.
Filtert Präteritum-Sätze heraus, dann wird jedes HD-Token
mit dem IPA-Token an derselben Position verglichen.

Bei unterschiedlicher Tokenanzahl werden nur die übereinstimmenden
Positionen (bis min(len_hd, len_ipa)) berücksichtigt.

Output: Data/ostschweiz_mapping_positional.csv
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

INPUT_CSV = DATA_DIR / "transcriptions_tenses.csv"
OUTPUT_CSV = DATA_DIR / "ostschweiz_mapping_positional.csv"
MIN_COUNT  = 2  # Paar muss mindestens N-mal positional übereinstimmen


def clean_hd(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def tokenize_ipa(text: str) -> list[str]:
    return str(text).strip().split()


def main():
    df = pd.read_csv(INPUT_CSV)
    df_ost = df[df["dialect_region"] == "Ostschweiz"].copy().reset_index(drop=True)
    print(f"Sätze (Ostschweiz):        {len(df_ost):,}")

    df_filtered = df_ost[df_ost["tense"] != "Präteritum"].reset_index(drop=True)
    print(f"Sätze ohne Präteritum:     {len(df_filtered):,}")
    print(f"Präteritum herausgefiltert: {len(df_ost) - len(df_filtered):,}")

    pair_counts: Counter = Counter()
    skipped_length = 0
    total_pairs = 0

    for _, row in df_filtered.iterrows():
        hd_tokens  = clean_hd(row["sentence"])
        ipa_tokens = tokenize_ipa(row["ipa_audio"])

        if len(hd_tokens) != len(ipa_tokens):
            skipped_length += 1

        for hd_tok, ipa_tok in zip(hd_tokens, ipa_tokens):
            pair_counts[(hd_tok, ipa_tok)] += 1
            total_pairs += 1

    print(f"\nSätze mit ungleicher Tokenlänge: {skipped_length:,} (werden bis min-Länge verwendet)")
    print(f"Positionale Paare total:         {total_pairs:,}")
    print(f"Einzigartige Paare:              {len(pair_counts):,}")

    records = [
        {"Hochdeutsch": hd, "IPA_Dialekt": ipa, "Anzahl": count}
        for (hd, ipa), count in pair_counts.items()
        if count >= MIN_COUNT
    ]
    records.sort(key=lambda x: x["Anzahl"], reverse=True)

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{len(df_out)} Paare (≥{MIN_COUNT}) gespeichert: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(df_out.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
