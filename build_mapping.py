"""
Build HD → IPA co-occurrence mapping for Ostschweiz dialect.
Uses greedy iterative alignment: each round the best HD–IPA pair is locked in,
then both tokens are removed from all working sentences before the next round.

Output: Data/ostschweiz_mapping_results.csv
Columns: Hochdeutsch, HD_Corpus_Frequency, IPA_Dialekt,
         IPA_Corpus_Frequency, Gemeinsame_Treffer, Kokkurrenz_Rate, Runde
"""

import math
import re
from collections import Counter, defaultdict

import pandas as pd

MAX_PAIRS    = 400
MIN_HITS     = 4      # mindestens N gemeinsame Treffer
MIN_KOKKURRENZ = 0.1   # nur Paare über dieser Rate werden gespeichert
MIN_PMI        = 2    # nur Paare mit positivem PMI (öfter als zufällig)
INPUT_CSV    = "Data/transcriptions_tenses.csv"
OUTPUT_CSV   = "Data/ostschweiz_mapping_results.csv"


def clean_hd(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def tokenize_ipa(text: str) -> list[str]:
    return str(text).strip().split()


def compute_rates(
    hd_sents: list[list[str]],
    ipa_sents: list[list[str]],
    n_clips: int,
) -> dict[tuple[str, str], tuple[int, int, float, float]]:
    """
    Berechnet Kokkurrenz-Raten und PMI für alle HD–IPA-Paare.
    Gibt {(hd, ipa): (clip_count, hits, rate, pmi)} zurück.
    """
    hd_word_rows: dict[str, set[int]] = defaultdict(set)
    for idx, tokens in enumerate(hd_sents):
        for word in tokens:
            hd_word_rows[word].add(idx)

    ipa_clip_counts: dict[str, int] = defaultdict(int)
    for sent in ipa_sents:
        for tok in set(sent):
            ipa_clip_counts[tok] += 1

    pairs: dict[tuple[str, str], tuple[int, int, float, float]] = {}
    for hd_word, indices in hd_word_rows.items():
        clip_count = len(indices)
        ipa_counts: dict[str, int] = defaultdict(int)
        for idx in indices:
            for ipa_tok in set(ipa_sents[idx]):
                ipa_counts[ipa_tok] += 1

        for ipa_tok, hits in ipa_counts.items():
            if hits >= MIN_HITS:
                rate      = hits / clip_count
                inv_rate  = hits / ipa_clip_counts[ipa_tok]
                pmi       = math.log2((hits * n_clips) / (clip_count * ipa_clip_counts[ipa_tok]))
                pairs[(hd_word, ipa_tok)] = (clip_count, hits, rate, inv_rate, pmi)

    return pairs


def main():
    df = pd.read_csv(INPUT_CSV)
    df_ost = df[df["dialect_region"] == "Ostschweiz"].copy().reset_index(drop=True)
    print(f"Sätze (Ostschweiz): {len(df_ost):,}")

    # Originalhäufigkeit für Output-Spalte
    corpus_freq: Counter = Counter()
    for _, row in df_ost.iterrows():
        for word in clean_hd(row["sentence"]):
            corpus_freq[word] += 1

    ipa_total_freq: Counter = Counter()
    for ipa_text in df_ost["ipa_audio"]:
        for tok in tokenize_ipa(ipa_text):
            ipa_total_freq[tok] += 1

    # Arbeitskopien der Sätze als Token-Listen
    hd_working  = [clean_hd(s)       for s in df_ost["sentence"]]
    ipa_working = [tokenize_ipa(s)    for s in df_ost["ipa_audio"]]

    results = []
    matched_hd  = set()

    for runde in range(1, MAX_PAIRS + 1):
        pairs = compute_rates(hd_working, ipa_working, len(hd_working))
        pairs = {k: v for k, v in pairs.items() if v[2] >= MIN_KOKKURRENZ and v[3] >= MIN_KOKKURRENZ and v[4] >= MIN_PMI}

        if not pairs:
            print(f"Runde {runde}: keine Paare mehr gefunden – Abbruch.")
            break

        # Bestes Paar = meiste Hits (absolute Evidenz)
        best_pair, (clip_count, hits, rate, inv_rate, pmi) = max(pairs.items(), key=lambda x: x[1][1])
        hd_word, ipa_tok = best_pair

        results.append(
            {
                "Hochdeutsch":           hd_word,
                "IPA_Dialekt":           ipa_tok,
                "IPA_Gesamt_Häufigkeit": ipa_total_freq[ipa_tok],
                "HD_Gesamt_Häufigkeit":  corpus_freq[hd_word],
                "Gemeinsame_Treffer":    hits,
                "Kokkurrenz_Rate":       round(rate, 4),
                "IPA_Kokkurrenz_Rate":   round(inv_rate, 4),
                "PMI":                   round(pmi, 4),
                "Runde":                 runde,
            }
        )
        print(f"Runde {runde:>3}: {hd_word:<15} → {ipa_tok:<15}  Rate={rate:.4f}  PMI={pmi:.2f}  Hits={hits}")

        # HD-Wort überall entfernen (gematcht); IPA-Token nur aus Co-Occurrence-Clips
        co_indices = {
            idx for idx, (h, i) in enumerate(zip(hd_working, ipa_working))
            if hd_word in h and ipa_tok in i
        }
        matched_hd.add(hd_word)
        hd_working  = [[t for t in sent if t != hd_word] for sent in hd_working]
        ipa_working = [
            [t for t in sent if t != ipa_tok] if idx in co_indices else sent
            for idx, sent in enumerate(ipa_working)
        ]

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{len(df_out)} Paare gespeichert: {OUTPUT_CSV}")
    print(df_out.to_string(index=False))

    # Remainder-Export: Sätze mit noch ungematchten Tokens
    remainder_rows = []
    for idx, (hd_rem, ipa_rem) in enumerate(zip(hd_working, ipa_working)):
        if hd_rem:
            row = df_ost.loc[idx, ["path", "dialect_region", "sentence", "ipa_reference", "ipa_audio", "ipa_swiss_whisper", "tense"]].to_dict()
            row["hd_remainder"]  = ", ".join(hd_rem)
            row["ipa_remainder"] = ", ".join(ipa_rem)
            remainder_rows.append(row)

    df_rem = pd.DataFrame(remainder_rows)
    df_rem.to_csv("Data/ostschweiz_remainder.csv", index=False)
    print(f"\n{len(df_rem)} Sätze mit Remainder gespeichert: Data/ostschweiz_remainder.csv")


if __name__ == "__main__":
    main()
