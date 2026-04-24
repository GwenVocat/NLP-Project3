#!/usr/bin/env python3
"""Terminal annotation tool for dialect–standard German word mapping."""

import csv
import json
import os
import sys

import pandas as pd

TRANSCRIPTIONS = "Data/transcriptions_clean.csv"
MAPPING        = "Data/ostschweiz_mapping_results.csv"
SENTENCES_FILE = "Data/annotation_sentences.csv"
RESULTS_FILE   = "Data/annotation_results.csv"
PROGRESS_FILE  = "Data/annotation_progress.json"

SAMPLE_SIZE  = 200
RANDOM_STATE = 42
RESULTS_COLS = [
    "sentence_id", "ipa_word", "hd_ground_truth",
    "auto_mapping", "auto_correct", "skipped",
]
SEP = "━" * 50


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup():
    if not os.path.exists(SENTENCES_FILE):
        df     = pd.read_csv(TRANSCRIPTIONS)
        n      = min(SAMPLE_SIZE, len(df))
        sample = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
        sample.index.name = "sentence_id"
        sample.to_csv(SENTENCES_FILE)
        print(f"  {n} Sätze gesampelt  →  {SENTENCES_FILE}")

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RESULTS_COLS).writeheader()
        print(f"  Leere Ergebnisdatei erstellt  →  {RESULTS_FILE}")

    if not os.path.exists(PROGRESS_FILE):
        _save_done(set())


# ── Progress helpers ──────────────────────────────────────────────────────────

def _load_done() -> set:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return set(json.load(f)["done"])
    return set()


def _save_done(done: set):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done)}, f)


# ── CSV writer ────────────────────────────────────────────────────────────────

def _write_result(sentence_id, ipa_word, hd_ground_truth, auto_mapping,
                  auto_correct, skipped):
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLS)
        writer.writerow({
            "sentence_id":     sentence_id,
            "ipa_word":        ipa_word,
            "hd_ground_truth": hd_ground_truth,
            "auto_mapping":    auto_mapping,
            "auto_correct":    auto_correct,
            "skipped":         skipped,
        })


# ── Core annotation ───────────────────────────────────────────────────────────

def annotate_sentence(sentence_id: int, row: pd.Series, auto_map: dict,
                      display_num: int, total: int):
    hd  = row.get("sentence", "")
    ipa = row.get("ipa_audio", "")

    if pd.isna(ipa) or str(ipa).strip() == "":
        _save_done(_load_done() | {sentence_id})
        return

    tokens = str(ipa).split()
    n      = len(tokens)

    print(f"\n{SEP}")
    print(f"Satz {display_num}/{total}")
    print(SEP)
    print(f"HD:   {hd}")
    print(f"IPA:  {ipa}")
    print(SEP)

    annotated     = 0
    skipped_words = 0
    sentence_skip = False

    for i, token in enumerate(tokens, start=1):
        suggestion = auto_map.get(token, "–")
        print(f"\nWort {i}/{n}:  {token}   [auto: {suggestion}]", end="  ")
        print("(Enter=auto | text=HD | s=skip | ss=Satz skip | q=quit)")

        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAnnotation unterbrochen. Fortschritt gespeichert.")
                sys.exit(0)

            if raw == "q":
                print("Fortschritt gespeichert. Tschüss!")
                sys.exit(0)
            elif raw == "ss":
                sentence_skip = True
                break
            elif raw == "s":
                _write_result(sentence_id, token, "", suggestion, False, True)
                skipped_words += 1
                break
            elif raw == "":
                _write_result(sentence_id, token, suggestion, suggestion, True, False)
                annotated += 1
                break
            else:
                _write_result(sentence_id, token, raw, suggestion, False, False)
                annotated += 1
                break

        if sentence_skip:
            break

    done = _load_done()
    done.add(sentence_id)
    _save_done(done)

    if sentence_skip:
        print(f"  ↷ Satz {display_num} übersprungen")
    else:
        print(f"\n✓ Satz {display_num} fertig – {annotated} annotiert, {skipped_words} skipped")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("── Dialekt-Annotation ──────────────────────────────────────────")
    setup()

    sentences_df = pd.read_csv(SENTENCES_FILE, index_col="sentence_id")
    auto_map     = dict(zip(
        pd.read_csv(MAPPING)["IPA_Dialekt"],
        pd.read_csv(MAPPING)["Hochdeutsch_Zuordnung"],
    ))
    done_ids  = _load_done()
    total     = len(sentences_df)
    remaining = [i for i in sentences_df.index if i not in done_ids]

    if not remaining:
        print(f"Alle {total} Sätze bereits annotiert!")
        return

    already_done = total - len(remaining)
    print(f"Fortschritt: {already_done}/{total} Sätze fertig – {len(remaining)} verbleibend")
    print("Befehle: Enter=Auto | text=eigene HD | s=Wort skip | ss=Satz skip | q=beenden\n")

    for pos, sentence_id in enumerate(remaining):
        display_num = already_done + pos + 1
        annotate_sentence(
            sentence_id,
            sentences_df.loc[sentence_id],
            auto_map,
            display_num,
            total,
        )

    print(f"\nAlle {total} Sätze annotiert! Ergebnisse: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
