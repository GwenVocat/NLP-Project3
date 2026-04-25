#!/usr/bin/env python3
"""Terminal annotation tool for dialect–standard German word mapping."""

import csv
import json
import os
import sys

import pandas as pd

TRANSCRIPTIONS = "Data/transcriptions_normalized.csv"
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
        df      = pd.read_csv(TRANSCRIPTIONS)
        mapping = pd.read_csv(MAPPING)

        # 1. HD-Worthäufigkeit aus "sentence"-Spalte
        hd_counts = (
            df["sentence"]
            .dropna()
            .str.lower()
            .str.split()
            .explode()
            .value_counts()
        )
        target_hd = set(hd_counts[hd_counts.between(50, 300)].index)
        print(f"  {len(target_hd)} Ziel-HD-Wörter mit Häufigkeit 50–300 gefunden")

        # 2. Mapping umkehren: HD → IPA (ein HD-Wort → mehrere IPA-Tokens möglich)
        hd_to_ipa = mapping.groupby(
            mapping["Hochdeutsch_Zuordnung"].str.lower()
        )["IPA_Dialekt"].apply(set)
        target_ipa = set().union(
            *(hd_to_ipa[w] for w in target_hd if w in hd_to_ipa)
        )
        print(f"  {len(target_ipa)} IPA-Tokens diesen HD-Wörtern zugeordnet")

        # 3. Sätze filtern, die mindestens ein Ziel-IPA-Token enthalten
        mask = df["ipa_audio"].apply(
            lambda x: not pd.isna(x) and bool(set(str(x).split()) & target_ipa)
        )
        candidates = df[mask]
        print(f"  {len(candidates)} Sätze enthalten mindestens ein Ziel-IPA-Token")

        n      = min(SAMPLE_SIZE, len(candidates))
        sample = candidates.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
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


def _undo_last_result() -> bool:
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return False
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines[:-1])
    return True


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
    last_action   = None  # "annotated" | "skipped"

    i = 0
    while i < n:
        token      = tokens[i]
        suggestion = auto_map.get(token, "–")
        print(f"\nWort {i+1}/{n}:  {token}   [auto: {suggestion}]", end="  ")
        print("(Enter=auto | text=HD | s=skip | ss=Satz skip | u=zurück+undo | d=undo | q=quit)")

        redo = False
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
            elif raw == "d":
                if last_action is not None:
                    if _undo_last_result():
                        if last_action == "annotated":
                            annotated -= 1
                        else:
                            skipped_words -= 1
                        last_action = None
                        print("  ✗ Aktueller Eintrag gelöscht")
                    else:
                        print("  Keine Einträge zum Löschen.")
                else:
                    print("  Kein Eintrag zum Löschen.")
                redo = True
                break
            elif raw == "u":
                if i > 0 and last_action is not None:
                    if _undo_last_result():
                        if last_action == "annotated":
                            annotated -= 1
                        else:
                            skipped_words -= 1
                        last_action = None
                        i -= 1
                        print("  ↩ Letzte Eingabe gelöscht")
                    else:
                        print("  Keine Einträge zum Rückgängigmachen.")
                else:
                    print("  Keine vorherige Eingabe in diesem Satz.")
                redo = True
                break
            elif raw == "s":
                _write_result(sentence_id, token, "", suggestion, False, True)
                skipped_words += 1
                last_action = "skipped"
                break
            elif raw == "":
                _write_result(sentence_id, token, suggestion, suggestion, True, False)
                annotated += 1
                last_action = "annotated"
                break
            else:
                _write_result(sentence_id, token, raw, suggestion, False, False)
                annotated += 1
                last_action = "annotated"
                break

        if sentence_skip:
            break

        if not redo:
            i += 1

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
    print("Befehle: Enter=Auto | text=eigene HD | s=Wort skip | ss=Satz skip | d=undo | u=zurück+undo | q=beenden\n")

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
