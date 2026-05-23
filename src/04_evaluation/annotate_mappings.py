"""
Create and annotate HD -> IPA mapping candidates.

The script combines the top-1 predictions from the three mapping methods,
deduplicates identical HD/IPA pairs, adds example sentences, and optionally
walks through the open annotations in the terminal.

Outputs:
  Data/annotation_candidates.csv

Usage:
  .venv/bin/python src/04_evaluation/annotate_mappings.py --prepare
  .venv/bin/python src/04_evaluation/annotate_mappings.py
  .venv/bin/python src/04_evaluation/annotate_mappings.py --summary
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

TRANSCRIPTIONS_CSV = DATA_DIR / "transcriptions_tenses.csv"
GREEDY_CSV = DATA_DIR / "ostschweiz_mapping_results.csv"
POSITIONAL_CSV = DATA_DIR / "ostschweiz_mapping_positional.csv"
IBM_CSV = DATA_DIR / "ostschweiz_mapping_ibm.csv"
ANNOTATION_CSV = DATA_DIR / "annotation_candidates.csv"

DIALECT_REGION = "Ostschweiz"
ANNOTATION_COLUMNS = ["correct", "comment"]


def clean_hd(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def tokenize_ipa(text: str) -> list[str]:
    return str(text).strip().split()


def as_bool_int(value: bool) -> int:
    return 1 if value else 0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fehlende Datei: {path.relative_to(PROJECT_ROOT)}")
    return pd.read_csv(path)


def best_greedy() -> pd.DataFrame:
    df = read_csv(GREEDY_CSV).copy()
    sort_cols = [c for c in ["Gemeinsame_Treffer", "PMI", "Kokkurrenz_Rate"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)
    df = df.drop_duplicates("Hochdeutsch", keep="first")

    return pd.DataFrame(
        {
            "Hochdeutsch": df["Hochdeutsch"].astype(str),
            "IPA_Dialekt": df["IPA_Dialekt"].astype(str),
            "method_greedy": 1,
            "greedy_hits": df.get("Gemeinsame_Treffer", pd.Series([math.nan] * len(df))).values,
            "greedy_pmi": df.get("PMI", pd.Series([math.nan] * len(df))).values,
            "greedy_rate": df.get("Kokkurrenz_Rate", pd.Series([math.nan] * len(df))).values,
        }
    )


def best_positional() -> pd.DataFrame:
    df = read_csv(POSITIONAL_CSV).copy()
    df = df.sort_values("Anzahl", ascending=False)
    df = df.drop_duplicates("Hochdeutsch", keep="first")

    return pd.DataFrame(
        {
            "Hochdeutsch": df["Hochdeutsch"].astype(str),
            "IPA_Dialekt": df["IPA_Dialekt"].astype(str),
            "method_positional": 1,
            "positional_count": df["Anzahl"].values,
        }
    )


def best_ibm() -> pd.DataFrame:
    df = read_csv(IBM_CSV).copy()
    if "Rang" in df.columns:
        df = df[df["Rang"] == 1].copy()

    sort_cols = [c for c in ["Score", "IBM_Prob", "Expected_Count"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)
    df = df.drop_duplicates("Hochdeutsch", keep="first")

    return pd.DataFrame(
        {
            "Hochdeutsch": df["Hochdeutsch"].astype(str),
            "IPA_Dialekt": df["IPA_Dialekt"].astype(str),
            "method_ibm": 1,
            "ibm_prob": df.get("IBM_Prob", pd.Series([math.nan] * len(df))).values,
            "ibm_expected_count": df.get("Expected_Count", pd.Series([math.nan] * len(df))).values,
            "ibm_sentence_support": df.get("Sentence_Support", pd.Series([math.nan] * len(df))).values,
        }
    )


def merge_methods() -> pd.DataFrame:
    candidates = pd.concat([best_greedy(), best_positional(), best_ibm()], ignore_index=True, sort=False)

    method_cols = ["method_greedy", "method_positional", "method_ibm"]
    numeric_cols = [
        "greedy_hits",
        "greedy_pmi",
        "greedy_rate",
        "positional_count",
        "ibm_prob",
        "ibm_expected_count",
        "ibm_sentence_support",
    ]

    grouped = candidates.groupby(["Hochdeutsch", "IPA_Dialekt"], as_index=False).agg(
        {
            **{col: "max" for col in method_cols},
            **{col: "max" for col in numeric_cols if col in candidates.columns},
        }
    )

    for col in method_cols:
        grouped[col] = grouped[col].fillna(0).astype(int)

    grouped["method_count"] = grouped[method_cols].sum(axis=1)
    grouped["methods"] = grouped.apply(format_methods, axis=1)
    return grouped


def format_methods(row: pd.Series) -> str:
    methods = []
    if as_bool_int(row.get("method_greedy", 0)):
        methods.append("greedy")
    if as_bool_int(row.get("method_positional", 0)):
        methods.append("positional")
    if as_bool_int(row.get("method_ibm", 0)):
        methods.append("ibm")
    return ", ".join(methods)


def add_examples(candidates: pd.DataFrame) -> pd.DataFrame:
    df = read_csv(TRANSCRIPTIONS_CSV)
    df = df[df["dialect_region"] == DIALECT_REGION].copy().reset_index(drop=True)
    df["_hd_tokens"] = df["sentence"].apply(clean_hd)
    df["_ipa_tokens"] = df["ipa_audio"].apply(tokenize_ipa)

    example_rows = []
    for _, row in candidates.iterrows():
        hd = str(row["Hochdeutsch"])
        ipa = str(row["IPA_Dialekt"])

        exact = df[
            df["_hd_tokens"].apply(lambda tokens: hd in tokens)
            & df["_ipa_tokens"].apply(lambda tokens: ipa in tokens)
        ]
        hd_only = df[df["_hd_tokens"].apply(lambda tokens: hd in tokens)]

        if not exact.empty:
            example = exact.iloc[0]
            example_type = "hd+ipa"
        elif not hd_only.empty:
            example = hd_only.iloc[0]
            example_type = "hd_only"
        else:
            example = None
            example_type = "none"

        if example is None:
            example_rows.append(
                {
                    "example_type": example_type,
                    "example_sentence": "",
                    "example_ipa_audio": "",
                    "example_ipa_reference": "",
                    "example_tense": "",
                }
            )
        else:
            example_rows.append(
                {
                    "example_type": example_type,
                    "example_sentence": example["sentence"],
                    "example_ipa_audio": example["ipa_audio"],
                    "example_ipa_reference": example["ipa_reference"],
                    "example_tense": example.get("tense", ""),
                }
            )

    examples = pd.DataFrame(example_rows)
    return pd.concat([candidates.reset_index(drop=True), examples], axis=1)


def preserve_existing_annotations(next_df: pd.DataFrame) -> pd.DataFrame:
    for col in ANNOTATION_COLUMNS:
        next_df[col] = ""

    if not ANNOTATION_CSV.exists():
        return next_df

    old_df = pd.read_csv(ANNOTATION_CSV, keep_default_na=False)
    required_cols = {"Hochdeutsch", "IPA_Dialekt", *ANNOTATION_COLUMNS}
    if not required_cols.issubset(old_df.columns):
        return next_df

    old_annotations = old_df[["Hochdeutsch", "IPA_Dialekt", *ANNOTATION_COLUMNS]].copy()
    merged = next_df.merge(
        old_annotations,
        on=["Hochdeutsch", "IPA_Dialekt"],
        how="left",
        suffixes=("", "_old"),
    )

    for col in ANNOTATION_COLUMNS:
        old_col = f"{col}_old"
        merged[col] = merged[old_col].fillna(merged[col]).astype(str)
        merged = merged.drop(columns=[old_col])

    return merged


def prepare_candidates() -> pd.DataFrame:
    candidates = merge_methods()
    candidates = add_examples(candidates)
    candidates = preserve_existing_annotations(candidates)

    candidates = candidates.sort_values(
        ["method_count", "greedy_hits", "positional_count", "ibm_sentence_support", "Hochdeutsch"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    candidates.insert(0, "id", range(1, len(candidates) + 1))

    ANNOTATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(ANNOTATION_CSV, index=False)
    return candidates


def load_or_prepare() -> pd.DataFrame:
    if ANNOTATION_CSV.exists():
        return pd.read_csv(ANNOTATION_CSV, keep_default_na=False)
    return prepare_candidates()


def is_open_annotation(value: object) -> bool:
    value = str(value).strip()
    return value == "" or value.lower() == "nan"


def print_item(row: pd.Series, index: int, total_open: int, total: int) -> None:
    print("\n" + "=" * 72)
    print(f"Offen {index}/{total_open} | ID {row['id']} von {total}")
    print(f"HD:  {row['Hochdeutsch']}")
    print(f"IPA: {row['IPA_Dialekt']}")
    print(f"Methoden: {row['methods']}")

    evidence = []
    if str(row.get("greedy_hits", "")).strip():
        evidence.append(f"greedy hits={row.get('greedy_hits')} pmi={row.get('greedy_pmi')}")
    if str(row.get("positional_count", "")).strip():
        evidence.append(f"positional count={row.get('positional_count')}")
    if str(row.get("ibm_prob", "")).strip():
        evidence.append(f"ibm prob={row.get('ibm_prob')} support={row.get('ibm_sentence_support')}")
    if evidence:
        print("Evidenz: " + " | ".join(evidence))

    print(f"\nSatz:      {row.get('example_sentence', '')}")
    print(f"IPA Audio: {row.get('example_ipa_audio', '')}")
    print(f"IPA Ref:   {row.get('example_ipa_reference', '')}")
    print(f"Beispiel:  {row.get('example_type', '')}")


def annotate_interactively(limit: int | None = None) -> None:
    df = load_or_prepare()
    open_mask = df["correct"].apply(is_open_annotation)
    open_indices = list(df[open_mask].index)

    if limit is not None:
        open_indices = open_indices[:limit]

    if not open_indices:
        print("Keine offenen Annotationen mehr.")
        print_summary(df)
        return

    total_open = len(open_indices)
    print(f"Annotation-Datei: {ANNOTATION_CSV.relative_to(PROJECT_ROOT)}")
    print("Eingabe: 1=korrekt, 0=falsch, ?=unsicher, s=skip, q=quit")

    for progress, idx in enumerate(open_indices, start=1):
        row = df.loc[idx]
        print_item(row, progress, total_open, len(df))

        while True:
            answer = input("\nBewertung [1/0/?/s/q]: ").strip().lower()
            if answer in {"1", "0", "?", "s", "q"}:
                break
            print("Bitte nur 1, 0, ?, s oder q eingeben.")

        if answer == "q":
            print("Gestoppt. Bisherige Annotationen bleiben gespeichert.")
            break
        if answer == "s":
            continue

        df.at[idx, "correct"] = answer
        if answer in {"0", "?"}:
            comment = input("Kommentar optional: ").strip()
            df.at[idx, "comment"] = comment

        df.to_csv(ANNOTATION_CSV, index=False)
        print("Gespeichert.")

    print_summary(df)


def print_summary(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = load_or_prepare()

    print("\nZusammenfassung")
    print("-" * 72)
    print(f"Kandidaten total: {len(df)}")
    print(f"Annotiert:        {(~df['correct'].apply(is_open_annotation)).sum()}")
    print(f"Offen:            {df['correct'].apply(is_open_annotation).sum()}")

    for method_col, label in [
        ("method_greedy", "Greedy/PMI"),
        ("method_positional", "Positional"),
        ("method_ibm", "IBM/EM"),
    ]:
        method_df = df[df[method_col].astype(int) == 1].copy()
        annotated = method_df[method_df["correct"].isin(["0", "1"])]
        unclear = method_df[method_df["correct"] == "?"]
        if len(annotated) == 0:
            precision = "n/a"
        else:
            precision = f"{(annotated['correct'] == '1').mean():.3f}"

        print(
            f"{label:<12} candidates={len(method_df):>3} "
            f"annotiert={len(annotated):>3} unsicher={len(unclear):>3} "
            f"precision={precision}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HD -> IPA Mapping-Kandidaten annotieren.")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Nur Data/annotation_candidates.csv neu aufbauen, nicht interaktiv annotieren.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Aktuellen Annotationstand zusammenfassen.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nur N offene Kandidaten in dieser Sitzung annotieren.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prepare:
        candidates = prepare_candidates()
        print(f"{len(candidates)} Kandidaten gespeichert: {ANNOTATION_CSV.relative_to(PROJECT_ROOT)}")
        print_summary(candidates)
        return

    if args.summary:
        print_summary()
        return

    annotate_interactively(limit=args.limit)


if __name__ == "__main__":
    main()
