"""
Evaluate manually annotated HD -> IPA mapping candidates.

Input:
  Data/annotation_candidates.csv

Outputs:
  Data/evaluation_method_summary.csv
  Data/evaluation_consensus_summary.csv
  Data/evaluation_error_examples.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

ANNOTATION_CSV = DATA_DIR / "annotation_candidates.csv"
METHOD_SUMMARY_CSV = DATA_DIR / "evaluation_method_summary.csv"
CONSENSUS_SUMMARY_CSV = DATA_DIR / "evaluation_consensus_summary.csv"
ERROR_EXAMPLES_CSV = DATA_DIR / "evaluation_error_examples.csv"

METHODS = [
    ("method_greedy", "Greedy/PMI"),
    ("method_positional", "Positional"),
    ("method_ibm", "IBM/EM"),
]


def load_annotations() -> pd.DataFrame:
    if not ANNOTATION_CSV.exists():
        raise FileNotFoundError(f"Fehlende Datei: {ANNOTATION_CSV.relative_to(PROJECT_ROOT)}")

    df = pd.read_csv(ANNOTATION_CSV, keep_default_na=False)
    df["correct"] = df["correct"].astype(str).str.strip()
    return df


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unique_hd_total = df["Hochdeutsch"].nunique()

    for method_col, method_name in METHODS:
        method_df = df[df[method_col].astype(int) == 1].copy()
        judged_df = method_df[method_df["correct"].isin(["0", "1"])]
        correct_count = int((judged_df["correct"] == "1").sum())
        wrong_count = int((judged_df["correct"] == "0").sum())
        unclear_count = int((method_df["correct"] == "?").sum())

        precision = correct_count / len(judged_df) if len(judged_df) else None
        precision_including_unclear_wrong = correct_count / len(method_df) if len(method_df) else None

        rows.append(
            {
                "method": method_name,
                "candidates": len(method_df),
                "unique_hd": method_df["Hochdeutsch"].nunique(),
                "coverage_unique_hd_total": method_df["Hochdeutsch"].nunique() / unique_hd_total,
                "judged": len(judged_df),
                "correct": correct_count,
                "wrong": wrong_count,
                "unclear": unclear_count,
                "precision_excluding_unclear": precision,
                "precision_unclear_as_wrong": precision_including_unclear_wrong,
            }
        )

    return pd.DataFrame(rows)


def consensus_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["method_count", "methods", "correct"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    rows = []
    for (method_count, methods), group in grouped.groupby(["method_count", "methods"]):
        correct = int(group.loc[group["correct"] == "1", "count"].sum())
        wrong = int(group.loc[group["correct"] == "0", "count"].sum())
        unclear = int(group.loc[group["correct"] == "?", "count"].sum())
        judged = correct + wrong
        total = correct + wrong + unclear

        rows.append(
            {
                "method_count": method_count,
                "methods": methods,
                "total": total,
                "judged": judged,
                "correct": correct,
                "wrong": wrong,
                "unclear": unclear,
                "precision_excluding_unclear": correct / judged if judged else None,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["method_count", "total"], ascending=[False, False]).reset_index(drop=True)


def error_examples(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    wrong = df[df["correct"] == "0"].copy()
    if wrong.empty:
        return wrong

    sort_cols = ["method_count"]
    ascending = [False]
    if "greedy_hits" in wrong.columns:
        sort_cols.append("greedy_hits")
        ascending.append(False)
    if "positional_count" in wrong.columns:
        sort_cols.append("positional_count")
        ascending.append(False)
    if "ibm_sentence_support" in wrong.columns:
        sort_cols.append("ibm_sentence_support")
        ascending.append(False)

    columns = [
        "id",
        "Hochdeutsch",
        "IPA_Dialekt",
        "methods",
        "example_sentence",
        "example_ipa_audio",
        "example_ipa_reference",
        "comment",
    ]
    return wrong.sort_values(sort_cols, ascending=ascending)[columns].head(n)


def print_summary(methods: pd.DataFrame, consensus: pd.DataFrame) -> None:
    print("\nMethoden")
    print("-" * 88)
    for _, row in methods.iterrows():
        print(
            f"{row['method']:<12} "
            f"candidates={row['candidates']:>3} "
            f"judged={row['judged']:>3} "
            f"correct={row['correct']:>3} "
            f"wrong={row['wrong']:>3} "
            f"unclear={row['unclear']:>2} "
            f"precision={row['precision_excluding_unclear']:.3f} "
            f"coverage_hd={row['coverage_unique_hd_total']:.3f}"
        )

    print("\nKonsens")
    print("-" * 88)
    print(consensus.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    df = load_annotations()
    methods = method_summary(df)
    consensus = consensus_summary(df)
    errors = error_examples(df)

    methods.to_csv(METHOD_SUMMARY_CSV, index=False)
    consensus.to_csv(CONSENSUS_SUMMARY_CSV, index=False)
    errors.to_csv(ERROR_EXAMPLES_CSV, index=False)

    print_summary(methods, consensus)
    print("\nGespeichert:")
    print(f"- {METHOD_SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"- {CONSENSUS_SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"- {ERROR_EXAMPLES_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
