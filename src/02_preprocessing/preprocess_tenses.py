"""
Klassifiziert HD-Sätze nach grammatikalischer Zeit.

Input:  Data/transcriptions_clean.csv
Output: Data/transcriptions_tenses.csv  (alle Sätze mit tense-Spalte)

Zeiten: Präsens, Präteritum, Perfekt, Plusquamperfekt, Futur
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

INPUT_CSV = DATA_DIR / "transcriptions_clean.csv"
OUTPUT_ALL = DATA_DIR / "transcriptions_tenses.csv"

_SEIN_PRAET      = {"war", "waren", "warst", "wart"}
_HABEN_PRAET     = {"hatte", "hatten", "hattest", "hattet"}
_HABEN_SEIN_PRAES = {
    "hat", "habe", "haben", "habt", "hast",
    "ist", "bin", "bist", "sind", "seid",
}
_WERDEN_PRAES    = {"wird", "werde", "werden", "werdet", "wirst"}

_PRAET_EXTRA = {
    # werden
    "wurde", "wurden", "wurdest", "wurdet",
    # Modalverben
    "konnte", "konnten", "konntest", "konntet",
    "wollte", "wollten", "wolltest", "wolltet",
    "sollte", "sollten", "solltest", "solltet",
    "musste", "mussten", "musstest", "musstet",
    "durfte", "durften", "durftest", "durftet",
    "mochte", "mochten", "mochtest", "mochtet",
    # starke Verben
    "ging", "gingen", "gingst", "gingt",
    "kam", "kamen", "kamst", "kamt",
    "gab", "gaben", "gabst",
    "nahm", "nahmen",
    "stand", "standen",
    "lag", "lagen",
    "saß", "saßen",
    "sah", "sahen",
    "sprach", "sprachen",
    "schrieb", "schrieben",
    "blieb", "blieben",
    "fand", "fanden",
    "hielt", "hielten",
    "zog", "zogen",
    "trat", "traten",
    "rief", "riefen",
    "lief", "liefen",
    "ließ", "ließen",
    "schlug", "schlugen",
    "verlor", "verloren",
    "gewann", "gewannen",
    "begann", "begannen",
    "erschien", "erschienen",
    "stieg", "stiegen",
    "trug", "trugen",
    "schloss", "schlossen",
    "bot", "boten",
    "wies", "wiesen",
}


def _has_partizip_ii(tokens: list[str]) -> bool:
    """ge...t / ge...en – deckt den Grossteil der Partizip-II-Formen ab."""
    return any(re.match(r"ge\w{2,}(t|en)$", t) for t in tokens)


def classify_tense(sentence: str) -> str:
    clean  = re.sub(r"[^\w\s]", "", sentence.lower())
    tokens = clean.split()
    tset   = set(tokens)
    partizip = _has_partizip_ii(tokens)

    if tset & _WERDEN_PRAES:
        return "Futur"
    if (tset & _HABEN_PRAET or tset & _SEIN_PRAET) and partizip:
        return "Plusquamperfekt"
    if tset & _HABEN_SEIN_PRAES and partizip:
        return "Perfekt"
    if tset & _HABEN_PRAET or tset & _SEIN_PRAET or tset & _PRAET_EXTRA:
        return "Präteritum"
    return "Präsens"


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Sätze gesamt: {len(df):,}")

    df["tense"] = df["sentence"].apply(classify_tense)

    counts = df["tense"].value_counts()
    print("\nZeitenverteilung:")
    for tense, count in counts.items():
        print(f"  {tense:<18} {count:>5}  ({count/len(df)*100:.1f}%)")

    df.to_csv(OUTPUT_ALL, index=False)
    print(f"\nAlle Sätze gespeichert: {OUTPUT_ALL.relative_to(PROJECT_ROOT)}")



if __name__ == "__main__":
    main()
