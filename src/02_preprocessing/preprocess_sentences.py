#!/usr/bin/env python3
"""Normalize German sentences for annotation:
- Numbers → German words (dreissig, zweitausendsiebzehn, ...)
- Symbols that are read aloud → German words (%, §, &, ...)
- Cleans invisible/typographic characters
"""

import re
from pathlib import Path

import pandas as pd
from num2words import num2words

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

INPUT = DATA_DIR / "transcriptions_clean.csv"
OUTPUT = DATA_DIR / "transcriptions_normalized.csv"


def _card(n: int) -> str:
    return num2words(n, lang="de")


def _ord(n: int) -> str:
    return num2words(n, lang="de", to="ordinal")


def _replace_decimal(m: re.Match) -> str:
    int_part = _card(int(m.group(1)))
    dec_digits = " ".join(_card(int(d)) for d in m.group(2))
    return f"{int_part} Komma {dec_digits}"


def _replace_thousands(m: re.Match) -> str:
    clean = re.sub(r"[.\s]", "", m.group(0))
    return _card(int(clean))


def normalize(text: str) -> str:
    if pd.isna(text):
        return text

    # ── Unsichtbare / typografische Zeichen ───────────────────────────────────
    text = text.replace("\xad", "")                         # weiches Trennzeichen
    for ch in ("«", "»", "‹", "›", "‘", "’", "̈"):
        text = text.replace(ch, "")

    # ── Symbole mit Zahl davor: "30%" → "30 Prozent" ──────────────────────────
    text = re.sub(r"(\d)\s*%", r"\1 Prozent", text)

    # ── Einheiten / Abkürzungen ────────────────────────────────────────────────
    text = re.sub(r"\bCHF\b", "Franken", text)
    text = re.sub(r"\bFr\.\s*", "Franken ", text)
    text = re.sub(r"\bPS\b", "Pferdestärken", text)
    text = re.sub(r"\bMio\.\s*", "Millionen ", text)
    text = re.sub(r"\bMrd\.\s*", "Milliarden ", text)
    text = re.sub(r"\bkm\b", "Kilometer", text)
    text = re.sub(r"\bkg\b", "Kilogramm", text)
    text = re.sub(r"\b%\b", "Prozent", text)              # allein stehendes %
    text = text.replace("§", "Paragraph ")
    text = text.replace("&", " und ")
    text = text.replace("€", " Euro ")
    text = text.replace("$", " Dollar ")

    # ── Ordinalzahlen: "2. Generation" / "sein 18." ───────────────────────────
    # Zahl + Punkt + Leerzeichen + Grossbuchstabe (vor Nomen)
    text = re.sub(
        r"\b(\d+)\.\s+([A-ZÄÖÜ])",
        lambda m: _ord(int(m.group(1))) + " " + m.group(2),
        text,
    )
    # Zahl + Punkt am Satzende
    text = re.sub(
        r"\b(\d+)\.\s*$",
        lambda m: _ord(int(m.group(1))),
        text,
    )

    # ── Dezimalzahlen mit deutschem Komma: "1,5" → "eins Komma fünf" ──────────
    text = re.sub(r"\b(\d+),(\d+)\b", _replace_decimal, text)

    # ── Tausendertrennzeichen: "10 000" oder "1.000" → "zehntausend" ──────────
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", _replace_thousands, text)
    text = re.sub(r"\b\d{1,3}(?: \d{3})+\b", _replace_thousands, text)

    # ── Restliche Kardinalzahlen ───────────────────────────────────────────────
    text = re.sub(r"\b(\d+)\b", lambda m: _card(int(m.group(1))), text)

    # ── Mehrfache Leerzeichen bereinigen ──────────────────────────────────────
    text = re.sub(r" {2,}", " ", text).strip()

    return text


def main():
    df = pd.read_csv(INPUT)
    original = df["sentence"].copy()

    df["sentence"] = df["sentence"].apply(normalize)

    changed = (df["sentence"] != original).sum()
    print(f"  {len(df)} Sätze verarbeitet")
    print(f"  {changed} Sätze verändert")

    # Beispiele ausgeben
    mask = df["sentence"] != original
    for orig, norm in zip(original[mask].head(8), df["sentence"][mask].head(8)):
        print(f"  VOR:  {orig}")
        print(f"  NACH: {norm}")
        print()

    df.to_csv(OUTPUT, index=False)
    print(f"  Gespeichert → {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
