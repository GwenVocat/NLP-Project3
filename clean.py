"""
Schritt 1 – Datenbereinigung: transcriptions.csv → transcriptions_clean.csv

Input:  Data/transcriptions.csv
Output: Data/transcriptions_clean.csv

Filterschritte:
  1. Nur dialect_region == "Ostschweiz"
  2. Zeilen mit errors.csv ausschliessen (falls vorhanden)
  3. Leere / zu kurze IPA-Felder entfernen (< 3 Zeichen)
  4. Garbled-Output-Erkennung: < 20% echte IPA-Zeichen in ipa_audio → raus
  5. Repetitions-Erkennung: Muster von 2–6 Zeichen, ≥ 4× hintereinander → raus
  6. IPA-Normalisierung: ˈ ˌ entfernen, Whitespace normalisieren

Verwendung: python clean.py
"""

import re
import unicodedata
import os
import pandas as pd


# ============================================================
# 1. Daten laden
# ============================================================
df = pd.read_csv("Data/transcriptions.csv")
n_start = len(df)
print(f"Geladen: {n_start:,} Zeilen gesamt")

removed_counts = {}


# ============================================================
# 2. Filter: nur Ostschweiz
# ============================================================
df = df[df["dialect_region"] == "Ostschweiz"].copy()
n_after_region = len(df)
removed_counts["Andere Region"] = n_start - n_after_region
print(f"Nach Regionfilter (Ostschweiz): {n_after_region:,} Zeilen")


# ============================================================
# 3. errors.csv ausschliessen (falls vorhanden)
# ============================================================
errors_path = "Data/errors.csv"
if os.path.exists(errors_path):
    errors = pd.read_csv(errors_path)
    before = len(df)
    # Annahme: errors.csv hat eine 'path'-Spalte als Schlüssel
    if "path" in errors.columns:
        df = df[~df["path"].isin(errors["path"])].copy()
    n_errors = before - len(df)
    removed_counts["In errors.csv"] = n_errors
    print(f"Nach errors.csv-Filter: {len(df):,} Zeilen ({n_errors} entfernt)")
else:
    print("Kein errors.csv gefunden – Schritt übersprungen")


# ============================================================
# 4. Leere / zu kurze IPA-Felder entfernen (< 3 Zeichen)
# ============================================================
IPA_COLS = ["ipa_audio", "ipa_reference", "ipa_swiss_whisper"]

for col in IPA_COLS:
    df[col] = df[col].fillna("")

before = len(df)
mask_short = (
    (df["ipa_audio"].str.len() < 3)
    | (df["ipa_reference"].str.len() < 3)
    | (df["ipa_swiss_whisper"].str.len() < 3)
)
df = df[~mask_short].copy()
n_short = before - len(df)
removed_counts["Zu kurze IPA-Felder"] = n_short
print(f"Nach Kurzfilter (< 3 Zeichen): {len(df):,} Zeilen ({n_short} entfernt)")


# ============================================================
# 5. Garbled-Output-Erkennung: Anteil IPA-Zeichen in ipa_audio
# ============================================================
IPA_SPECIFIC = set("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸœɶʘɹɺɾɻʀʁɽʂʃʈθʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˡˢˣˤ")

def ipa_char_ratio(text):
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    ipa_chars = [c for c in non_space if c in IPA_SPECIFIC]
    return len(ipa_chars) / len(non_space)


before = len(df)
df["_ipa_ratio"] = df["ipa_audio"].apply(ipa_char_ratio)
mask_garbled = df["_ipa_ratio"] < 0.20
df = df[~mask_garbled].copy()
df = df.drop(columns=["_ipa_ratio"])
n_garbled = before - len(df)
removed_counts["Garbled ipa_audio (< 20% IPA-Zeichen)"] = n_garbled
print(f"Nach Garbled-Filter: {len(df):,} Zeilen ({n_garbled} entfernt)")


# ============================================================
# 6. Repetitions-Erkennung in ipa_audio
# ============================================================
def has_repetition(text: str, min_len: int = 2, max_len: int = 6, min_reps: int = 4) -> bool:
    """True wenn ein Muster von min_len–max_len Zeichen ≥ min_reps-mal hintereinander vorkommt."""
    if not isinstance(text, str) or not text:
        return False
    for n in range(min_len, max_len + 1):
        if re.search(r'(.{' + str(n) + r'})\1{' + str(min_reps - 1) + r',}', text):
            return True
    return False


before = len(df)
mask_rep = df["ipa_audio"].apply(has_repetition)
df = df[~mask_rep].copy()
n_rep = before - len(df)
removed_counts["Repetitiver ipa_audio-Output"] = n_rep
print(f"Nach Repetitions-Filter: {len(df):,} Zeilen ({n_rep} entfernt)")


# ============================================================
# 7. IPA-Normalisierung: Stressmarker entfernen, Whitespace normalisieren
# ============================================================
STRESS_REMOVE = str.maketrans("", "", "ˈˌ")

for col in IPA_COLS:
    df[col] = (
        df[col]
        .str.translate(STRESS_REMOVE)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

print(f"IPA normalisiert (ˈ ˌ  entfernt, Whitespace bereinigt)")


# ============================================================
# 8. Speichern
# ============================================================
df.to_csv("Data/transcriptions_clean.csv", index=False)
print(f"\nGespeichert: Data/transcriptions_clean.csv")


# ============================================================
# Zusammenfassung
# ============================================================
n_end = len(df)
print("\n" + "=" * 50)
print("Zusammenfassung")
print("=" * 50)
print(f"  Zeilen vorher  (gesamt):    {n_start:>6,}")
print(f"  Zeilen nachher (Ostschweiz bereinigt): {n_end:>6,}")
print(f"  Entfernt gesamt: {n_start - n_end:,}")
print()
print("  Aufschlüsselung:")
for reason, count in removed_counts.items():
    print(f"    {reason:<45} {count:>5,}")
