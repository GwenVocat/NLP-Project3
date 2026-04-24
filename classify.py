import pandas as pd
from collections import defaultdict, Counter
import re

print("1. Lade Daten und filtere nach Ostschweiz...")
df = pd.read_csv("Data/transcriptions_clean.csv")

# FOR TESTING: Assuming df_ost is already created as in your original code
df_ost = df[df["dialect_region"] == "Ostschweiz"].copy()

word_mapping = defaultdict(Counter)

print(f"2. Analysiere Sätze auf Kookkurrenzen...")

for index, row in df_ost.iterrows():
    hg_sentence = str(row["sentence"]).lower()
    hg_sentence = re.sub(r'[^\w\s]', '', hg_sentence)
    hg_words = hg_sentence.split()

    ipa_sentence = str(row["ipa_audio"])
    ipa_words = ipa_sentence.split()

    for ipa_w in ipa_words:
        if len(ipa_w) > 1:
            for hg_w in hg_words:
                word_mapping[ipa_w][hg_w] += 1

print("\n3. Extrahiere ALLE systematischen Zuordnungen...")
systematic_mapping = []

# MIN_HITS bestimmt, wie oft ein IPA-Wort und ein Hochdeutsches Wort
# gemeinsam in einem Satz auftauchen müssen, um gespeichert zu werden.
# Das filtert reines Rauschen (1-2 zufällige Treffer) heraus.
MIN_HITS = 5

for ipa_word, hg_counts in word_mapping.items():
    # Anstatt nur das häufigste (most_common) zu nehmen, gehen wir durch ALLE
    for hg_word, count in hg_counts.items():
        if count >= MIN_HITS:
            systematic_mapping.append({
                "IPA_Dialekt": ipa_word,
                "Hochdeutsch_Zuordnung": hg_word,
                "Gemeinsame_Treffer": count
            })

results_df = pd.DataFrame(systematic_mapping)

# Optional: Sortieren, damit die stärksten Paare oben stehen
results_df = results_df.sort_values(by="Gemeinsame_Treffer", ascending=False)

print("\nTop 10 Ergebnisse:")
print(results_df.head(10))

results_df.to_csv("Data/ostschweiz_mapping_results.csv", index=False)
print(f"\nErfolgreich {len(results_df)} Kombinationen gespeichert!")