# NLP Projekt – Projektübersicht für ChatGPT / Codex

Verwende dieses Dokument als vollständigen Kontext für alle Coding-Aufgaben zu diesem Projekt.
Lies es vollständig, bevor du Code schreibst oder Fragen beantwortest.

---

## 1. Forschungsfrage

> «Wie gut lässt sich der dialektspezifische Wortschatz des Ostschweizer Dialekts automatisch
> hochdeutschen Äquivalenten zuordnen – gemessen an Wörtern mittlerer Häufigkeit?»

**Kurs:** Introduction to NLP, Spring 2026
**Team:** Gwen, Nati, Sarruja (betreut von einem Dozenten)
**Deadline:** 15. Juni 2026, Final Paper (max. 4 Seiten, ACL-Format)

---

## 2. Dataset

**Mozilla Common Voice STT4SG-350** – Ostschweiz-Subset
- 3 084 Audio-Clips mit je einem hochdeutschen Referenzsatz
- Gespeichert in `Data/transcriptions_clean.csv` (Hauptinput)
- Erweiterung mit Tempus-Spalte: `Data/transcriptions_tenses.csv`

### Relevante Spalten (transcriptions_tenses.csv)
| Spalte | Beschreibung |
|---|---|
| `path` | Dateipfad zum Audio-Clip |
| `dialect_region` | Dialektregion – wir filtern auf `"Ostschweiz"` |
| `sentence` | Hochdeutsch-Referenzsatz (aus Datensatz) |
| `ipa_audio` | IPA-Transkription des gesprochenen Dialekts (via `neurlang/ipa-whisper-base`) |
| `ipa_reference` | IPA aus Referenztext |
| `ipa_swiss_whisper` | Hochdeutsch-Transkription via `Flurin17/whisper-large-v3-turbo-swiss-german` |
| `tense` | Erkannte grammatische Zeit (Präsens, Perfekt, etc.) |

---

## 3. Pipeline-Übersicht

```
Audio-Clips
    │
    ├──► neurlang/ipa-whisper-base         → ipa_audio  (Dialekt-IPA)
    └──► Flurin17/whisper-large-v3-turbo   → ipa_swiss_whisper (Hochdeutsch)
    
Data/transcriptions_clean.csv
    │
    └──► src/03_matching/build_mapping.py
              │
              ├──► Data/ostschweiz_mapping_results.csv   (Haupt-Mapping-Output)
              └──► Data/ostschweiz_remainder.csv         (nicht gemappte Tokens)
              
Data/ostschweiz_mapping_results.csv + Data/transcriptions_tenses.csv
    │
    └──► annotate.py  →  Data/annotation_sentences.csv
                     →  Data/annotation_results.csv

Data/annotation_results.csv
    │
    └──► eval.py  →  Precision, Recall, weitere Metriken
```

---

## 4. Kern-Skript: src/03_matching/build_mapping.py

### Was es macht
Erstellt automatisch ein HD→IPA-Mapping via **greedy iterativem Alignment**.

### Algorithmus (Schritt für Schritt)

1. Lade `Data/transcriptions_tenses.csv`, filtere auf `dialect_region == "Ostschweiz"`
2. Erstelle **Arbeitskopien** der HD-Sätze und IPA-Sätze (Original bleibt unverändert)
3. **Für jede Runde:**
   - Berechne Kookkurrenz-Statistiken für alle verbleibenden HD–IPA Token-Paare:
     - `hits`: Anzahl Sätze, in denen das Paar gemeinsam vorkommt (via `set()` – pro Satz max. 1 gezählt)
     - `rate`: `hits / Anzahl Sätze mit HD-Token`
     - `PMI`: `log2(P(hd, ipa) / (P(hd) * P(ipa)))` – korrigiert für Corpushäufigkeit
   - Wähle das beste Paar gemäss Ranking (z.B. nach PMI oder kombiniertem Score)
   - **Filtere** nach drei simultanen Schwellenwerten: `Hits ≥ 4`, `Rate ≥ 0.25`, `PMI ≥ 7`
   - Speichere das Paar als Match
   - **Entferne** den HD-Token aus allen HD-Arbeitssätzen
   - **Entferne** den IPA-Token aus allen IPA-Arbeitssätzen
4. Wiederhole bis kein Paar mehr die Schwellenwerte erfüllt

### Warum diese Schwellenwerte?
- **Hits ≥ 4**: Verhindert statistische Bedeutungslosigkeit (1–2 zufällige Kookkurrenzen)
- **Rate ≥ 0.25**: Stellt sicher, dass der IPA-Token in ≥25% der relevanten Sätze vorkommt
- **PMI ≥ 7**: Korrigiert für corpusweite Häufigkeit (analog zu TF-IDF) – verhindert, dass häufige Funktionswörter wie `diː` oder `ʊnt` alles dominieren

### Warum PMI statt nur Rate?
Rate allein bevorzugt seltene Wörter (1 Clip, 1 Treffer → Rate = 1.0).
PMI bestraft Tokens, die im ganzen Corpus sehr häufig sind und deshalb zufällig kookkurrieren.

### Output-Spalten (ostschweiz_mapping_results.csv)
| Spalte | Beschreibung |
|---|---|
| `Hochdeutsch` | HD-Wort |
| `IPA_Dialekt` | IPA-Äquivalent |
| `HD_Gesamt_Häufigkeit` | Wie oft HD-Wort im Corpus vorkommt |
| `IPA_Gesamt_Häufigkeit` | Wie oft IPA-Token im Corpus vorkommt |
| `Gemeinsame_Treffer` | Anzahl Sätze mit Kookkurrenz |
| `Kokkurrenz_Rate` | hits / HD-Häufigkeit |
| `PMI` | Pointwise Mutual Information Score |
| `Runde` | In welcher Iteration das Paar gefunden wurde |

### Aktuell gefundene Paare (Beispiele, Stand Mai 2026)
```
budget → byːtʃə    (Hits: 7, Rate: 0.50, PMI: 7.59, Runde 1)
beiden → baɪdə     (Hits: 6, Rate: 0.375, PMI: 7.01, Runde 2)
geschäft → gʃɛft   (Hits: 6, Rate: 0.60, PMI: 7.68, Runde 3)
gemacht → gmaxt    (Hits: 6, Rate: 0.43, PMI: 7.05, Runde 4)
schweiz → ʃviːts   (Hits: 5, Rate: 0.25, PMI: 7.27, Runde 6)
nichts → nyːd      (Hits: 4, Rate: 0.31, PMI: 7.31, Runde 21)
```
Ca. 23 Paare mit den aktuellen Schwellenwerten – alle linguistisch plausibel.

### Bekannte linguistische Muster in den Matches
- `ge-` Präfix → `g-` (z.B. `gemacht → gmaxt`, `geschäft → gʃɛft`)
- Endung `-en` → `-ə` (z.B. `frauen → fraʊə`, `bleiben → bliːbə`)
- `-ichts` → `-yːd` (klassisch Schweizerdeutsch «nüt»: `nichts → nyːd`)
- Konsonantenwechsel `g → k` (z.B. `genau → knaʊ`)

---

## 5. Weitere Skripte

### src/01_transcription/transcribe.py
- Transkribiert Audio mit beiden Whisper-Modellen
- Output: `Data/transcriptions_clean.csv`

### src/02_preprocessing/clean.py
- Filtert fehlerhafte Transkriptionen (IPA-Zeichenratio-Threshold)
- Entfernt Repetitionen
- Normalisiert IPA (Stressmarker-Entfernung)

### src/02_preprocessing/preprocess_sentences.py
- Normalisiert Zahlen, Symbole und typografische Zeichen in HD-Sätzen
- Output: `Data/transcriptions_normalized.csv`

### src/02_preprocessing/preprocess_tenses.py
- Klassifiziert HD-Sätze nach Tempus
- Output: `Data/transcriptions_tenses.csv`

### src/03_matching/build_mapping_positional.py
- Positionale Baseline: HD-Token und IPA-Token an gleicher Position
- Output: `Data/ostschweiz_mapping_positional.csv`

### src/03_matching/build_mapping_ibm.py
- IBM-Model-1/EM-Alignment mit Positions-Prior
- Output: `Data/ostschweiz_mapping_ibm.csv`

### annotate.py
- Terminal-basiertes Annotationstool
- Zeigt Sätze mit mittelhäufigen HD-Wörtern
- Annotator bewertet ob das automatische Mapping korrekt ist
- Speichert in `Data/annotation_sentences.csv` und `Data/annotation_results.csv`
- Bekanntes früheres Problem: String-Mismatch zwischen HD-Tokens in `sentence` und Mapping-Werten (ob gelöst: unklar)
- Ziel: ~200 annotierte Sätze

### eval.py (noch zu schreiben)
- Liest `Data/annotation_results.csv`
- Berechnet Precision, Recall (und ggf. MRR) auf mittelhäufigen Wörtern
- Ist der Kern der Forschungsfrage – ohne dieses Skript gibt es kein Resultat für das Paper

### notebooks/03_matching_remainder_analysis.ipynb
- Jupyter Notebook zur Analyse der nicht gemappten Tokens
- Untersucht: Tempus-Verteilung, Wortposition im Satz, Muster bei ungemappten HD- und IPA-Tokens
- Input: `Data/ostschweiz_remainder.csv`

---

## 6. Remainder-Export (Data/ostschweiz_remainder.csv)

`src/03_matching/build_mapping.py` exportiert am Ende zusätzlich alle Sätze **mit den gemappten Tokens entfernt**.
Struktur identisch zu `transcriptions_tenses.csv`:
`path, dialect_region, sentence, ipa_reference, ipa_audio, ipa_swiss_whisper, tense`
– aber HD- und IPA-Tokens die bereits gemappt wurden, sind aus den Sätzen entfernt.

---

## 7. Design-Entscheidungen (nicht ändern!)

Diese Entscheidungen wurden bewusst getroffen und sollen **nicht** umgangen werden:

| Entscheidung | Begründung |
|---|---|
| Keine Funktionswort-Filterung | Empirischer Ansatz – statistische Schwellen reichen aus |
| Keine explizite Frequenzband-Filterung im Mapping | Schwellen (Hits, Rate, PMI) ersetzen den Filter |
| Greedy iteratives Entfernen auch bei n:m-Mapping | Reduziert Rauschen in späteren Runden |
| IPA via `set()` pro Satz zählen | Verhindert Mehrfachzählung bei wiederholten Tokens im selben Satz |
| Frequenz aus `sentence`-Spalte (HD), nicht aus IPA | IPA-Whisper tokenisiert anders → Vergleich nicht verlässlich |

---

## 8. Offene Aufgaben (Stand Mai 2026)

- [ ] `src/04_evaluation/eval.py` schreiben (Precision, Recall auf `annotation_results.csv`)
- [ ] Annotation auf ~200 Sätze vervollständigen (`annotate.py`)
- [ ] `notebooks/03_matching_remainder_analysis.ipynb` vervollständigen (Muster im Remainder)
- [ ] Final Paper schreiben (ACL-Format, max. 4 Seiten, Deadline 15. Juni 2026)

---

## 9. Projektverzeichnis

```
NLP-Project3/
├── src/
│   ├── 01_transcription/
│   │   └── transcribe.py
│   ├── 02_preprocessing/
│   │   ├── clean.py
│   │   ├── preprocess_sentences.py
│   │   └── preprocess_tenses.py
│   ├── 03_matching/
│   │   ├── build_mapping.py
│   │   ├── build_mapping_positional.py
│   │   └── build_mapping_ibm.py
│   └── 04_evaluation/
├── notebooks/
│   ├── 01_raw_check_ipa.ipynb
│   ├── 01_raw_test_whisper_settings_ostschweiz.ipynb
│   ├── 02_preprocessing_hd_wordfreq_analysis.ipynb
│   ├── 03_matching_analysis.ipynb
│   ├── 03_matching_comparison_mapping.ipynb
│   ├── 03_matching_ibm_analysis.ipynb
│   ├── 03_matching_remainder_analysis.ipynb
│   └── 03_matching_unmatched_analysis.ipynb
├── Data/
│   ├── transcriptions_clean.csv
│   ├── transcriptions_tenses.csv
│   ├── ostschweiz_mapping_results.csv
│   ├── ostschweiz_mapping_positional.csv
│   ├── ostschweiz_mapping_ibm.csv
│   ├── ostschweiz_remainder.csv
│   ├── annotation_sentences.csv
│   └── annotation_results.csv
```

**Umgebung:** Python 3.12, pyenv, `.venv`, Mac
