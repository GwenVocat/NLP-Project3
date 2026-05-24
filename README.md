# NLP Project 3 – Ostschweizer Dialekt-Mapping

Forschungsfrage:
«Wie gut lassen sich automatisch erzeugte IPA-Transkriptionen des Ostschweizer Dialekts hochdeutschen Wörtern zuordnen?»

---

## Überblick

```
Audio (mp3)
    │
    ▼
src/01_transcription/transcribe.py
                      IPA-Whisper + Swiss-Whisper → transcriptions.csv
    │
    ▼
src/02_preprocessing/clean.py
                      Bereinigung & Filterung → transcriptions_clean.csv
    │
    ▼
src/03_matching/build_mapping.py
                      Greedy/PMI-Mapping → ostschweiz_mapping_results.csv
    │
    ▼
src/04_evaluation/annotate_mappings.py
                      Manuelle Mapping-Annotation → annotation_candidates.csv
```

---

## IPA-Ausspracheführer

Kurze Referenz der häufigsten IPA-Zeichen im Schweizerdeutschen/Deutschen:

### Konsonanten

| Zeichen | Aussprache | Beispiel |
|---|---|---|
| `ʃ` | **sch** | **Sch**ule, **sch**ön |
| `ʒ` | weiches **sch** (wie frz. *j*) | Gara**g**e, Genie |
| `ç` | **ch** nach hellen Vokalen (i, e, ä) | i**ch**, rei**ch** |
| `x` | **ch** nach dunklen Vokalen (a, o, u) | Ba**ch**, Bu**ch** |
| `ŋ` | **ng** | si**ng**en, la**ng** |
| `ts` | **z** | **Z**eit, Pflan**z**e |
| `tʃ` | **tsch** | Deu**tsch**, **tsch**üss |
| `dʒ` | weiches **dsch** | **Dsch**ungel |
| `ʁ` / `r` | **r** (uvular oder gerollt) | **R**egen, Fah**r**t |
| `j` | **j** | **j**a, **J**ahr |
| `v` | **w** | **W**asser, **w**enn |
| `β` | stimmhaftes b/v (zwischen b und w) | — |
| `ʔ` | Knacklaut (Glottalstop) | be**-**achten |

### Vokale

| Zeichen | Aussprache | Beispiel |
|---|---|---|
| `iː` | langes **ie** | L**ie**be |
| `ɪ` | kurzes **i** | m**i**t, b**i**tte |
| `uː` | langes **u** | **U**hr |
| `ʊ` | kurzes **u** | M**u**tter, H**u**nd |
| `eː` | langes **e** | **E**he, s**ee** |
| `ɛ` | kurzes **e** / **ä** | B**e**tt, H**ä**nde |
| `aː` | langes **a** | V**a**ter |
| `a` | kurzes **a** | St**a**dt |
| `oː` | langes **o** | **O**hr, gr**o**ß |
| `ɔ` | kurzes **o** | S**o**nne |
| `øː` | langes **ö** | h**ö**ren, sch**ö**n |
| `œ` | kurzes **ö** | H**ö**lle, zwölf |
| `yː` | langes **ü** | **ü**ber |
| `ʏ` | kurzes **ü** | h**ü**bsch, fünf |
| `ə` | **Schwa** – unbetontes e | bitt**e**, hab**e** |
| `ɐ` | abgeschwächtes **a/er** | bess**er**, üb**er** |

### Diakritika & Marker

| Zeichen | Bedeutung |
|---|---|
| `ˈ` | Hauptbetonung (folgende Silbe ist betont) |
| `ˌ` | Nebenbetonung |
| `ː` | Längezeichen (langer Vokal/Konsonant) |

> **Hinweis:** In diesem Projekt werden Stressmarker (`ˈ ˌ`) durch `clean.py` entfernt.

---

## Ordnerstruktur

```text
src/
├── 01_transcription/      Audio → Transkriptionen
├── 02_preprocessing/      Cleaning, Normalisierung, Tempus-Erkennung
├── 03_matching/           Matching-Methoden
└── 04_evaluation/         Evaluation-Skripte

notebooks/
├── 01_raw_...             Rohdaten-/Whisper-Checks
├── 02_preprocessing_...   Preprocessing-Analysen
└── 03_matching_...        Matching- und Fehleranalysen

Data/                      Eingabe-, Zwischen- und Ergebnisdateien
```

---

## Daten

| Datei | Inhalt |
|---|---|
| `Data/test.tsv` | Mozilla Common Voice – Ostschweiz Testset |
| `Data/transcriptions.csv` | Rohe Whisper-Ausgaben (IPA + HD) |
| `Data/transcriptions_clean.csv` | Bereinigtes Korpus (Ostschweiz, gefiltert) |
| `Data/ostschweiz_mapping_results.csv` | Automatische IPA↔HD Kookkurrenz-Paare |
| `Data/ostschweiz_mapping_phrases.csv` | Phrase-Level HD→IPA n-gram Mapping |
| `Data/annotation_candidates.csv` | Eindeutige HD→IPA-Kandidaten für manuelle Evaluation |

---

## Skripte

### `src/01_transcription/transcribe.py` – Audio → IPA + Hochdeutsch

Transkribiert alle Ostschweizer Clips mit zwei Whisper-Modellen sequenziell (RAM-schonend: je Modell laden, transkribieren, entladen).

**Modelle:**
- [`neurlang/ipa-whisper-base`](https://huggingface.co/neurlang/ipa-whisper-base) – Audio direkt → IPA
- [`Flurin17/whisper-large-v3-turbo-swiss-german`](https://huggingface.co/Flurin17/whisper-large-v3-turbo-swiss-german) – Audio → Hochdeutsch
- `espeak-ng` via `phonemizer` – Hochdeutsch-Text → IPA (Referenz)

```bash
.venv/bin/python src/01_transcription/transcribe.py
```

Output: `Data/transcriptions.csv`, `Data/errors.csv`

---

### `src/02_preprocessing/clean.py` – Datenbereinigung

Filtert und normalisiert `transcriptions.csv` in mehreren Schritten:

1. Nur `dialect_region == "Ostschweiz"`
2. Fehlerhafte Clips (`errors.csv`) ausschliessen
3. Zu kurze IPA-Felder entfernen (< 3 Zeichen)
4. Garbled-Output erkennen (< 20 % echte IPA-Zeichen in `ipa_audio`)
5. Repetitiven Output erkennen (Muster ≥ 4× wiederholt)
6. IPA normalisieren (Stressmarker `ˈ ˌ` entfernen, Whitespace bereinigen)

```bash
.venv/bin/python src/02_preprocessing/clean.py
```

Output: `Data/transcriptions_clean.csv`

---

### `src/03_matching/build_mapping.py` – greedy PMI-Mapping

Hauptmethode mit iterativem HD→IPA-Mapping über Kookkurrenz, beidseitige
Rate und PMI.

```bash
.venv/bin/python src/03_matching/build_mapping.py
```

Output: `Data/ostschweiz_mapping_results.csv`, `Data/ostschweiz_remainder.csv`

### `src/03_matching/build_mapping_positional.py` – positionale Baseline

Vergleicht HD- und IPA-Tokens an gleicher Position.

```bash
.venv/bin/python src/03_matching/build_mapping_positional.py
```

Output: `Data/ostschweiz_mapping_positional.csv`

### `src/03_matching/build_mapping_ibm.py` – IBM/EM-Alignment

Lernt `P(IPA | HD)` aus Satzpaaren mit EM und leichtem Positions-Prior.

```bash
.venv/bin/python src/03_matching/build_mapping_ibm.py
```

Output: `Data/ostschweiz_mapping_ibm.csv`

### `src/03_matching/build_mapping_phrases.py` – Phrase-/n-gram-Mapping

Vergleicht HD-n-grams und IPA-n-grams der Länge 1 bis 3 an ähnlicher
Satzposition. Diese explorative Methode soll Mehrwort- und Kontraktionsfälle
finden, z.B. `gibt es -> gits`.

```bash
.venv/bin/python src/03_matching/build_mapping_phrases.py
```

Output: `Data/ostschweiz_mapping_phrases.csv`

---

### `src/04_evaluation/annotate_mappings.py` – Manuelle Mapping-Annotation

Terminal-Tool zur Annotation eindeutiger HD→IPA-Mapping-Paare. Gleiche Paare
aus mehreren Methoden werden nur einmal annotiert und dann für alle beteiligten
Methoden ausgewertet.

Kandidaten-Datei erstellen:

```bash
.venv/bin/python src/04_evaluation/annotate_mappings.py --prepare
```

Interaktiv annotieren:

```bash
.venv/bin/python src/04_evaluation/annotate_mappings.py
```

Annotation zusammenfassen:

```bash
.venv/bin/python src/04_evaluation/annotate_mappings.py --summary
```

| Eingabe | Aktion |
|---|---|
| `1` | Mapping korrekt |
| `0` | Mapping falsch |
| `?` | Unsicher |
| `s` | Kandidat überspringen |
| `q` | Beenden |

Output: `Data/annotation_candidates.csv`

### `src/04_evaluation/evaluate_annotations.py` – Evaluationszahlen

Berechnet aus der manuellen Annotation Precision und Coverage pro Methode
sowie eine Konsens-Auswertung.

```bash
.venv/bin/python src/04_evaluation/evaluate_annotations.py
```

Output:
- `Data/evaluation_method_summary.csv`
- `Data/evaluation_consensus_summary.csv`
- `Data/evaluation_error_examples.csv`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Notebooks

| Notebook | Zweck |
|---|---|
| `notebooks/01_raw_check_ipa.ipynb` | Qualitätsprüfung der IPA-Transkriptionen |
| `notebooks/01_raw_test_whisper_settings_ostschweiz.ipynb` | Experimente mit Whisper-Parametern |
| `notebooks/02_preprocessing_hd_wordfreq_analysis.ipynb` | Wortfrequenzen und Preprocessing-Checks |
| `notebooks/03_matching_analysis.ipynb` | Explorative Analyse des Korpus und Mapping-Ergebnisse |
| `notebooks/03_matching_comparison_mapping.ipynb` | Vergleich der Matching-Methoden |
| `notebooks/03_matching_phrase_analysis.ipynb` | Analyse der Phrase-/n-gram-Mapping-Kandidaten |
| `notebooks/03_matching_ibm_analysis.ipynb` | Auswertung des IBM/EM-Mappings |
| `notebooks/03_matching_remainder_analysis.ipynb` | Analyse der nicht gemappten Tokens |
| `notebooks/03_matching_unmatched_analysis.ipynb` | Analyse nicht gematchter HD-Wörter |
| `notebooks/04_evaluation_annotation_analysis.ipynb` | Finale Auswertung der manuellen Annotation |

---

## Paper

Erster Manuskript-Draft:

```text
paper/paper_draft.md
```

Der Draft ist auf einen ACL-style Short Paper Aufbau ausgelegt
(max. 4 Seiten plus Referenzen/Figuren).
