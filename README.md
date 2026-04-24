# NLP Project 3 – Ostschweizer Dialekt-Mapping

Forschungsfrag:
«Wie gut lässt sich der dialektspezifische Wortschatz des Ostschweizer Dialekts automatisch hochdeutschen Äquivalenten zuordnen – gemessen an Wörtern mittlerer Häufigkeit?»

---

## Überblick

```
Audio (mp3)
    │
    ▼
transcribe.py         IPA-Whisper + Swiss-Whisper → transcriptions.csv
    │
    ▼
clean.py              Bereinigung & Filterung → transcriptions_clean.csv
    │
    ▼
classify.py           Kookkurrenz-Mapping IPA↔HD → ostschweiz_mapping_results.csv
    │
    ▼
annotate.py           Manuelle Ground-Truth-Annotation (Terminal)
```

---

## Daten

| Datei | Inhalt |
|---|---|
| `Data/test.tsv` | Mozilla Common Voice – Ostschweiz Testset |
| `Data/transcriptions.csv` | Rohe Whisper-Ausgaben (IPA + HD) |
| `Data/transcriptions_clean.csv` | Bereinigtes Korpus (Ostschweiz, gefiltert) |
| `Data/ostschweiz_mapping_results.csv` | Automatische IPA↔HD Kookkurrenz-Paare |
| `Data/annotation_sentences.csv` | 200 zufällig gesampelte Sätze für Annotation |
| `Data/annotation_results.csv` | Annotierte Ground-Truth-Einträge |

---

## Skripte

### `transcribe.py` – Audio → IPA + Hochdeutsch

Transkribiert alle Ostschweizer Clips mit zwei Whisper-Modellen sequenziell
(RAM-schonend: je Modell laden, transkribieren, entladen).

**Modelle:**
- [`neurlang/ipa-whisper-base`](https://huggingface.co/neurlang/ipa-whisper-base) – Audio direkt → IPA
- [`Flurin17/whisper-large-v3-turbo-swiss-german`](https://huggingface.co/Flurin17/whisper-large-v3-turbo-swiss-german) – Audio → Hochdeutsch
- `espeak-ng` via `phonemizer` – Hochdeutsch-Text → IPA (Referenz)

```bash
python transcribe.py
```

Output: `Data/transcriptions.csv`, `Data/errors.csv`

---

### `clean.py` – Datenbereinigung

Filtert und normalisiert `transcriptions.csv` in mehreren Schritten:

1. Nur `dialect_region == "Ostschweiz"`
2. Fehlerhafte Clips (`errors.csv`) ausschliessen
3. Zu kurze IPA-Felder entfernen (< 3 Zeichen)
4. Garbled-Output erkennen (< 20 % echte IPA-Zeichen in `ipa_audio`)
5. Repetitiven Output erkennen (Muster ≥ 4× wiederholt)
6. IPA normalisieren (Stressmarker `ˈ ˌ` entfernen, Whitespace bereinigen)

```bash
python clean.py
```

Output: `Data/transcriptions_clean.csv`

---

### `classify.py` – Kookkurrenz-Mapping

Analysiert alle Satzpaare (IPA-Audio ↔ Hochdeutsch) auf Wortebene.
Ein IPA-Wort und ein HD-Wort werden gespeichert, wenn sie in ≥ 5 Sätzen
gemeinsam auftreten (`MIN_HITS = 5`).

```bash
python classify.py
```

Output: `Data/ostschweiz_mapping_results.csv`

Spalten: `IPA_Dialekt`, `Hochdeutsch_Zuordnung`, `Gemeinsame_Treffer`

---

### `annotate.py` – Manuelle Ground-Truth-Annotation

Terminal-Tool zur wortweisen Annotation von IPA-Dialektwörtern mit
ihrer korrekten Hochdeutsch-Entsprechung im Satzkontext.

**Beim ersten Start** werden automatisch:
- 200 Sätze zufällig gesampelt (`random_state=42`) → `annotation_sentences.csv`
- Leere `annotation_results.csv` angelegt
- Fortschrittsdatei `annotation_progress.json` initialisiert

**Fortschritt bleibt erhalten** – bei erneutem Start wird dort weitergemacht,
wo aufgehört wurde.

```bash
python annotate.py
```

**Befehle während der Annotation:**

| Eingabe | Aktion |
|---|---|
| `Enter` | Auto-Mapping übernehmen (`auto_correct = True`) |
| Text | Eigene HD-Übersetzung eingeben |
| `s` | Dieses Wort überspringen |
| `ss` | Ganzen Satz überspringen |
| `q` | Sofort beenden (Fortschritt gespeichert) |

Output: `Data/annotation_results.csv`

Spalten: `sentence_id`, `ipa_word`, `hd_ground_truth`, `auto_mapping`, `auto_correct`, `skipped`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Notebooks

| Notebook | Zweck |
|---|---|
| `analysis.ipynb` | Explorative Analyse des Korpus und Mapping-Ergebnisse |
| `check_ipa.ipynb` | Qualitätsprüfung der IPA-Transkriptionen |
| `test_whisper_settings_ostschweiz.ipynb` | Experimente mit Whisper-Parametern |
