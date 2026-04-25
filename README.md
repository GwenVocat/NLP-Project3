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
