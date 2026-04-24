"""
Transkription Ostschweizer Dialektaufnahmen – OPTIMIERTE VERSION

Optimierungen gegenüber transcribe.py:
  1. Sequenzielles Modell-Laden (halber RAM: nur 1 Modell zur Zeit)
  2. Phonemizer gebatcht am Schluss (2 Aufrufe statt ~7'000)
  3. Einzel-Clip-Inferenz auf MPS (stabil, kein Memory-Leak)

Verwendung: python transcribe_fast.py
"""

import gc
import os
import time
import warnings

import pandas as pd
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from phonemizer import phonemize
from phonemizer.separator import Separator

warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

# ============================================================
# Konfiguration
# ============================================================
DATA_TSV   = "Data/test.tsv"
CLIPS_DIR  = "Data/clips__test"
OUTPUT_CSV = "Data/transcriptions.csv"
ERRORS_CSV = "Data/errors.csv"

MODEL_IPA_WHISPER   = "neurlang/ipa-whisper-base"
MODEL_SWISS_WHISPER = "Flurin17/whisper-large-v3-turbo-swiss-german"

PHONE_SEP = Separator(phone="", word=" ", syllable="")

# ============================================================
# Device
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("Device: CPU (wird langsam!)")

# ============================================================
# Hilfsfunktionen
# ============================================================

def free_model():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_single(y, processor, model, forced_ids=None) -> str:
    """Ein einzelner Clip → Text (stabil auf MPS)."""
    features = processor(y, sampling_rate=16000, return_tensors="pt").input_features
    features = features.to(device=device, dtype=model.dtype)

    gen_kwargs = {}
    if forced_ids is not None:
        gen_kwargs["forced_decoder_ids"] = forced_ids

    with torch.no_grad():
        ids = model.generate(features, **gen_kwargs)

    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def batch_text_to_ipa(texts: list[str]) -> list[str]:
    """Alle Texte → IPA in einem einzigen espeak-ng-Aufruf."""
    clean = [t if isinstance(t, str) and t.strip() else "." for t in texts]
    results = phonemize(
        clean,
        language="de",
        backend="espeak",
        separator=PHONE_SEP,
        strip=True,
        preserve_punctuation=False,
    )
    return [
        r.strip() if (isinstance(t, str) and t.strip()) else ""
        for r, t in zip(results, texts)
    ]


def run_whisper_phase(df, model_name, phase_label, forced_ids_fn=None):
    """Lädt Modell, transkribiert alle Clips einzeln, entlädt Modell."""

    print(f"\n{'=' * 60}")
    print(f"{phase_label}")
    print(f"{'=' * 60}")

    print(f"\nLade {model_name} ...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    forced_ids = forced_ids_fn(processor) if forced_ids_fn else None
    print("Modell geladen – starte Transkription ...\n")

    transcriptions = {}
    errors = []
    total = len(df)
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        audio_path = os.path.join(CLIPS_DIR, row["path"])
        try:
            y, _ = librosa.load(audio_path, sr=16000)
            txt = transcribe_single(y, processor, model, forced_ids)
            transcriptions[row["path"]] = txt
        except Exception as e:
            errors.append({"path": row["path"], "error": str(e)})

        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            remain  = (elapsed / (i + 1)) * (total - i - 1)
            print(
                f"  [{i+1:4d}/{total}]  "
                f"{elapsed/60:.1f} min | ~{remain/60:.1f} min verbleibend"
            )

    # Modell entladen
    del model, processor
    free_model()
    print(f"Modell entladen.\n")

    return transcriptions, errors


# ============================================================
# Hauptprogramm
# ============================================================
overall_start = time.time()

# Daten laden
df_full = pd.read_csv(DATA_TSV, sep="\t")
df = df_full[df_full["dialect_region"] == "Ostschweiz"].reset_index(drop=True)
print(f"\nAufnahmen: {len(df):,} Ostschweiz (von {len(df_full):,} gesamt)")

# Phase 1: IPA-Whisper (Audio → IPA)
ipa_audio_map, errors_1 = run_whisper_phase(
    df, MODEL_IPA_WHISPER, "Phase 1/3: IPA-Whisper (Audio → IPA)"
)

# Phase 2: Swiss-Whisper (Audio → Hochdeutsch)
swiss_hd_map, errors_2 = run_whisper_phase(
    df, MODEL_SWISS_WHISPER, "Phase 2/3: Swiss-Whisper (Audio → Hochdeutsch)",
    forced_ids_fn=lambda p: p.get_decoder_prompt_ids(language="german", task="transcribe"),
)

# Phase 3: Phonemizer (gebatcht – 2 Aufrufe statt ~7'000)
print(f"\n{'=' * 60}")
print("Phase 3/3: Phonemizer (Text → IPA)")
print(f"{'=' * 60}")

sentences = [str(row["sentence"]) for _, row in df.iterrows()]
swiss_hds = [swiss_hd_map.get(row["path"], "") for _, row in df.iterrows()]

print("  Referenz-Sätze → IPA ...")
ipa_refs = batch_text_to_ipa(sentences)
print("  Swiss-Whisper-Output → IPA ...")
ipa_swiss = batch_text_to_ipa(swiss_hds)
print("  Fertig.\n")

# Zusammenführen
all_errors  = errors_1 + errors_2
error_paths = {e["path"] for e in all_errors}

results = []
for i, (_, row) in enumerate(df.iterrows()):
    if row["path"] in error_paths:
        continue
    results.append({
        "path":              row["path"],
        "dialect_region":    row["dialect_region"],
        "sentence":          row["sentence"],
        "ipa_reference":     ipa_refs[i],
        "ipa_audio":         ipa_audio_map.get(row["path"], ""),
        "ipa_swiss_whisper": ipa_swiss[i],
    })

# Speichern
df_out = pd.DataFrame(results, columns=[
    "path", "dialect_region", "sentence",
    "ipa_reference", "ipa_audio", "ipa_swiss_whisper",
])
df_out.to_csv(OUTPUT_CSV, index=False)

if all_errors:
    pd.DataFrame(all_errors).to_csv(ERRORS_CSV, index=False)

elapsed_total = time.time() - overall_start
print(f"{'=' * 60}")
print(f"Fertig!")
print(f"   Erfolgreich: {len(results):,} / {len(df):,}")
print(f"   Fehler:      {len(all_errors):,}  {'→ ' + ERRORS_CSV if all_errors else ''}")
print(f"   Dauer:       {elapsed_total / 60:.1f} Minuten")
print(f"   Output:      {OUTPUT_CSV}")
print(f"{'=' * 60}")