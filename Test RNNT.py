# Monkey-patch: Define SIGKILL on Windows if missing
import signal
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = 9  # Use Unix value for SIGKILL

import os
import glob
import logging
import librosa
import numpy as np
import torch
import pandas as pd
import re
from tqdm import tqdm
from jiwer import wer, cer, mer, wil, wip
import nemo.collections.asr as nemo_asr

# ---------------------- CONFIGURATION ----------------------
MODEL_NAME = "nvidia/parakeet-rnnt-1.1b"  # RNNT model from NVIDIA NeMo
DATASET_PATH = os.path.normpath("C:/Users/A-plus store/OneDrive - MOE Stem Schools/Desktop/Testing And Evaluating Models/Dataset")  # Windows-compatible path
CSV_OUTPUT = "RNNT_Evaluation_results.csv"
BATCH_SIZE = 4
MIN_SAMPLES = 400  # At 16kHz, this is 25ms of audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------- LOAD MODEL ----------------------
def load_model():
    logging.info("🔄 Loading RNNT model...")
    # Load the RNNT model using NeMo’s ASR collection
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    logging.info("✅ RNNT model loaded successfully.")
    return model

model = load_model()

# ---------------------- AUDIO PREPROCESSING ----------------------
def preprocess_audio(audio_path):
    """Loads, trims, and normalizes an audio file."""
    try:
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)  # Normalize amplitude
        if len(waveform) < MIN_SAMPLES:
            waveform = np.pad(waveform, (0, MIN_SAMPLES - len(waveform)))
        return waveform
    except Exception as e:
        logging.warning(f"⚠️ Error processing {audio_path}: {e}")
        return None

# ---------------------- TRANSCRIPTION FUNCTION ----------------------
def transcribe_batch(audio_paths):
    """
    Transcribes a batch of audio files using the RNNT model.
    The NeMo RNNT model provides a .transcribe() method that accepts a list of file paths.
    """
    transcriptions = {}
    for path in audio_paths:
        try:
            # The transcribe() function takes a list and returns a list of transcription strings.
            result = model.transcribe([path])
            transcriptions[path] = result[0]
        except Exception as e:
            logging.warning(f"⚠️ Error transcribing {path}: {e}")
            transcriptions[path] = ""
    return transcriptions

# ---------------------- TEXT NORMALIZATION ----------------------
def normalize_text(text):
    """Normalizes text by converting to uppercase and removing special characters."""
    return re.sub(r'[^A-Z0-9 ]', '', text.upper().strip())

# ---------------------- SESSION EVALUATION ----------------------
def evaluate_session(session_path):
    """
    Processes a session directory and evaluates ASR performance using the RNNT model.
    The session is expected to contain a 'prompts' folder with reference text files
    and a 'wav_arrayMic' folder with corresponding .wav files.
    """
    results = []
    prompts_dir = os.path.join(session_path, "prompts")
    wav_dir = os.path.join(session_path, "wav_arrayMic")

    if not os.path.isdir(prompts_dir) or not os.path.isdir(wav_dir):
        logging.warning(f"⚠️ Skipping {session_path} due to missing folders.")
        return results

    # Identify common file identifiers (assuming same filename before extension)
    common_files = sorted(set(f[:-4] for f in os.listdir(wav_dir)) & set(f[:-4] for f in os.listdir(prompts_dir)))
    file_pairs = [(os.path.join(prompts_dir, f"{c}.txt"), os.path.join(wav_dir, f"{c}.wav")) for c in common_files]

    for i in range(0, len(file_pairs), BATCH_SIZE):
        batch_pairs = file_pairs[i:i+BATCH_SIZE]
        batch_prompts, batch_audio = zip(*batch_pairs)

        actual_texts = []
        for pf in batch_prompts:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    actual_texts.append(normalize_text(f.read()))
            except Exception as e:
                logging.warning(f"⚠️ Error reading {pf}: {e}")
                actual_texts.append("")

        predicted_dict = transcribe_batch(batch_audio)

        for pf, af, a_text in zip(batch_prompts, batch_audio, actual_texts):
            if not a_text:
                continue
            p_text = normalize_text(predicted_dict.get(af, ""))
            if not p_text:
                logging.warning(f"⚠️ No transcription for {af}.")
                continue

            wer_score = wer(a_text, p_text)
            cer_score = cer(a_text, p_text)
            mer_score = mer(a_text, p_text)
            wil_score = wil(a_text, p_text)
            wip_score = wip(a_text, p_text)

            result = {
                "Prompt_File": os.path.basename(pf),
                "Audio_File": os.path.basename(af),
                "Actual": a_text,
                "Predicted": p_text,
                "WER": wer_score,
                "CER": cer_score,
                "MER": mer_score,
                "WIL": wil_score,
                "WIP": wip_score,
            }
            results.append(result)

    return results

# ---------------------- MAIN FUNCTION ----------------------
def main():
    all_results = []
    # Assume session directories follow a pattern; adjust the glob pattern as needed.
    session_dirs = glob.glob(os.path.join(DATASET_PATH, "*", "*", "session_*"))
    logging.info(f"📂 Found {len(session_dirs)} session directories.")

    for session in tqdm(session_dirs, desc="🔍 Processing sessions"):
        all_results.extend(evaluate_session(session))

    if all_results:
        os.makedirs(os.path.dirname(CSV_OUTPUT) or ".", exist_ok=True)
        pd.DataFrame(all_results).to_csv(CSV_OUTPUT, index=False)
        logging.info(f"✅ Evaluation results saved to {CSV_OUTPUT}")
    else:
        logging.info("⚠️ No results to save.")

if __name__ == '__main__':
    main()
