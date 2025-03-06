import os
import glob
import logging
import librosa
import numpy as np
import deepspeech
import pandas as pd
import re
from tqdm import tqdm
from jiwer import wer, cer, mer, wil, wip

# ---------------------- CONFIGURATION ----------------------
# Paths to the DeepSpeech model and scorer files.
MODEL_FILE = "deepspeech-0.9.3-models.pbmm"  # Update with the correct path to your DeepSpeech model
SCORER_FILE = "deepspeech-0.9.3-models.scorer"  # Update with the correct path to your external scorer file

DATASET_PATH = os.path.normpath("C:/Users/A-plus store/OneDrive - MOE Stem Schools/Desktop/Testing And Evaluating Models/Dataset")
CSV_OUTPUT = "DeepSpeech_Evaluation_results.csv"
BATCH_SIZE = 4
MIN_SAMPLES = 400  # At 16kHz, 400 samples ~25ms of audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------- LOAD MODEL ----------------------
def load_model():
    logging.info("🔄 Loading DeepSpeech model and external scorer...")
    model = deepspeech.Model(MODEL_FILE)
    model.enableExternalScorer(SCORER_FILE)
    logging.info("✅ DeepSpeech model and scorer loaded successfully.")
    return model

model = load_model()

# ---------------------- AUDIO PREPROCESSING ----------------------
def preprocess_audio(audio_path):
    """
    Loads, trims, and converts an audio file to 16-bit PCM format for DeepSpeech.
    DeepSpeech requires audio sampled at 16kHz in 16-bit PCM.
    """
    try:
        # Load the audio file with librosa at 16kHz
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Trim leading and trailing silence
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        # If audio is shorter than MIN_SAMPLES, pad it
        if len(waveform) < MIN_SAMPLES:
            waveform = np.pad(waveform, (0, MIN_SAMPLES - len(waveform)))
        # Convert the floating-point waveform (-1.0 to 1.0) to int16 PCM
        waveform_int16 = (waveform * 32767).astype(np.int16)
        return waveform_int16
    except Exception as e:
        logging.warning(f"⚠️ Error processing {audio_path}: {e}")
        return None

# ---------------------- TRANSCRIPTION FUNCTION ----------------------
def transcribe_audio(audio_path):
    """
    Transcribes a single audio file using DeepSpeech.
    Since DeepSpeech does not support batch processing natively,
    each file is processed one-by-one.
    """
    audio_data = preprocess_audio(audio_path)
    if audio_data is None:
        return ""
    try:
        transcription = model.stt(audio_data)
        return transcription
    except Exception as e:
        logging.warning(f"⚠️ Error transcribing {audio_path}: {e}")
        return ""

def transcribe_batch(audio_paths):
    """
    Processes a batch of audio files sequentially.
    Batching here is used solely for organizing evaluation.
    """
    transcriptions = {}
    for path in audio_paths:
        transcriptions[path] = transcribe_audio(path)
    return transcriptions

# ---------------------- TEXT NORMALIZATION ----------------------
def normalize_text(text):
    """
    Normalizes text by converting to uppercase and removing special characters.
    This ensures consistency when comparing actual and predicted texts.
    """
    return re.sub(r'[^A-Z0-9 ]', '', text.upper().strip())

# ---------------------- SESSION EVALUATION ----------------------
def evaluate_session(session_path):
    """
    Processes a session directory, transcribes the corresponding audio files using DeepSpeech,
    and evaluates the performance by comparing against reference prompts.
    Metrics computed include WER, CER, MER, WIL, and WIP.
    """
    results = []
    prompts_dir = os.path.join(session_path, "prompts")
    wav_dir = os.path.join(session_path, "wav_arrayMic")

    if not os.path.isdir(prompts_dir) or not os.path.isdir(wav_dir):
        logging.warning(f"⚠️ Skipping {session_path} due to missing required folders.")
        return results

    # Find common files (by filename without extension) in both folders
    common_files = sorted(set(f[:-4] for f in os.listdir(wav_dir) if f.endswith('.wav')) &
                          set(f[:-4] for f in os.listdir(prompts_dir) if f.endswith('.txt')))
    file_pairs = [(os.path.join(prompts_dir, f"{c}.txt"), os.path.join(wav_dir, f"{c}.wav")) for c in common_files]

    for i in range(0, len(file_pairs), BATCH_SIZE):
        batch_pairs = file_pairs[i:i + BATCH_SIZE]
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
                logging.warning(f"⚠️ No transcription obtained for {af}.")
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
