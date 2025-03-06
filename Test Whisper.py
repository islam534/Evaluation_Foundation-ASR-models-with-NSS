import os
import glob
import pandas as pd
import numpy as np
import librosa
import torch
import logging
import traceback
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from jiwer import wer, cer, mer, wil  # Standard ASR error metrics
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from bert_score import score as bert_score
from phonemizer import phonemize

# ------------------- CONFIGURATION -------------------
MODEL_ID = "openai/whisper-small"
DATASET_PATH = os.path.normpath(
    "C:/Users/A-plus store/OneDrive - MOE Stem Schools/Desktop/Testing And Evaluating Models/Dataset"
)  # Windows-compatible path
CSV_OUTPUT = "whisper_asr_evaluation_results.csv"
SUMMARY_OUTPUT = "whisper_asr_summary.csv"
LANGUAGE = "en"
SAMPLING_RATE = 16000

# ------------------- LOAD MODEL -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Optimize GPU performance

try:
    print(f"🔄 Loading Whisper model: {MODEL_ID} on {device}")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE)
    print(f"✅ Model loaded successfully. Expected sampling rate: {SAMPLING_RATE}")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    raise

# ------------------- TRANSCRIPTION FUNCTION -------------------
def transcribe_audio(audio_path):
    """Transcribes an audio file using Whisper ASR."""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
        if len(audio) == 0 or np.all(audio == 0) or np.isnan(audio).any():
            print(f"⚠️ Skipping empty/silent file: {audio_path}")
            return None

        # Trim or pad audio
        max_length = SAMPLING_RATE * 30
        if len(audio) < max_length:
            padded_audio = np.zeros(max_length, dtype=np.float32)
            padded_audio[:len(audio)] = audio
            audio = padded_audio
        else:
            audio = audio[:max_length]

        # Prepare input features
        inputs = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                inputs["input_features"],
                max_length=448,
                forced_decoder_ids=forced_decoder_ids,
                attention_mask=inputs.get("attention_mask", None),
            )
        transcription = processor.batch_decode(output, skip_special_tokens=True)
        return transcription[0].strip() if transcription else None

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {str(e)}")
        return None

# ------------------- PROCESS DATASET -------------------
results = []
session_dirs = sorted(glob.glob(os.path.join(DATASET_PATH, "*", "*", "session_*")))
print(f"📂 Found {len(session_dirs)} session directories.")

def process_file_pair(file_pair):
    """Processes a single transcript-audio file pair and evaluates ASR performance."""
    p_file, a_file = file_pair
    try:
        with open(p_file, "r", encoding="utf-8") as f:
            actual_text = f.read().strip()
    except Exception:
        actual_text = ""

    predicted_text = transcribe_audio(a_file)
    if not predicted_text:
        return None

    wer_score_val = wer(actual_text, predicted_text) if actual_text else None
    cer_score_val = cer(actual_text, predicted_text) if actual_text else None
    mer_score_val = mer(actual_text, predicted_text) if actual_text else None
    wil_score_val = wil(actual_text, predicted_text) if actual_text else None

    bert_f1 = None
    try:
        bert_f1 = bert_score([predicted_text], [actual_text], lang="en")[2].item() if actual_text else None
    except Exception:
        pass

    per_score_val = None
    try:
        actual_phonemes = phonemize(actual_text, language="en-us")
        predicted_phonemes = phonemize(predicted_text, language="en-us")
        per_score_val = cer(actual_phonemes, predicted_phonemes) if actual_text else None
    except Exception:
        pass

    return {
        "Prompt_File": os.path.basename(p_file),
        "Audio_File": os.path.basename(a_file),
        "Actual": actual_text,
        "Predicted": predicted_text,
        "WER": wer_score_val,
        "CER": cer_score_val,
        "MER": mer_score_val,
        "WIL": wil_score_val,
        "BERTScore": bert_f1,
        "PER": per_score_val
    }

for session_path in tqdm(session_dirs, desc="🔍 Processing sessions"):
    prompts_dir = os.path.join(session_path, "prompts")
    wav_dir = os.path.join(session_path, "wav_arrayMic")
    if not os.path.isdir(prompts_dir) or not os.path.isdir(wav_dir):
        print(f"⚠️ Skipping {session_path}: Missing required folders.")
        continue

    prompt_files = glob.glob(os.path.join(prompts_dir, "*.txt"))
    audio_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    prompt_map = {os.path.splitext(os.path.basename(f))[0]: f for f in prompt_files}
    audio_map = {os.path.splitext(os.path.basename(f))[0]: f for f in audio_files}
    matched_pairs = [(prompt_map[k], audio_map[k]) for k in prompt_map if k in audio_map]

    with ThreadPoolExecutor() as executor:
        session_results = list(executor.map(process_file_pair, matched_pairs))
    results.extend(filter(None, session_results))

print(f"✅ Processing complete. Total processed files: {len(results)}")

# ------------------- SAVE RESULTS TO CSV -------------------
if results:
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(CSV_OUTPUT) or ".", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = CSV_OUTPUT.replace(".csv", f"_{timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"✅ Transcription results saved to: {csv_filename}")

# ------------------- GENERATE SUMMARY REPORT -------------------
if results:
    summary = {
        "Total Samples": len(df),
        "Mean WER": df["WER"].mean(skipna=True),
        "Mean CER": df["CER"].mean(skipna=True),
        "Mean MER": df["MER"].mean(skipna=True),
        "Mean WIL": df["WIL"].mean(skipna=True),
        "Mean BERTScore": df["BERTScore"].mean(skipna=True),
        "Mean PER": df["PER"].mean(skipna=True),
    }
    summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"]).fillna(0)
    summary_filename = SUMMARY_OUTPUT.replace(".csv", f"_{timestamp}.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"📊 ASR Performance Summary saved as: {summary_filename}")
