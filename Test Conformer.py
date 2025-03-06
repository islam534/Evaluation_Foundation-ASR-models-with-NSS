import os
import glob
import logging
import librosa
import numpy as np
import torch
import pandas as pd
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC
from tqdm import tqdm
from jiwer import wer, cer, mer, wil, wip

# ---------------------- CONFIGURATION ----------------------
# Using a Conformer-based ASR model (facebook/wav2vec2-conformer-rope-large-960h-ft).
# Note that while Conformer uses convolution and self-attention to capture local and global context,
# additional techniques (e.g. dynamic chunk training) may be needed for streaming very long utterances.
MODEL_NAME = "facebook/wav2vec2-conformer-rope-large-960h-ft"
DATASET_PATH = os.path.normpath("C:/Users/A-plus store/OneDrive - MOE Stem Schools/Desktop/Testing And Evaluating Models/Dataset")
CSV_OUTPUT = "Conformer_Evaluation_results.csv"
BATCH_SIZE = 4
MIN_SAMPLES = 400  # At 16kHz, ~25ms of audio
USE_AMP = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------- LOAD MODEL ----------------------
def load_model_and_processor():
    logging.info("🔄 Loading Conformer model and processor...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ConformerForCTC.from_pretrained(MODEL_NAME).to(device).eval()
    logging.info("✅ Conformer model and processor loaded successfully.")
    return model, processor

model, processor = load_model_and_processor()

# ---------------------- AUDIO PREPROCESSING ----------------------
def preprocess_audio(audio_path):
    """Loads, trims, and normalizes an audio file for Conformer ASR.
    
    Note: Conformer models (like other transformer-based ASR) expect 16kHz mono audio.
    For very long utterances, additional streaming techniques might be necessary.
    """
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
    """Transcribes a batch of audio files using the Conformer model."""
    # Preprocess audio files; note that calling preprocess_audio twice per file can be optimized
    waveforms = [preprocess_audio(p) for p in audio_paths if preprocess_audio(p) is not None]
    if not waveforms:
        return {p: "" for p in audio_paths}
    
    max_len = max(map(len, waveforms))
    padded_waveforms = [np.pad(w, (0, max_len - len(w))) for w in waveforms]
    
    # Prepare inputs for the model
    inputs = processor(padded_waveforms, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
    
    # Perform inference using automatic mixed precision (without the device_type arg)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_AMP):
        logits = model(inputs).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return dict(zip(audio_paths, transcriptions))

# ---------------------- TEXT NORMALIZATION ----------------------
def normalize_text(text):
    """Normalizes text by converting to uppercase and removing special characters."""
    return re.sub(r'[^A-Z0-9 ]', '', text.upper().strip())

# ---------------------- SESSION EVALUATION ----------------------
def evaluate_session(session_path):
    """Processes a session directory and evaluates ASR performance using Conformer."""
    results = []
    prompts_dir = os.path.join(session_path, "prompts")
    wav_dir = os.path.join(session_path, "wav_arrayMic")

    if not os.path.isdir(prompts_dir) or not os.path.isdir(wav_dir):
        logging.warning(f"⚠️ Skipping {session_path} due to missing folders.")
        return results

    common_files = sorted(set(f[:-4] for f in os.listdir(wav_dir)) & set(f[:-4] for f in os.listdir(prompts_dir)))
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
                logging.warning(f"⚠️ No transcription for {af}.")
                continue
            
            # Compute error metrics for ASR evaluation
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
