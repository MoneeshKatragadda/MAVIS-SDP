import os
import json
import logging
import numpy as np
import soundfile as sf
import librosa
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
EVENTS_FILE = "output/events.json"
AUDIO_DIR = "output/audio"
VOICES_DIR = "output/voices"
LOG_FILE = "output/audio_evaluation.log"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filename=LOG_FILE,
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def load_audio(path):
    """Load audio with Librosa (default 22050Hz, mono)."""
    try:
        y, sr = librosa.load(path, sr=22050)
        return y, sr
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        return None, None

def extract_features(y, sr):
    """Extract MFCCs and RMS."""
    if y is None or len(y) == 0:
        return None, None
    
    # MFCCs (Timbre)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # RMS (Volume / Energy)
    rms = librosa.feature.rms(y=y)
    
    return mfcc, rms

def calculate_mcd_dtw(mfcc1, mfcc2):
    """
    Calculate Dynamic Time Warping distance between MFCCs (Proxy for MCD).
    Lower is better (closer timbre).
    NOTE: Real MCD uses Cepstral coefficients from SPTK, but MFCC+DTW is a standard Python proxy.
    """
    if mfcc1 is None or mfcc2 is None: return float('inf')
    
    # Transpose for fastdtw (requires [n_samples, n_features])
    d1 = mfcc1.T
    d2 = mfcc2.T
    
    # Calculate DTW distance
    distance, path = fastdtw(d1, d2, dist=euclidean)
    
    # Normalize by path length
    normalized_distance = distance / len(path)
    return normalized_distance

def evaluate_audio():
    logging.info("--- STARTING AUDIO EVALUATION ---")
    
    if not os.path.exists(EVENTS_FILE):
        logging.error("Events file not found.")
        return

    with open(EVENTS_FILE, "r") as f:
        data = json.load(f)

    # 1. Load Master References
    master_voices = {}
    master_features = {}
    
    registry = data.get("character_registry", {})
    # Add Narrator
    registry["Narrator"] = {}
    
    logging.info("Loading Master References...")
    for name in registry.keys():
        path = os.path.join(VOICES_DIR, f"{name}_master.wav")
        if os.path.exists(path):
            y, sr = load_audio(path)
            if y is not None:
                master_voices[name] = y
                mfcc, _ = extract_features(y, sr)
                master_features[name] = mfcc
                logging.info(f"  > Loaded {name} Reference")
        else:
            logging.warning(f"  ! Missing Reference for {name}")

    results = []

    # 2. Iterate Timelines
    for scene in data.get("timeline", []):
        for beat in scene.get("beats", []):
            if beat["type"] not in ["dialogue", "narration"]:
                continue
                
            beat_id = beat.get("sub_scene_id")
            speaker = beat.get("speaker", "Narrator")
            if beat["type"] == "narration": speaker = "Narrator"
            text_len = len(beat.get("text", ""))
            
            audio_path = os.path.join(AUDIO_DIR, f"{beat_id}.wav")
            
            if not os.path.exists(audio_path):
                continue
                
            # Analysis
            y, sr = load_audio(audio_path)
            mfcc, rms = extract_features(y, sr)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Metrics
            
            # A. Speaker Similarity (MCD-DTW)
            sim_score = "N/A"
            if speaker in master_features:
                ref_mfcc = master_features[speaker]
                dist = calculate_mcd_dtw(mfcc, ref_mfcc)
                # Lower distance = Higher Similarity. 
                # Let's verify valid range. Usually 20-50 for different, 5-10 for same.
                sim_score = round(dist, 2)
                
            # B. Speaking Rate (char / sec)
            wpm = (text_len / 5) / (duration / 60) if duration > 0 else 0
            
            # C. Energy ( Loudness avg)
            avg_rms = np.mean(rms) if rms is not None else 0
            
            result = {
                "id": beat_id,
                "speaker": speaker,
                "duration": round(duration, 2),
                "mcd_dtw": sim_score,
                "wpm": round(wpm, 0),
                "energy": round(avg_rms, 3)
            }
            results.append(result)

    # 3. Aggregation & Report
    logging.info("\n--- EVALUATION REPORT ---")
    print(f"{'ID':<12} | {'Speaker':<10} | {'Dur(s)':<6} | {'MCD(DTW)':<8} | {'WPM':<5} | {'Energy'}")
    print("-" * 65)
    
    speakers_mcd = {}
    
    for r in results:
        print(f"{r['id']:<12} | {r['speaker']:<10} | {r['duration']:<6} | {r['mcd_dtw']:<8} | {r['wpm']:<5} | {r['energy']}")
        
        # Aggregate MCD
        if r['mcd_dtw'] != "N/A":
            if r['speaker'] not in speakers_mcd: speakers_mcd[r['speaker']] = []
            speakers_mcd[r['speaker']].append(r['mcd_dtw'])

    print("-" * 65)
    logging.info("--- AGGREGATE IDENTITY CONSISTENCY (Lower is Better) ---")
    for spk, scores in speakers_mcd.items():
        avg_mcd = sum(scores) / len(scores)
        logging.info(f"  {spk}: Avg MCD-DTW = {avg_mcd:.2f}")

    logging.info("Evaluation Complete.")

if __name__ == "__main__":
    evaluate_audio()
