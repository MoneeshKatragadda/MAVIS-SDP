import json
import os
import torch
import gc
import logging
import soundfile as sf
import numpy as np
from tqdm import tqdm

# --- CRITICAL FIX 1: PyTorch 2.6+ Compatibility ---
_original_load = torch.load
def strict_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = strict_load

# --- CRITICAL FIX 2: Prevent CUDA Hangs on Windows ---
# If your GPU hangs, set this to True to force CPU casting (slower but safer)
FORCE_CPU_CASTING = False 

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_EMOTIONAL_CAST")

# Constants
OUTPUT_DIR = "output/audio"
VOICES_DIR = "output/voices"
EVENTS_FILE = "output/events.json"

EMOTION_MAP = {
    "neutral": "neutral", "curiosity": "neutral", "ominous": "neutral",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger", "fury": "anger",
    "joy": "joy", "approval": "joy", "gratitude": "joy", "excitement": "joy",
    "fear": "fear", "nervous": "fear", "desperate": "fear", "dread": "fear", "suspense": "fear",
    "sadness": "sadness", "grief": "sadness", "melancholy": "sadness",
    "whisper": "whisper"
}

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- PHASE 1: CASTING (Parler TTS) ---
class CastingDirector:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # Use CPU if forced, otherwise check CUDA
        if FORCE_CPU_CASTING:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        logger.info(f"  > [Phase 1] Loading Parler TTS on {self.device}...")
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    def generate_emotional_set(self, character_name, archetype, gender):
        gender_term = "female" if gender.lower() == "female" else "male"
        core_emotions = ["neutral", "anger", "fear", "sadness", "joy", "whisper"]
        
        for emotion in core_emotions:
            filename = f"{character_name}_{emotion}.wav"
            filepath = os.path.join(VOICES_DIR, filename)
            
            if os.path.exists(filepath):
                continue

            # Prompt Construction
            prompt = f"A {gender_term} speaker "
            if emotion == "neutral":
                if "stoic" in archetype.lower(): prompt += "with a deep, calm, stoic voice speaking clearly."
                elif "narrator" in character_name.lower(): prompt += "with a very deep, resonant, cinematic narration voice."
                else: prompt += "speaking normally and clearly."
            elif emotion == "anger": prompt += "speaking loudly and aggressively with an angry tone."
            elif emotion == "fear": prompt += "speaking hesitantly with a shaky, terrified voice."
            elif emotion == "sadness": prompt += "speaking very slowly and quietly with a sad, depressed tone."
            elif emotion == "joy": prompt += "speaking efficiently with a happy, bright, and cheerful tone."
            elif emotion == "whisper": prompt += "whispering very quietly and intensely."

            full_prompt = f"{prompt} High quality audio."
            
            ref_text = (
                f"I am {character_name}. This is my voice when I am feeling {emotion}. "
                "I need to capture this specific tone perfectly."
            )
            
            # --- FIX: Explicit Attention Mask ---
            input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
            prompt_input_ids = self.tokenizer(ref_text, return_tensors="pt").input_ids.to(self.device)
            
            # Manually create attention mask to stop the warning/hang
            attention_mask = torch.ones_like(input_ids)
            prompt_attention_mask = torch.ones_like(prompt_input_ids)

            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    do_sample=True, 
                    temperature=1.0
                )
            
            audio_arr = generation.cpu().numpy().squeeze()
            sf.write(filepath, audio_arr, self.model.config.sampling_rate)
            logger.info(f"    > Cast {character_name} [{emotion}]")

    def unload(self):
        del self.model
        del self.tokenizer
        flush_memory()
        logger.info("  > [Phase 1] Casting Complete. Unloading Parler.")

# --- PHASE 2: PRODUCTION (XTTS) ---
class AudioProducer:
    def __init__(self):
        self.model = None
    
    def load(self):
        logger.info("  > [Phase 2] Loading XTTS for Production...")
        from TTS.api import TTS
        # XTTS is heavy; ensure we are on CUDA if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def generate_line(self, text, speaker_name, emotion_label, output_path):
        core_emotion = EMOTION_MAP.get(emotion_label, "neutral")
        
        # Reference Selection Logic
        ref_file = f"{speaker_name}_{core_emotion}.wav"
        ref_path = os.path.join(VOICES_DIR, ref_file)
        
        if not os.path.exists(ref_path):
            # Fallback chain: specific emotion -> neutral -> narrator
            ref_path = os.path.join(VOICES_DIR, f"{speaker_name}_neutral.wav")
            if not os.path.exists(ref_path):
                ref_path = os.path.join(VOICES_DIR, "Narrator_neutral.wav")

        text = text.strip()
        if core_emotion == "anger": text = text.upper()
        if core_emotion == "fear": text = text.replace(", ", "... ").replace(". ", "... ")

        try:
            self.model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=ref_path,
                language="en",
                split_sentences=True
            )
        except Exception as e:
            logger.error(f"XTTS Error on {output_path}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(VOICES_DIR): os.makedirs(VOICES_DIR)
    
    if not os.path.exists(EVENTS_FILE):
        logger.error("No events.json found.")
        return
    with open(EVENTS_FILE, "r") as f:
        data = json.load(f)
    
    registry = data.get("character_registry", {})
    if "Narrator" not in registry:
        registry["Narrator"] = {"archetype": "Deep Narrative Voice", "gender": "Male"}

    # Phase 1
    caster = CastingDirector()
    caster.load()
    logger.info("--- CASTING CALL ---")
    for name, details in registry.items():
        caster.generate_emotional_set(
            character_name=name,
            archetype=details.get("archetype", "Generic"),
            gender=details.get("gender", "Male")
        )
    caster.unload() 
    
    # Phase 2
    producer = AudioProducer()
    producer.load()
    logger.info("--- RECORDING SESSION ---")
    
    for scene in tqdm(data.get("timeline", []), desc="Recording"):
        for beat in scene.get("beats", []):
            if beat["type"] not in ["dialogue", "narration"]: continue
            
            beat_id = beat.get("sub_scene_id")
            text = beat.get("text")
            emotion = beat.get("emotion", {}).get("label", "neutral").lower()
            
            speaker = beat.get("speaker", "Narrator")
            if beat["type"] == "narration": speaker = "Narrator"
            
            out_path = os.path.join(OUTPUT_DIR, f"{beat_id}.wav")
            if os.path.exists(out_path): os.remove(out_path) 

            producer.generate_line(text, speaker, emotion, out_path)

    logger.info("Production Wrap!")

if __name__ == "__main__":
    main()