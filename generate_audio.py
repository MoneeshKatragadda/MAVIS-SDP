import json
import os
import torch
import gc
import logging
import soundfile as sf
import shutil
import numpy as np
from tqdm import tqdm

# --- PYTORCH FIXES ---
_original_load = torch.load
def strict_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = strict_load

# --- CONFIGURATION ---
FORCE_CPU_CASTING = False 
CLEAN_SLATE = False  # Set to False now to save time (we don't need to re-cast everyone)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_DIRECTOR")

# Constants
OUTPUT_DIR = "output/audio"
VOICES_DIR = "output/voices"
EVENTS_FILE = "output/events.json"

# --- DIRECTOR'S OVERRIDES ---
# Fix specific lines where the JSON emotion doesn't match the story.
# Format: "SCENE_ID": "emotion_label"
MANUAL_OVERRIDES = {
    "SC_008_01": "whisper",  # "I can't breathe!" -> Force Whisper
    "SC_008_03": "whisper",  # "Do you have any idea..." -> Force Whisper
    "SC_006_03": "whisper"   # "I don't like the way the cook..." -> Force Whisper (optional)
}

# --- SEEDS ---
CHARACTER_SEEDS = {
    "Narrator": 42,
    "Julian": 1234,
    "Silas": 2024,
    "Lena": 303
}

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
        self.device = "cpu" if FORCE_CPU_CASTING else ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        logger.info(f"  > [Phase 1] Loading Parler TTS on {self.device}...")
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    def generate_master_reference(self, character_name, archetype, gender):
        filename = f"{character_name}_master.wav"
        filepath = os.path.join(VOICES_DIR, filename)
        
        if os.path.exists(filepath): return 

        if "narrator" in character_name.lower():
            desc = "A male speaker with a very deep, slow, resonant, cinematic narration voice."
        elif "julian" in character_name.lower():
            desc = "A male speaker with a clear, standard American voice."
        elif "silas" in character_name.lower():
            desc = "A male speaker with a rough, low-pitched, gravelly voice."
        elif "lena" in character_name.lower():
            desc = "A female speaker with a clear, high-pitched voice."
        else:
            gender_term = "female" if gender.lower() == "female" else "male"
            desc = f"A {gender_term} speaker with a clear, high quality voice."

        full_prompt = f"{desc} He is speaking normally and clearly. High quality audio."
        if "female" in desc.lower():
            full_prompt = f"{desc} She is speaking normally and clearly. High quality audio."

        ref_text = f"My name is {character_name}. I am speaking clearly to establish my voice."

        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(ref_text, return_tensors="pt").input_ids.to(self.device)
        
        seed = CHARACTER_SEEDS.get(character_name, 555)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        with torch.no_grad():
            generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, do_sample=True, temperature=1.0)
        
        sf.write(filepath, generation.cpu().numpy().squeeze(), self.model.config.sampling_rate)
        logger.info(f"    > Cast {character_name} [Master] (Seed: {seed})")

    def unload(self):
        del self.model
        del self.tokenizer
        flush_memory()

# --- PHASE 2: PRODUCTION (XTTS) ---
class AudioProducer:
    def __init__(self):
        self.model = None
    
    def load(self):
        logger.info("  > [Phase 2] Loading XTTS...")
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def generate_line(self, text, speaker_name, emotion_label, output_path):
        ref_path = os.path.join(VOICES_DIR, f"{speaker_name}_master.wav")
        if not os.path.exists(ref_path): ref_path = os.path.join(VOICES_DIR, "Narrator_master.wav")

        speed = 1.0
        text_processed = text.strip()
        emotion = emotion_label.lower() if emotion_label else "neutral"

        # --- EMOTION ACTING LOGIC ---
        if emotion in ["anger", "fury", "annoyance", "disapproval"]:
            speed = 1.1 
            text_processed = text_processed.upper() 
        
        elif emotion in ["fear", "nervous", "dread", "desperate", "suspense"]:
            speed = 0.95
            text_processed = text_processed.replace(", ", "... ").replace(". ", "... ")
        
        elif emotion in ["sadness", "grief"]:
            speed = 0.85
            text_processed = text_processed.lower()
        
        elif emotion in ["whisper", "ominous"]:
            # Whisper Logic: Slow, lowercase, remove exclamation marks (they trigger loudness)
            speed = 0.9
            text_processed = text_processed.lower().replace("!", ".") 

        try:
            self.model.tts_to_file(
                text=text_processed,
                file_path=output_path,
                speaker_wav=ref_path,
                language="en",
                speed=speed,
                split_sentences=True
            )
        except Exception as e:
            logger.error(f"XTTS Error {output_path}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    if CLEAN_SLATE and os.path.exists(VOICES_DIR):
        shutil.rmtree(VOICES_DIR)
    if not os.path.exists(VOICES_DIR): os.makedirs(VOICES_DIR)
    
    if not os.path.exists(EVENTS_FILE): return
    with open(EVENTS_FILE, "r") as f: data = json.load(f)
    
    registry = data.get("character_registry", {})
    if "Narrator" not in registry:
        registry["Narrator"] = {"archetype": "Deep Narrative Voice", "gender": "Male"}

    # Phase 1: Casting
    caster = CastingDirector()
    caster.load()
    logger.info("--- CASTING CALL ---")
    for name, details in registry.items():
        caster.generate_master_reference(name, details.get("archetype", "Generic"), details.get("gender", "Male"))
    caster.unload() 
    
    # Phase 2: Production
    producer = AudioProducer()
    producer.load()
    logger.info("--- RECORDING SESSION ---")
    
    for scene in tqdm(data.get("timeline", []), desc="Recording"):
        for beat in scene.get("beats", []):
            if beat["type"] not in ["dialogue", "narration"]: continue
            
            beat_id = beat.get("sub_scene_id")
            text = beat.get("text")
            emotion = beat.get("emotion", {}).get("label", "neutral")
            speaker = beat.get("speaker", "Narrator")
            if beat["type"] == "narration": speaker = "Narrator"
            
            # --- APPLY OVERRIDES ---
            # If the ID exists in MANUAL_OVERRIDES, use that emotion instead
            if beat_id in MANUAL_OVERRIDES:
                logger.info(f"  > Overriding {beat_id}: {emotion} -> {MANUAL_OVERRIDES[beat_id]}")
                emotion = MANUAL_OVERRIDES[beat_id]

            out_path = os.path.join(OUTPUT_DIR, f"{beat_id}.wav")
            # Always overwrite to ensure overrides are applied
            if os.path.exists(out_path): os.remove(out_path)

            producer.generate_line(text, speaker, emotion, out_path)

    logger.info("Production Wrap!")

if __name__ == "__main__":
    main()
    