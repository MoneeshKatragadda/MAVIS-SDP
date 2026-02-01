import json
import os
import torch
import gc
import logging
import soundfile as sf
import shutil
import numpy as np
from tqdm import tqdm

# --- FORCE ONLINE MODE ---
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# --- PYTORCH FIXES ---
_original_load = torch.load
def strict_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = strict_load

# --- CONFIGURATION ---
FORCE_CPU_CASTING = False 
CLEAN_SLATE = False
SKIP_EXISTING = True # Speed up iterations if files exist

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

# --- PHASE 3: SCORING (MusicGen) ---
class MusicComposer:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu" if FORCE_CPU_CASTING else ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        logger.info(f"  > [Phase 3] Loading MusicGen on {self.device}...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)

    def compose_bgm(self, style, duration, output_path):
        if not style or style.lower() == "none" or style.lower() == "silence":
            return
            
        # Refined Prompts for Noir/Ambience
        # MusicGen is better at "Music" so we mix musical terms with atmospheric ones.
        prompt_map = {
            "Rainy Noir Ambience": "Rain drops hitting glass window, loud, distinct water sounds and distant sounds of thunder.",
            "Dark Suspense Drone": "Dark ambient drone, low synth textures, ominous, suspenseful, cinematic thriller soundtrack.",
            "Tense Industrial Pulse": "Industrial rhythmic pulse, metallic texture, tense, anxious, slow tempo.",
            "Suspenseful Drone": "Deep bass drone, suspenseful, minimal, scary movie atmosphere.",
            "Low Hum / City Ambience": "Low frequency hum, ambient, night time.",
            "Melancholic Saxophone & Rain": "Slow sad saxophone solo with rain in background, emotional, noir jazz.",
            "Aggressive Bass Drone": "Distorted heavy bass drone, aggressive, dangerous, horror texture.",
            "Smooth Jazz": "Smooth slow jazz, double bass, brushed drums, piano, relaxing, noir bar.",
            "Mystery Piano": "Minimal mysterious piano melody, reverb, enigmatic, investigating."
        }

        # Fallback if style isn't in map
        base_prompt = prompt_map.get(style, f"A {style} background music track, noir atmosphere, cinematic, high quality.")
        
        inputs = self.processor(
            text=[base_prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Calculate Duration
        duration = min(duration, 30.0) 
        max_tokens = int(duration * 50) + 10

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=True, 
                guidance_scale=3.0,
                temperature=1.0
            )

        # Save
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        sf.write(output_path, audio_data, sampling_rate)
        logger.info(f"    > Composed '{style}' ({duration}s) -> {output_path}")

    def unload(self):
        del self.model
        del self.processor
        flush_memory()

# --- PHASE 4: FOLEY (AudioGen) ---
class FoleyArtist:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu" if FORCE_CPU_CASTING else ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        logger.info(f"  > [Phase 4] Loading SFX Model on {self.device}...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        # Fallback to MusicGen-Small as AudioGen had loading issues. 
        # We will prompt it carefully to produce sound attributes.
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)

    def generate_sfx(self, description, duration, output_path):
        if not description or description.lower() == "none":
            return

        # Prompt engineering for short, precise SFX
        # We explicitly ask for "Isolated" and "Short" to prevent musical drift
        prompt = f"Isolated foley sound of {description}, short, distinct, high fidelity, realistic, no music."
        
        # Force short duration for precision
        duration = min(duration, 2.0) 
        
        # 50 tokens ~ 1 second roughly. 
        max_tokens = int(duration * 50) + 5

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=True, 
                guidance_scale=3.5, # Increased guidance to follow prompt strictly
                temperature=1.0
            )

        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        sf.write(output_path, audio_data, sampling_rate)
        logger.info(f"    > Created SFX '{description}' ({duration}s) -> {output_path}")

    def unload(self):
        del self.model
        del self.processor
        flush_memory()

def generate_audio(events_path="output/events.json"):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    if CLEAN_SLATE and os.path.exists(VOICES_DIR):
        shutil.rmtree(VOICES_DIR)
    if not os.path.exists(VOICES_DIR): os.makedirs(VOICES_DIR)
    
    if not os.path.exists(events_path): 
        logger.error(f"Events file not found at {events_path}")
        return

    with open(events_path, "r") as f: data = json.load(f)
    
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
            
            if SKIP_EXISTING and os.path.exists(out_path):
                continue
                
            # Always overwrite to ensure overrides are applied (if not skipping)
            if os.path.exists(out_path): os.remove(out_path)

            producer.generate_line(text, speaker, emotion, out_path)
    
    # Unload Producer to free VRAM for Scoring
    del producer
    flush_memory()
    
    # Phase 3: Scoring (BGM)
    # We generate unique BGM tracks per unique style found in narration beats, or per scene?
    # Logic: Look at beats, if they have BGM, generate it.
    # To save time, we deduplicate by Style.
    
    composer = MusicComposer()
    composer.load()
    logger.info("--- SCORING SESSION ---")
    
    # Collect unique BGM requirements
    bgm_registry = {} # style -> duration needed (max)
    
    for scene in data.get("timeline", []):
        for beat in scene.get("beats", []):
            prod = beat.get("production", {})
            bgm = prod.get("bgm", {})
            style = bgm.get("style", "None")
            if style and style.lower() not in ["none", "silence"]:
                # Use a sanitized filename key
                key = style.replace(" ", "_").replace("/", "-").lower()
                dur = beat.get("duration", 10.0)
                if key not in bgm_registry:
                    bgm_registry[key] = {"style": style, "duration": dur}
                else:
                    # Update max duration
                    bgm_registry[key]["duration"] = max(bgm_registry[key]["duration"], dur)
    
    bgm_dir = os.path.join(OUTPUT_DIR, "bgm")
    if not os.path.exists(bgm_dir): os.makedirs(bgm_dir)
    
    for key, info in tqdm(bgm_registry.items(), desc="Composing"):
        out_path = os.path.join(bgm_dir, f"{key}.wav")
        # Ensure we don't skip if updated? Assume SKIP_EXISTING applies globally
        if SKIP_EXISTING and os.path.exists(out_path): continue
        if os.path.exists(out_path): os.remove(out_path)

        composer.compose_bgm(info["style"], info["duration"], out_path)
            
    composer.unload()

    # Phase 4: Foley (SFX)
    foley = FoleyArtist()
    foley.load()
    logger.info("--- FOLEY SESSION ---")
    
    sfx_dir = os.path.join(OUTPUT_DIR, "sfx")
    if not os.path.exists(sfx_dir): os.makedirs(sfx_dir)
    
    # Collect deduplicated SFX
    sfx_registry = {} # name -> max_duration
    
    for scene in data.get("timeline", []):
        for beat in scene.get("beats", []):
            prod = beat.get("production", {})
            sfx_list = prod.get("sfx", [])
            for s in sfx_list:
                name = s.get("name", "None")
                if name and name.lower() != "none":
                    key = name.replace(" ", "_").replace("/", "-").lower()
                    
                    # Estimate Duration: Short and precise
                    dur = 2.0 
                    if key not in sfx_registry:
                        sfx_registry[key] = {"name": name, "duration": dur}

    for key, info in tqdm(sfx_registry.items(), desc="Foley"):
        out_path = os.path.join(sfx_dir, f"{key}.wav")
        if SKIP_EXISTING and os.path.exists(out_path):
            continue
        if os.path.exists(out_path): os.remove(out_path)
        
        foley.generate_sfx(info["name"], info["duration"], out_path)
    
    foley.unload()

    logger.info("Production Wrap!")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/events.json"
    generate_audio(path)
    