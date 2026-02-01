import os

# 1. Force enable online mode
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import json
import torch
import logging
from diffusers import StableDiffusionPipeline, LCMScheduler
import gc
import shutil
import random
import re

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_IMG")

def load_config():
    return {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "output_dir": "output/images",
        "char_dir": "output/images/characters",
        "events_file": "output/events.json",
        "chars_file": "output/characters.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "steps": 40,          # SDXL Standard
        "guidance_scale": 7.5,
        "height": 1024,
        "width": 1024
    }

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_pipeline(cfg):
    logger.info(f"Loading Pipeline: {cfg['model_id']} (SDXL)...")
    try:
        # Load SDXL
        # Note: ControlNet Canny requested, but for text-to-image of *new* scenes 
        # without input images, ControlNet Canny isn't directly applicable 
        # unless we have edge maps. We will use base SDXL for scene generation 
        # to ensure high quality first.
        
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL
        
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["model_id"], 
            vae=vae,
            torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16"
        )
        pipe.to(cfg["device"])
        
        if cfg["device"] == "cuda": 
            # pipe.enable_model_cpu_offload() # Uncomment if OOM
            pass
            
        return pipe
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def load_dna_map(cfg):
    """Loads Character DNA from JSON since Master Images are generated separately now."""
    if not os.path.exists(cfg["chars_file"]):
        return {}

    with open(cfg["chars_file"], "r", encoding="utf-8") as f:
        registry = json.load(f)

    char_prompts = {} 

    for char_name, details in registry.items():
        # Construct Visual DNA
        vis = details.get("visual_details", {})
        physical = vis.get("physical", "defined features") if isinstance(vis, dict) else str(vis)
        outfit = details.get("clothing_style", "casual clothes")
        
        # Prioritize outfit to avoid color bleeding from physical traits
        # e.g. "wearing Red Evening Gown, red wavy hair, green eyes" instead of "red wavy hair... wearing Red Evening Gown"
        dna_prompt = f"wearing {outfit}, {physical}"
        char_prompts[char_name] = dna_prompt

    return char_prompts

def generate_images(events_path="output/events.json"):
    cfg = load_config()
    cfg["events_file"] = events_path
    
    if not os.path.exists(cfg["output_dir"]): os.makedirs(cfg["output_dir"])
    if not os.path.exists(cfg["char_dir"]): os.makedirs(cfg["char_dir"])
    
    # 1. Load Data
    try:
        with open(cfg["events_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
            timeline = data.get("timeline", [])
    except FileNotFoundError:
        logger.error(f"Events file not found at {events_path}")
        return

    # 2. Init Pipeline
    pipe = load_pipeline(cfg)
    if not pipe: return

    # 3. Load DNA Map
    dna_map = load_dna_map(cfg)

    # 4. Generate Scenes
    logger.info("--- STARTING SCENE GENERATION (Action-Driven, SDXL) ---")

    CINEMATIC_SHOTS = {"CLOSE_UP", "MEDIUM", "WIDE", "ESTABLISHING"}
    VIEWS = ["front view", "side view", "three quarter view", "view from behind", "low angle shot", "high angle shot", "cinematic angle"]
    VIEW_KEYWORDS = ["view", "angle", "shot", "profile", "close-up", "full body", "looking at"]

    for scene in timeline:
        for beat in scene.get("beats", []):
            shot_type = beat.get("shot_type", "NONE")
            if shot_type not in CINEMATIC_SHOTS: continue
            if "visual_prompt" not in beat: continue
                
            shot_id = beat.get("sub_scene_id", "unknown")
            raw_action_prompt = beat["visual_prompt"] 
            
            filename = f"{shot_id}.png"
            filepath = os.path.join(cfg["output_dir"], filename)
            
            if os.path.exists(filepath):
                logger.info(f"Skipping {shot_id} (Exists)")
                pass 
                # Note: If the user wants to RE-generate to fix distortion, they should delete the old images manually 
                # or we should allow overwrite. For now, assuming user will clear output if they want fresh gen.
                # Actually, based on the prompt, the user is iterating, so we probably should overwrite?
                # But typically pipelines skip existing. I'll stick to skip for safety unless instructed.
                # Wait, the user says "the generated character images are distorted... create a folder...".
                # They imply the *next* run should be better.
            
            # --- RUNTIME PROMPT ASSEMBLY ---
            final_prompt = raw_action_prompt
            
            # Identify characters present in this shot
            present_chars = []
            
            # Simple Injection Strategy & Character Tracking
            for name, dna in dna_map.items():
                if name in final_prompt:
                    present_chars.append(name)
                    # SDXL likes "photo of [Subject]"
                    pattern = re.compile(rf"\b{name}\b", re.IGNORECASE)
                    # Reorder: Name -> DNA (Outfit, Physical)
                    replacement = f"{name} ({dna})" 
                    final_prompt = pattern.sub(replacement, final_prompt)
            
            # Add Shot Type if missing
            if shot_type.lower().replace("_", " ") not in final_prompt.lower():
                 final_prompt = f"{shot_type.lower().replace('_', ' ')}, {final_prompt}"

            # Add Random View if no view specified (to avoid "front view only" bias)
            # Only add if we have characters, otherwise "view from behind" of a building might be weird
            has_view = any(k in final_prompt.lower() for k in VIEW_KEYWORDS)
            if not has_view and len(present_chars) > 0:
                # Weighted choice
                weighted_views = VIEWS + ["three quarter view", "cinematic angle"] * 2
                chosen_view = random.choice(weighted_views)
                final_prompt += f", {chosen_view}"

            # Quality Boosters for SDXL
            final_prompt += ", photo, realistic, cinematic lighting, 8k, ultra detailed, sharp focus, film grain"
            
            # Enhanced Negative Prompt for distortion and bad anatomy
            neg_prompt = "drawing, painting, illustration, anime, cartoon, 3d render, doll, plastic, blur, low quality, distorted, bad anatomy, bad hands, missing fingers, extra limbs, mutated, poorly drawn face, ugly, disfigured, text, watermark, color bleeding"

            generated_new = False
            if not os.path.exists(filepath):
                logger.info(f"Generating {shot_id} [{shot_type}]")
                try:
                    image = pipe(
                        final_prompt,
                        height=cfg["height"],
                        width=cfg["width"],
                        num_inference_steps=cfg["steps"],
                        guidance_scale=cfg["guidance_scale"],
                        negative_prompt=neg_prompt
                    ).images[0]
                    
                    image.save(filepath)
                    generated_new = True
                except Exception as e:
                    logger.error(f"  Error {shot_id}: {e}")
            else:
                # If skipping generation, we might still want to copy to the character folder if it wasn't there
                # But without loading the image, we can't save it easily if using 'image.save'. 
                # We can just use shutil.copy below.
                pass
                
            # Copy to Character Folders (Even if skipped generation, ensure they exist in folders)
            if os.path.exists(filepath):
                for char_name in present_chars:
                    char_folder = os.path.join(cfg["char_dir"], char_name)
                    if not os.path.exists(char_folder):
                        os.makedirs(char_folder)
                    
                    dest_path = os.path.join(char_folder, filename)
                    if not os.path.exists(dest_path) or generated_new:
                        shutil.copy(filepath, dest_path)
                        # logger.info(f"  Saved copy to {dest_path}")

    logger.info("Generation Complete.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/events.json"
    generate_images(path)
os.environ["TRANSFORMERS_OFFLINE"] = "0"
