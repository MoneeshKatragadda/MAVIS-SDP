import os

# 1. Force enable online mode
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import json
import torch
import logging
from diffusers import StableDiffusionPipeline
import gc

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_IMG")

def load_config():
    return {
        "model_id": "runwayml/stable-diffusion-v1-5", 
        "output_dir": "output/images",
        "events_file": "output/events.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "steps": 35, # Increased for quality
        "guidance_scale": 8.0, # Increased for prompt adherence
        "height": 512,
        "width": 768
    }

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_images():
    cfg = load_config()
    
    if not os.path.exists(cfg["output_dir"]): os.makedirs(cfg["output_dir"])
    
    try:
        with open(cfg["events_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
            timeline = data.get("timeline", [])
    except FileNotFoundError:
        logger.error("Events file not found.")
        return

    logger.info(f"Loading Model: {cfg['model_id']}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model_id"], 
            torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32
        )
        pipe = pipe.to(cfg["device"])
        if cfg["device"] == "cuda": pipe.enable_attention_slicing()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("--- STARTING PRODUCTION (Text-Only High Fidelity) ---")

    for scene in timeline:
        for beat in scene.get("beats", []):
            if "visual_prompt" not in beat: continue
                
            shot_id = beat.get("sub_scene_id", "unknown")
            raw_prompt = beat["visual_prompt"]
            
            # File Check
            filename = f"{shot_id}.png"
            filepath = os.path.join(cfg["output_dir"], filename)
            
            if os.path.exists(filepath):
                logger.info(f"Skipping {shot_id} (Exists)")
                continue
            
            # --- PROMPT REFINEMENT FOR "WAIST UP" ---
            # We inject framing instructions to avoid 'headless' or 'extreme close up' issues.
            # "High quality color cinematic shot of..." is already in raw_prompt.
            # We add framing explicitly.
            
            # 1. Detect if it's a character shot
            character_shot = any(x in raw_prompt for x in ["Julian", "Silas", "Lena", "Man", "Woman"])
            
            final_prompt = raw_prompt
            if character_shot:
                # Force waist-up framing
                framing = "medium shot from waist up, perfectly framed face and body, "
                # Replace generic "cinematic shot" or prepend if missing
                if "cinematic shot of" in final_prompt:
                     final_prompt = final_prompt.replace("cinematic shot of", f"cinematic shot of {framing}")
                else:
                     final_prompt = f"{framing} {final_prompt}"
            
            # 2. Add quality boosters
            final_prompt += ", highly detailed, 8k, photorealistic, sharp focus"

            # 3. Enhanced Negative Prompt
            neg_prompt = (
                "black and white, monochrome, grayscale, sepia, "
                "extreme close up, head shot, facial close up, " # Anti-CloseUp
                "long shot, tiny character, " # Anti-Far
                "headless, cropped head, cut off head, buried face, "
                "bad anatomy, deformed, extra fingers, disfigured, "
                "cartoon, 3d, anime, text, watermark"
            )

            logger.info(f"Generating {shot_id}...")
            # logger.info(f"  Prompt: {final_prompt[:100]}...")
            
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
                logger.info(f"  Saved to {filepath}")
            except Exception as e:
                logger.error(f"  Error generating {shot_id}: {e}")

    logger.info("Generation Complete.")

if __name__ == "__main__":
    generate_images()
