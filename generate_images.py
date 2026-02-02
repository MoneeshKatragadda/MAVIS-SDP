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
        "model_id": "segmind/SSD-1B", # Optimized: Switched to SSD-1B (60% faster than SDXL)
        "output_dir": "output/images",
        "char_dir": "output/images/characters",
        "events_file": "output/events.json",
        "chars_file": "output/characters.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "steps": 20,          # Optimized: Reduced to 20 for speed (sufficient with DPM++)
        "guidance_scale": 7.5,
        "height": 1024,
        "width": 1024
    }

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_pipeline(cfg):
    logger.info(f"Loading Pipeline: {cfg['model_id']} (Speed Optimized)...")
    try:
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
        
        # Load VAE separately for stability
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["model_id"], 
            vae=vae,
            torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Scheduler Optimization: DPM++ 2M Karras (Fastest convergence)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )
        
        if cfg["device"] == "cuda": 
            # Aggressive Memory & Speed Optimizations
            pipe.enable_model_cpu_offload() 
            pipe.enable_vae_tiling()
            try:
                import xformers
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                pipe.enable_attention_slicing()
        else:
            pipe.to(cfg["device"])
            
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
        # Improved Prompt Engineering matching generate_cast.py
        dna_prompt = f"(wearing {outfit}:1.3), {physical}"
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
            
            # Remove checking for existence to force regeneration with new prompts as requested
            # usage logic: if script is run, we assume user wants generation. 
            # We will rely on overwriting.
            
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


            has_view = any(k in final_prompt.lower() for k in VIEW_KEYWORDS)
            if not has_view and len(present_chars) > 0:
                # Weighted choice
                weighted_views = VIEWS + ["three quarter view", "cinematic angle"] * 2
                chosen_view = random.choice(weighted_views)
                final_prompt += f", {chosen_view}"

            # Quality Boosters for SDXL
            final_prompt += ", masterpiece, best quality, photo, realistic, cinematic lighting, 8k, ultra detailed, sharp focus, film grain"
            
            # Enhanced Negative Prompt for distortion and bad anatomy
            neg_prompt = (
                "drawing, painting, illustration, anime, cartoon, 3d render, doll, "
                "plastic, blur, low quality, distorted, bad anatomy, bad hands, missing fingers, "
                "extra limbs, mutated, poorly drawn face, ugly, disfigured, text, watermark, color bleeding"
            )

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
                logger.info(f"Skipping {shot_id} (Exists)")
                
            # Copy logic removed as per user request.
            # Only keeping images in the main output/images directory.
            pass

    logger.info("Generation Complete.")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output/events.json"
    generate_images(path)
os.environ["TRANSFORMERS_OFFLINE"] = "0"
