import os
import sys
import logging
import json
import torch
# Force online mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "0"
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_CAST_SDXL")

def load_config():
    return {
        "model_id": "segmind/SSD-1B", # SDXL Mini (Distilled, 50% smaller)
        "char_dir": "output/images/characters",
        "chars_file": "output/characters.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "steps": 20,          
        "guidance_scale": 7.0, 
        "height": 768,        # Reduced to 768x768 for Speed + Stability (SSD-1B works well here)
        "width": 768
    }

def generate_cast():
    cfg = load_config()
    
    if not os.path.exists(cfg["chars_file"]):
        logger.error(f"Characters file not found at {cfg['chars_file']}")
        return

    # Load SDXL Mini (SSD-1B)
    logger.info(f"Loading SDXL Mini (SSD-1B): {cfg['model_id']}...")
    try:
        # FIX: Load VAE in FP32 separately to avoid black/fried images & type mismatch errors
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16 # Use the fixed FP16 VAE which is faster & doesn't break
        )
        
        # Standard load
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["model_id"], 
            vae=vae, # Inject fixed VAE
            torch_dtype=torch.float16 if cfg["device"] == "cuda" else torch.float32,
            variant="fp16",
            use_safetensors=True
        )
        
        if cfg["device"] == "cuda":
            logger.info("  Enabling Memory Optimizations (CPU Offload, Tiling)...")
            # Aggressive Memory Optimization
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
            
            # Try xformers for speed boost
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("  Enabled xformers memory efficient attention.")
            except Exception:
                logger.warning("  xformers not installed/available. Using attention slicing.")
                pipe.enable_attention_slicing()
        else:
            logger.warning("  Running on CPU. Expect slow generation.")
            pipe.to(cfg["device"])

        # Optimize: Switch to DPMSolverMultistepScheduler (Faster convergence)
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, 
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++" 
        )

    except Exception as e:
        logger.error(f"Failed to load specific optimizations or model: {e}")
        return

    # Load Registry
    with open(cfg["chars_file"], "r", encoding="utf-8") as f:
        registry = json.load(f)

    if not os.path.exists(cfg["char_dir"]): os.makedirs(cfg["char_dir"])

    logger.info("--- GENERATING 4-VIEW REFERENCE SHEETS (SDXL) ---")


    
    for char_name, details in registry.items():
        logger.info(f"> Processing Character: {char_name}")
        
        # DNA Construction
        vis = details.get("visual_details", {})
        physical = vis.get("physical", "defined features") if isinstance(vis, dict) else str(vis)
        outfit = details.get("clothing_style", "casual clothes")
        
        # Create character specific folder
        char_subdir = os.path.join(cfg["char_dir"], char_name.replace(" ", "_"))
        if not os.path.exists(char_subdir):
            os.makedirs(char_subdir)
            
        # Reordered prompt: Outfit BEFORE Physical to anchor clothing color and reduce bleeding from other features (like "green eyes")
        base_prompt = (
            f"photo, realistic, wearing {outfit}, {physical}, "
            f"neutral background, studio lighting, 8k, ultra detailed, sharp focus, real skin texture"
        )
        
        neg_prompt = (
            "drawing, painting, illustration, anime, cartoon, 3d render, doll, "
            "plastic, blur, low quality, distorted, bad anatomy, extra limbs, text, watermark"
        )

        # Updated strict prompt instructions (Framing without style bias)
        views = {
            "waist_front": "Upper body medium shot, from waist up only, detailed torso and face, looking at camera",
            "waist_back": "Upper body medium shot, from waist up only, back view, detailed back",
            "waist_side": "Upper body medium shot, side profile view, from waist up only",
            "full_front": "extreme wide full body shot, standing pose, displaying entire body from head to shoes, feet visible, front view, far away"
        }

        # Base seed for the character
        seed = abs(hash(char_name)) % (2**32)
        logger.info(f"  Using base seed: {seed}")
        
        # Base Negative Prompt
        base_neg = neg_prompt + ", close up, extremes close up, macro, headshot, face shot, portrait, cropped, zoom, shoulders only, neck only, hat, cowboy hat, headwear"

        for view_key, view_prompt_prefix in views.items():
            view_filename = f"{char_name.lower()}_{view_key}.png"
            filepath = os.path.join(char_subdir, view_filename)
            
            # Dynamic Negative Prompting
            if "waist" in view_key:
                # Force cut at waist by forbidding lower body features
                current_neg_prompt = base_neg + ", full body, legs, feet, shoes, boots, knees, wide shot, long shot, far away"
            else:
                # Full body needs to allow feet, so we don't ban them
                current_neg_prompt = base_neg

            # Reset generator seed for EACH view to ensure identical starting noise (better consistency)
            generator = torch.Generator(device=cfg["device"]).manual_seed(seed)
            
            # Construct the full prompt - framing FIRST
            # Prompt Structure: [View], [Character], [Outfit], [Physical], [Style/Lighting]
            full_prompt = f"{view_prompt_prefix}, {char_name}, wearing {outfit}, {physical}, neutral background, studio lighting, 8k"
            
            logger.info(f"  Generating {view_key}...")
            
            try:
                image = pipe(
                    prompt=full_prompt,
                    negative_prompt=current_neg_prompt, # Use improved neg prompt
                    height=cfg["height"],
                    width=cfg["width"],
                    num_inference_steps=cfg["steps"],
                    guidance_scale=cfg["guidance_scale"],
                    generator=generator
                ).images[0]
                
                image.save(filepath)
            except Exception as e:
                logger.error(f"  Failed {view_key}: {e}")
                
    logger.info("Cast Generation Complete.")

if __name__ == "__main__":
    generate_cast()
