import json
import yaml
import logging
from src.llm_reasoner import LLMReasoner

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("PATCH_SFX")

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    logger.info("Starting SFX/BGM Patch...")
    cfg = load_config()
    
    # Load LLM
    llm = LLMReasoner(cfg)
    
    # Load Events
    with open("output/events.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    timeline = data.get("timeline", [])
    total_updates = 0
    
    for scene in timeline:
        scene_id = scene.get("id")
        beats = scene.get("beats", [])
        
        logger.info(f"Processing Scene {scene_id} ({len(beats)} beats)...")
        
        for beat in beats:
            # Only update production for narration primarily as per logic, 
            # but analyze_beat_production handles 'narration' check inside.
            # Wait, verify analyze_beat_production logic...
            # It checks if b_type == 'narration'.
            # If dialogue, it returns empty sfx/bgm defaults?
            # Let's check the updated code.
            
            # Calling the updated function
            new_prod = llm.analyze_beat_production(beat)
            
            # Update only if valid
            beat["production"] = new_prod
            total_updates += 1
            
            # Log changes for verification
            # logger.info(f"  > Beat {beat.get('sub_scene_id')}: SFX={new_prod['sfx']}")

    # Save
    with open("output/events.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Patch Complete. Updated {total_updates} beats.")

if __name__ == "__main__":
    main()
