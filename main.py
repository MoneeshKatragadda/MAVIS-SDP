# -*- coding: utf-8 -*-
import yaml, json, re, logging
from collections import Counter
from src.extractor import NLPExtractor
from src.llm_reasoner import LLMReasoner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS_CORE")

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def aggregate_scene_emotion(beats):
    labels = [b["emotion"]["label"] for b in beats]
    if not labels: return {"dominant_emotion": "neutral", "intensity": 0.0}
    common = Counter(labels).most_common(1)[0]
    return {"dominant_emotion": common[0], "intensity": 0.8}

def main():
    logger.info("Initializing MAVIS Pipeline 2.2 (Refined Entities & Visuals)")
    cfg = load_config()
    
    with open(cfg["paths"]["input_file"], "r", encoding="utf-8") as f:
        raw = f.read()
        clean_text = re.sub(r'\\', '', raw).strip()
        # Clean smart quotes
        replacements = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
        for k, v in replacements.items():
            clean_text = clean_text.replace(k, v)
        
    scenes_text = [s.strip() for s in clean_text.split("\n\n") if s.strip()]
    
    nlp = NLPExtractor(cfg)
    nlp.load_emotion_model()
    llm = LLMReasoner(cfg)

    characters = nlp.extract_characters_from_text(clean_text)
    logger.info(f"Cast: {characters}")

    timeline = []
    global_cursor = 0.0
    global_assets = {"locations": set(), "props": set()}
    last_location = "Start"

    for i, s_text in enumerate(scenes_text, 1):
        logger.info(f"Processing Scene {i}")
        
        struct = nlp.parse_scene_structure(s_text, characters)
        entities = nlp.extract_scene_entities(s_text, characters)
        
        for e in entities:
            if e['type'] == 'location': global_assets['locations'].add(e['name'])
            if e['type'] == 'prop': global_assets['props'].add(e['name'])
            
        scene_duration = 0.0
        beats = []
        for beat in struct['beats']:
            beat['emotion'] = nlp.get_emotion(beat['text'], beat['type'])
            beat['semantic'] = nlp.extract_svo(beat['text'])
            
            beat_dur = beat['duration']
            beat['timing'] = {
                "rel_start": round(scene_duration, 2),
                "rel_end": round(scene_duration + beat_dur, 2),
                "global_start": round(global_cursor + scene_duration, 2),
                "global_end": round(global_cursor + scene_duration + beat_dur, 2)
            }
            beat['audio_prompt'] = nlp.build_audio_prompt(beat, beat['emotion'])
            beats.append(beat)
            scene_duration += beat_dur

        scene_emo = aggregate_scene_emotion(beats)
        prod_meta = llm.analyze_scene_production(s_text, last_location, scene_emo)
        sfx_triggers = nlp.extract_sfx(s_text)

        event = {
            "id": f"SC_{i:03d}",
            "meta": {
                "global_start": round(global_cursor, 2),
                "global_end": round(global_cursor + scene_duration, 2),
                "duration": round(scene_duration, 2),
                "transition_to_next": prod_meta['transition']
            },
            "script": {
                "text": s_text,
                "active_cast": struct['active_chars']
            },
            "production": {
                "bgm": prod_meta['bgm'],
                "camera": prod_meta['camera'],
                "sfx_cues": sfx_triggers,
                "lighting": f"{scene_emo['dominant_emotion']} lighting"
            },
            "entities": entities,
            "visuals": {
                # New Field
                "visual_prompt": prod_meta['visual_prompt']
            },
            "beats": beats
        }
        
        timeline.append(event)
        global_cursor += scene_duration
        
        locs = [e['name'] for e in entities if e['type'] == 'location']
        if locs: last_location = locs[0]

    logger.info("Generating Global Registry...")
    registry = llm.generate_rich_registry(characters, clean_text)
    
    final_output = {
        "project_meta": {
            "title": "MAVIS_Project", 
            "fps": 24, 
            "total_duration": round(global_cursor, 2)
        },
        "global_assets": {
            "locations": sorted(list(global_assets['locations'])),
            "props": sorted(list(global_assets['props'])),
            "cast": characters
        },
        "character_registry": registry,
        "timeline": timeline
    }

    with open(cfg["paths"]["output_file"], "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    logger.info("Success.")

if __name__ == "__main__":
    main()