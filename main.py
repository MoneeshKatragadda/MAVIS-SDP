# -*- coding: utf-8 -*-
import yaml, json, re, logging
import warnings
import os
from collections import Counter, deque

# Silence library noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.extractor import NLPExtractor
from src.llm_reasoner import LLMReasoner

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
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
    logger.info("Initializing MAVIS Pipeline 3.3 (Visual Hallucination Fix)")
    cfg = load_config()
    
    with open(cfg["paths"]["input_file"], "r", encoding="utf-8") as f:
        raw = f.read()
        clean_text = raw.replace('\\', '').replace('’', "'").replace('“', '"').replace('”', '"')
        
    scenes_text = [s.strip() for s in clean_text.split("\n\n") if s.strip()]
    
    nlp = NLPExtractor(cfg)
    nlp.load_emotion_model()
    llm = LLMReasoner(cfg)

    # 1. Identify Cast
    characters = nlp.extract_characters_from_text(clean_text)
    logger.info(f"Cast identified: {characters}")

    # 2. Dynamic Profiling
    logger.info("Analyzing character archetypes...")
    cast_profiles = llm.analyze_cast_profiles(clean_text, characters)

    timeline = []
    global_cursor = 0.0
    global_assets = {"locations": set(), "props": set()}
    last_location = "Start"
    total_beats_count = 0

    context_buffer = deque(maxlen=3) 

    for i, s_text in enumerate(scenes_text, 1):
        logger.info(f"--- Processing Scene {i} ---")
        
        struct = nlp.parse_scene_structure(s_text, characters)
        entities = nlp.extract_scene_entities(s_text, characters)
        
        for e in entities:
            if e['type'] == 'location': global_assets['locations'].add(e['name'])
            if e['type'] == 'prop': global_assets['props'].add(e['name'])
            
        scene_duration = 0.0
        beats = []
        
        current_active_char = None
        if struct['active_chars']:
            current_active_char = struct['active_chars'][0]

        for b_idx, beat in enumerate(struct['beats'], 1):
            if beat.get('speaker') and beat['speaker'] != "Unknown":
                current_active_char = beat['speaker']

            # 1. Base Emotion (LOCKED)
            base_emotion = nlp.get_emotion(beat['text'], beat['type'])
            beat['emotion'] = base_emotion 
            
            # 2. LLM Refinement (LOCKED)
            if beat['type'] == 'dialogue' and beat['speaker'] in cast_profiles:
                context_str = " | ".join(list(context_buffer)) if context_buffer else "Start of scene"
                archetype = cast_profiles[beat['speaker']]
                
                refined_emotion = llm.refine_dialogue_emotion(
                    speaker=beat['speaker'],
                    text=beat['text'],
                    archetype=archetype,
                    context_window=context_str,
                    base_emotion=base_emotion['label']
                )
                beat['emotion'] = refined_emotion
                context_buffer.append(f"{beat['speaker']}: {beat['text']}")

            # 3. Attributes
            beat['semantic'] = nlp.extract_svo(beat['text'], context_subject=current_active_char)
            beat['audio_prompt'] = nlp.build_audio_prompt(beat, beat['emotion'])
            beat['sub_scene_id'] = f"SC_{i:03d}_{b_idx:02d}"
            
            # 4. Visual Prompt (UPDATED: Fix Hallucinations)
            if beat['type'] == 'narration':
                vis_text = nlp.strip_dialogue(beat['text'])
                # 🔽 CHANGED: Pass struct['active_chars'] (Scene Specific) instead of characters (Global)
                # This prevents the model from inserting people who aren't in the scene.
                beat['visual_prompt'] = llm.generate_visual_prompt_for_beat(
                    vis_text, 
                    beat['emotion']['label'], 
                    struct['active_chars'] 
                )
            else:
                beat['visual_prompt'] = f"Close-up of {beat['speaker']}, {beat['emotion']['label']} expression, noir lighting."

            beat_dur = beat['duration']
            beat['timing'] = {
                "rel_start": round(scene_duration, 2),
                "rel_end": round(scene_duration + beat_dur, 2),
                "global_start": round(global_cursor + scene_duration, 2),
                "global_end": round(global_cursor + scene_duration + beat_dur, 2)
            }
            beats.append(beat)
            scene_duration += beat_dur
            total_beats_count += 1

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
            "beats": beats
        }
        
        timeline.append(event)
        global_cursor += scene_duration
        
        locs = [e['name'] for e in entities if e['type'] == 'location']
        if locs: last_location = locs[0]

    logger.info("Generating Global Registry...")
    registry = llm.generate_rich_registry(characters, cast_profiles)

    final_output = {
        "project_meta": {
            "title": "MAVIS_Project", 
            "fps": 24, 
            "total_duration": round(global_cursor, 2),
            "total_beats": total_beats_count,
            "validation_status": "Passed" if total_beats_count > 0 else "Failed"
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
    
    logger.info(f"Success. Total Beats: {total_beats_count}")

if __name__ == "__main__":
    main()