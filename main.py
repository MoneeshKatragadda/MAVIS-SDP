# -*- coding: utf-8 -*-
import yaml, json, re, logging
from collections import Counter
from src.extractor import NLPExtractor
from src.llm_reasoner import LLMReasoner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("MAVIS")

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def aggregate_scene_emotion(beats):
    labels, intensities = [], []
    for b in beats:
        emo = b.get("emotion")
        if emo:
            labels.append(emo["label"])
            intensities.append(emo["intensity"])
    if not labels:
        return {"dominant_emotion": "neutral", "emotion_intensity": 0.3}
    return {
        "dominant_emotion": Counter(labels).most_common(1)[0][0],
        "emotion_intensity": round(sum(intensities) / len(intensities), 3)
    }

def main():
    logger.info("Starting MAVIS Phase-1.10 (Production Logic)")

    cfg = load_config()
    with open(cfg["paths"]["input_file"], "r", encoding="utf-8") as f:
        raw_text = f.read()
        # Clean [source] tags and normalize quotes
        text = re.sub(r'\\', '', raw_text)
        text = re.sub(r"[“”]", '"', text)

    # Split into paragraphs/scenes
    scenes = [s.strip() for s in text.split("\n\n") if s.strip()]
    
    extractor = NLPExtractor(cfg)
    extractor.load_emotion_model()
    llm = LLMReasoner(cfg)

    # 1. EXTRACT CHARACTERS
    # The new logic will exclude 'Board' (ORG) and 'Adam' (Possessive)
    characters = extractor.extract_characters_from_text(text)
    logger.info(f"Final Cast: {characters}")

    timeline = []
    global_cursor = 0.0

    for i, scene in enumerate(scenes, 1):
        logger.info(f"Processing scene {i}/{len(scenes)}")

        # 2. Parse Structure (Dialogue vs Narration)
        struct = extractor.parse_scene_structure(scene, characters)
        
        # 3. Entity Classification (Props vs Characters)
        scene_entities = extractor.extract_scene_entities(scene, characters)
        for char in struct["active_chars"]:
            scene_entities.append({"name": char, "type": "character", "role": "foreground"})

        # 4. Beat Processing with Absolute Timing
        scene_dur = 0.0
        proc_beats = []
        for beat in struct["beats"]:
            # NLP & Semantics
            emo = extractor.get_emotion(beat["text"], beat["type"])
            beat["emotion"] = emo
            beat["semantic"] = extractor.extract_svo(beat["text"])
            beat["audio_prompt"] = extractor.build_audio_prompt(beat, emo)
            
            # Timing
            beat["timing"] = {
                "start": round(scene_dur, 2),
                "end": round(scene_dur + beat["duration"], 2),
                "duration": beat["duration"]
            }
            scene_dur += beat["duration"]
            proc_beats.append(beat)

        # 5. Scene-Level AI Analysis
        scene_emotion = aggregate_scene_emotion(proc_beats)
        intent = llm.analyze_scene_intent(scene, scene_emotion)
        sfx_list = extractor.extract_sfx(scene)

        # 6. Event Object Construction
        timeline.append({
            "id": f"scene_{i:02d}",
            "timing": {
                "global_start": round(global_cursor, 2),
                "global_end": round(global_cursor + scene_dur, 2),
                "duration": round(scene_dur, 2)
            },
            "scene_entities": scene_entities,
            "script": {
                "text": scene,
                "active_chars": struct["active_chars"]
            },
            "audio": {
                "bgm": intent["bgm"],
                "sfx": sfx_list,
                "dialogue_emotion_mood": scene_emotion['dominant_emotion']
            },
            "visuals": {
                "camera": intent["camera"],
                "prompt": f"Cinematic shot, {scene_emotion['dominant_emotion']} mood"
            },
            "beats": proc_beats
        })
        global_cursor += scene_dur

    # 7. Generate Registry (Voices)
    registry = llm.generate_registry(characters, text)

    # 8. Save
    output = {
        "meta": {"fps": 24, "total_duration": round(global_cursor, 2)},
        "registry": registry,
        "timeline": timeline
    }

    with open(cfg["paths"]["output_file"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("MAVIS Pipeline Complete. Check events.json.")

if __name__ == "__main__":
    main()