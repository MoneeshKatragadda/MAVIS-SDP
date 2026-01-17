# -*- coding: utf-8 -*-
from llama_cpp import Llama
import json
import re

class LLMReasoner:
    def __init__(self, config):
        m = config["models"]
        self.llm = Llama(
            model_path=m["llm_model_path"],
            n_ctx=m.get("llm_context_window", 2048),
            n_gpu_layers=min(m.get("llm_gpu_layers", 12), 12),
            verbose=False
        )

    def _clean_json(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == -1: return {}
            return json.loads(text[start:end])
        except:
            return {}

    def analyze_scene_production(self, scene_text, prev_loc, emotions):
        prompt = f"""[INST] Director Mode.
Scene: "{scene_text[:500]}"
Mood: {emotions['dominant_emotion']}

JSON (bgm, camera, transition):
{{
  "bgm": "string",
  "camera": "string",
  "transition": "string"
}}
[/INST]"""
        
        out = self.llm(prompt, max_tokens=100, stop=["}"])
        raw = "{" + out["choices"][0]["text"] + "}"
        data = self._clean_json(raw)
        
        return {
            "bgm": data.get("bgm", "Ambient"),
            "camera": data.get("camera", "Static Medium"),
            "transition": data.get("transition", "Hard Cut")
        }

    def generate_visual_prompt_for_beat(self, beat_text, emotion, allowed_cast):
        cast_str = ", ".join(allowed_cast)
        prompt = f"""[INST] Describe image.
Action: {beat_text}
Mood: {emotion}
Allowed Characters: {cast_str}
Constraint: Visuals only. Max 12 words. No dialogue.

Description: [/INST]"""
        
        out = self.llm(prompt, max_tokens=48, stop=["\n", ".", '"'])
        vis = out["choices"][0]["text"].strip()
        
        # Validation: Check for hallucinated names
        words = vis.split()
        safe_vis = vis
        
        for w in words:
            clean_w = w.strip(",.")
            if clean_w[0].isupper() and clean_w not in allowed_cast and clean_w.lower() not in ["cinematic", "noir", "the", "a", "an"]:
                safe_vis = None # Found hallucination, revert to fallback
                break

        # Fallback Logic (Updated to avoid 30 char slicing)
        if not safe_vis or len(safe_vis) < 5 or "import" in safe_vis:
            clean_beat = beat_text.replace('"', '').strip()
            # Limit to 15 words to ensure video generator context without overflow
            truncated_beat = " ".join(clean_beat.split()[:15])
            safe_vis = f"Cinematic shot of {truncated_beat}, {emotion} lighting."
            
        return safe_vis

    def generate_rich_registry(self, characters, summary):
        char_list = ", ".join(characters)
        prompt = f"""[INST] Assign Voices.
Characters: {char_list}
Context: {summary[:400]}...

Format: Name: VoiceID | Archetype
Output: [/INST]"""

        out = self.llm(prompt, max_tokens=256)
        raw_text = out["choices"][0]["text"]
        
        registry = {}
        for line in raw_text.splitlines():
            match = re.search(r"(\w+):\s*([\w_]+)\s*\|\s*(.+)", line)
            if match:
                name, voice, arch = match.groups()
                if name in characters:
                    registry[name] = {
                        "voice_model_id": voice.strip(),
                        "archetype": arch.strip()
                    }
        return registry