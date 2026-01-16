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
        # Enriched Prompt for Visuals
        prompt = f"""[INST] You are a Video Director. Analyze this scene for production.
Scene: "{scene_text}"
Context: Previous loc: {prev_loc}. Mood: {emotions['dominant_emotion']}.

Output JSON with:
1. "visual_prompt": A detailed Stable Diffusion prompt (Subject, Action, Lighting, Style).
2. "bgm": Genre.
3. "camera": Shot type.
4. "transition": Cut type.

JSON:
{{
  "visual_prompt": "string",
  "bgm": "string",
  "camera": "string",
  "transition": "string"
}}
[/INST]"""
        
        out = self.llm(prompt, max_tokens=200)
        data = self._clean_json(out["choices"][0]["text"])
        
        # Fallback if LLM fails
        return {
            "visual_prompt": data.get("visual_prompt", f"Cinematic shot, {emotions['dominant_emotion']} atmosphere, {scene_text[:30]}..."),
            "bgm": data.get("bgm", "Ambient"),
            "camera": data.get("camera", "Static Medium"),
            "transition": data.get("transition", "Hard Cut")
        }

    def generate_rich_registry(self, characters, summary):
        char_list = ", ".join(characters)
        prompt = f"""[INST] Create a Character Registry for: {char_list}.
Story Context: {summary[:500]}...

Assign:
1. 'voice_model_id': A specific TTS identifier string.
2. 'archetype': Personality description.

JSON Format:
{{
  "Name": {{ "voice_model_id": "string", "archetype": "string" }}
}}
[/INST]"""

        out = self.llm(prompt, max_tokens=512)
        return self._clean_json(out["choices"][0]["text"])