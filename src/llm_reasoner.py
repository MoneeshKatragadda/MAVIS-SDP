# -*- coding: utf-8 -*-
from llama_cpp import Llama
import re
import logging

logger = logging.getLogger("MAVIS_LLM")

class LLMReasoner:
    def __init__(self, config):
        m = config["models"]
        self.llm = Llama(
            model_path=m["llm_model_path"],
            n_ctx=m.get("llm_context_window", 2048),
            n_gpu_layers=min(m.get("llm_gpu_layers", 12), 20),
            verbose=False
        )
        
        self.VALID_EMOTIONS = {
            "suspicion", "paranoia", "dread", "calculating", "cynical", 
            "defensive", "threatening", "desperate", "sarcastic", "cold",
            "relieved", "urgent", "intimidating", "neutral", "anger",
            "fear", "annoyance", "curiosity", "joy", "sadness", "amusement",
            "disapproval"
        }

        self.FORBIDDEN_LABELS = {"intensity", "emotion", "score", "value", "label", "tone"}

    def _parse_key_value(self, text, key):
        pattern = rf"{key}[:\-\s]+([a-zA-Z0-9_\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _clean_visual_output(self, text):
        """
        Removes hallucinations like 'Output:', 'Input:', brackets, and meta-text.
        """
        # Remove common LLM prefixes
        text = re.sub(r"^(Output|Response|Visual|Description|Input|Stable Diffusion)[:\-\s]*", "", text, flags=re.IGNORECASE)
        
        # Remove anything in square brackets [Mood] or (Context)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        
        # Remove quotes
        text = text.replace('"', '').replace("'", "")
        
        # Remove trailing "lighting" if it dangles (e.g., "Julian stood up, lighting.")
        text = re.sub(r",\s*lighting\.?$", "", text, flags=re.IGNORECASE)
        
        return text.strip()

    def analyze_cast_profiles(self, story_text, characters):
        char_list = ", ".join(characters)
        intro_text = story_text[:1000].replace("\n", " ")
        
        prompt = f"""Instruction: Assign a 3-word Noir Archetype to each character.
Story: {intro_text}...
Characters: {char_list}

Format:
Name: Adjective, Adjective, Adjective

Response:
"""
        out = self.llm(prompt, max_tokens=256, stop=["Instruction:", "Story:"])
        raw_text = out["choices"][0]["text"]
        
        profiles = {}
        for line in raw_text.split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                name = parts[0].strip()
                arch = parts[1].strip()
                for c in characters:
                    if c in name and len(arch) > 3:
                        profiles[c] = arch
                        break
        
        defaults = ["Stoic", "Nervous", "Femme Fatale", "Enforcer"]
        for i, c in enumerate(characters):
            if c not in profiles:
                profiles[c] = f"Noir Character, {defaults[i % len(defaults)]}"
                
        return profiles

    def refine_dialogue_emotion(self, speaker, text, archetype, context_window, base_emotion):
        prompt = f"""Instruction: Identify the hidden Noir subtext.
Character: {speaker} ({archetype})
Line: "{text}"
Context: {context_window}
Surface Emotion: {base_emotion}

Task: Ignore surface meaning. If text is "I'm clean" or "You're welcome", find the dark subtext (Defensive, Sarcastic).
Valid Options: Suspicion, Paranoia, Dread, Defensive, Sarcastic, Threatening, Desperate.

Format:
EMOTION: [One word]
INTENSITY: [0.0 - 1.0]

Response:
"""
        out = self.llm(prompt, max_tokens=60, stop=["Instruction:", "Line:"], temperature=0.1)
        raw_text = out["choices"][0]["text"]
        
        pred_label = self._parse_key_value(raw_text, "EMOTION")
        pred_score = self._parse_key_value(raw_text, "INTENSITY")
        
        final_label = base_emotion
        final_score = 0.8

        if pred_label:
            clean_label = pred_label.lower().strip()
            if clean_label in self.FORBIDDEN_LABELS:
                pass
            elif clean_label in self.VALID_EMOTIONS:
                final_label = clean_label
            elif len(clean_label) > 2:
                final_label = clean_label

        if final_label in ["approval", "joy", "gratitude", "caring"]:
            lower_text = text.lower()
            if "clean" in lower_text or "swear" in lower_text:
                final_label = "defensive"
            elif "welcome" in lower_text or "thanks" in lower_text:
                final_label = "sarcastic"
            else:
                final_label = "relief"

        if pred_score:
            try:
                val = float(pred_score)
                final_score = min(max(val, 0.0), 1.0)
            except ValueError:
                final_score = 0.8

        return {"label": final_label, "intensity": final_score}

    def analyze_scene_production(self, scene_text, prev_loc, emotions):
        prompt = f"""Instruction: Noir Production Design.
Scene: {scene_text[:200]}...
Mood: {emotions['dominant_emotion']}

Format:
BGM: [Music Style]
CAMERA: [Angle]

Response:
"""
        out = self.llm(prompt, max_tokens=100, stop=["Instruction:"])
        raw = out["choices"][0]["text"]
        
        bgm = self._parse_key_value(raw, "BGM") or "Suspenseful Drone"
        cam = self._parse_key_value(raw, "CAMERA") or "Static Medium"
        
        bgm = bgm.replace('"', '')
        cam = cam.replace('"', '')

        return {
            "bgm": bgm,
            "camera": cam,
            "transition": "Hard Cut"
        }

    # 🔽 UPDATED: Visual Prompt Generator (Anti-Hallucination)
    def generate_visual_prompt_for_beat(self, beat_text, emotion, allowed_cast):
        # Handle empty cast list for narration beats
        if not allowed_cast:
            cast_context = "No specific characters"
        else:
            cast_context = ", ".join(allowed_cast)
        
        # We use a "Completion Prompt" starting with 'Visual:' to force the model
        prompt = f"""Task: Describe a single cinematic shot.
Input: "{beat_text}"
Mood: {emotion}
Characters present: {cast_context}

Instructions:
- Describe ONLY what is written in the Input.
- Do NOT include characters unless they are mentioned in the Input.
- No meta-text like "Output:" or "Input:".

Visual: A noir cinematic shot of"""

        # Generate (Temperature 0.3 for creativity but control)
        out = self.llm(prompt, max_tokens=60, stop=["\n", "Task:", "Input:"], temperature=0.3)
        generated_suffix = out["choices"][0]["text"].strip()
        
        # Combine prefix + generation
        full_vis = f"A noir cinematic shot of {generated_suffix}"
        
        # Clean up artifacts
        clean_vis = self._clean_visual_output(full_vis)
        
        # Fallback: If cleaning killed the prompt or it's too short
        if len(clean_vis) < 15:
            # Safe Fallback
            clean_beat = beat_text.replace('"', '')
            clean_vis = f"A noir cinematic shot of {clean_beat}, {emotion} lighting."
            
        return clean_vis

    def generate_rich_registry(self, characters, profiles):
        registry = {}
        for char in characters:
            registry[char] = {
                "voice_model_id": f"en_us_generic_{char.lower()}",
                "archetype": profiles.get(char, "Standard")
            }
        return registry