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
        
        self.character_visuals = {} # Store consistency profiles
        self.character_metadata = {} # Store gender and style

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

    def analyze_cast_visuals(self, story_text, characters):
        """Generates consistent visual descriptions for characters (Face, Clothes)."""
        char_list = ", ".join(characters)
        intro_text = story_text[:1500].replace("\n", " ")
        
        prompt = f"""Instruction: Create a specific, consistent VIDEO GENERATION visual description for each character.
Story Context: {intro_text}...
Characters: {char_list}

Rules:
1. Include Ethnicity, Age, Gender, Hair Style, Facial Features, and Clothing.
2. Clothing must fit a Noir/Detective setting.
3. Keep it to 1 sentence per character.
4. Keep the character style and facial features consistent throughout the story.

Format:
Name | Gender | Clothing Style | [Visual Description]

Response:
"""
        out = self.llm(prompt, max_tokens=300, stop=["Instruction:", "Story:"], temperature=0.7)
        raw_text = out["choices"][0]["text"]
        
        visuals = {}
        metadata = {}

        for line in raw_text.split("\n"):
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    name = parts[0]
                    gender = parts[1]
                    style = parts[2]
                    desc = parts[3]
                
                    if not desc.strip(): continue
                    
                    # Fuzzy match name
                    for c in characters:
                        if c in name:
                            visuals[c] = desc
                            metadata[c] = {"gender": gender, "style": style}
                            break
        
        # Defaults
        default_looks = [
            ("Male", "Detective", "wearing a trench coat and fedora, sharp facial features, 30s"),
            ("Female", "Elegant", "wearing an elegant evening dress, wavy hair, 20s"),
            ("Male", "Disheveled", "wearing a crumpled suit, tired expression, 40s"),
            ("Male", "Tough", "wearing a leather jacket, intense gaze, 30s")
        ]
        
        for i, c in enumerate(characters):
            if c not in visuals:
                # Assign a stable default implementation based on index
                gender, style, look = default_looks[i % len(default_looks)]
                visuals[c] = f"{c}, {look}"
                metadata[c] = {"gender": gender, "style": style}
        
        self.character_visuals = visuals
        self.character_metadata = metadata
        logger.info(f"Generated Visual Profiles: {visuals}")
        return visuals

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

    def generate_visual_prompt_v2(self, beat_data, location, active_cast):
        """
        New handler for strict visual prompts.
        beat_data: dict containing 'type', 'text', 'speaker', 'emotion'
        location: str current location name/desc
        active_cast: list of char names present in scene
        """
        b_type = beat_data['type']
        text = beat_data['text']
        emotion = beat_data.get('emotion', {}).get('label', 'neutral')
        
        # Resolve Characters
        # For dialogue, focus on speaker
        if b_type == 'dialogue':
            speaker = beat_data.get('speaker', 'Unknown')
            
            # Robust lookup
            char_desc = self.character_visuals.get(speaker, "")
            if not char_desc:
                # If lookup fails, fallback to name+noir text
                char_desc = f"{speaker}, a noir character in detective attire"
            
            # Strict Template
            # ensure no empty description
            if not char_desc.strip(): char_desc = f"{speaker}, a noir character"

            # "Cinematic close up shot of <character> with <emotion> expression with blurry backround of dimly lit dinner"
            prompt = f"Cinematic close up shot of {char_desc} with {emotion} expression with blurry background of {location}"
            return prompt

        # For Narration
        else:
            # Construct context of who is visible
            visible_people = ", ".join([f"{c} ({self.character_visuals.get(c, 'Noir figure')})" for c in active_cast])
            if not visible_people: visible_people = "No specific characters, focus on environment"

            prompt = f"""Task: Convert the Narration into a precise Visual Description for Video Generation.
Narration: "{text}"
Location: {location}
Characters: {visible_people}

Instructions:
1. STRICTLY adhere to the actions in the text. Do NOT add movement (walking, standing, entering) unless explicitly stated.
2. If no action is described, assume characters are STATIONARY (sitting).
3. PRESERVE specific object names (e.g., "briefcase", "envelope") exactly. Do not change them to generic items like "box".
4. Replace abstract atmosphere (e.g., "tense") with visual lighting details (e.g., "sharp shadows", "dim lighting").
5. Describe the scene visually but DO NOT invent new narrative events.

Example:
Narration: "Silas closed the lid."
Visual Description: Cinematic close up shot of Silas closing the lid of the briefcase, his hands steady, under a dim overhead light.

Narration: "{text}"
Visual Description:"""
            
            out = self.llm(prompt, max_tokens=150, stop=["\n", "Narration:", "Task:"], temperature=0.3)
            gen = out["choices"][0]["text"].strip()
            
            # Clean
            gen = self._clean_visual_output(gen)
            
            # Improved Fallback
            if not gen: 
                 # Fallback: Use the text but dress it up visually
                 clean_text = text
                 for prefix in ["Inside, ", "Outside, ", "Suddenly, ", "Meanwhile, "]:
                     if clean_text.startswith(prefix):
                         clean_text = clean_text[len(prefix):]
                 gen = f"Cinematic shot of {clean_text}"
            
            # Prefix check - Ensure it starts with a camera shot type or Cinematic
            lower_gen = gen.lower()
            if not any(x in lower_gen for x in ["cinematic", "shot", "close up", "wide", "view", "pan", "zoom"]):
                 gen = f"Cinematic color shot of {gen}"
                 
            return gen

    def generate_rich_registry(self, characters, profiles):
        registry = {}
        for char in characters:
            meta = self.character_metadata.get(char, {"gender": "Unknown", "style": "Noir"})
            registry[char] = {
                "voice_model_id": f"en_us_generic_{char.lower()}",
                "archetype": profiles.get(char, "Standard"),
                "gender": meta.get("gender", "Unknown"),
                "clothing_style": meta.get("style", "Noir Standard")
            }
        return registry