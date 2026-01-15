# -*- coding: utf-8 -*-
from llama_cpp import Llama
import json

class LLMReasoner:
    def __init__(self, config):
        m = config["models"]
        # Load local GGUF model
        self.llm = Llama(
            model_path=m["llm_model_path"],
            n_ctx=m.get("llm_context_window", 2048),
            n_gpu_layers=min(m.get("llm_gpu_layers", 12), 12),
            verbose=False
        )

    def analyze_scene_intent(self, scene_text, scene_emotion):
        # Focused prompt for Video/Audio metadata
        prompt = f"""[INST] Analyze the scene.
Scene: "{scene_text}"
Emotion: {scene_emotion['dominant_emotion']}

Output format:
BGM: [Genre like Suspense, Jazz, Cyberpunk]
CAMERA: [Angle like Low Angle, Tracking Shot, Close Up]
[/INST]"""
        
        out = self.llm(prompt, max_tokens=64, stop=["\n"])
        text = out["choices"][0]["text"].strip().lower()
        
        bgm = "Suspense"
        camera = "Static Medium Shot"
        
        for line in text.splitlines():
            if "bgm:" in line: bgm = line.split("bgm:")[1].strip().title()
            if "camera:" in line: camera = line.split("camera:")[1].strip().title()
            
        return {"bgm": bgm, "camera": camera}

    def generate_registry(self, characters, full_text):
        char_list = ", ".join(characters)
        # Generate Character Voice Registry
        prompt = f"""[INST] Assign voice archetypes to: {char_list}.
Context: "{full_text[:800]}..."

Format JSON:
{{ "Name": {{ "voice": "Deep/Soft/Gravelly", "base_emotion": "Calm/Anxious" }} }}
[/INST]"""

        out = self.llm(prompt, max_tokens=256)
        try:
            text = out["choices"][0]["text"]
            # Extract JSON from response
            s = text.find("{")
            e = text.rfind("}") + 1
            if s != -1 and e != -1:
                return json.loads(text[s:e])
        except:
            pass
            
        # Fallback if LLM parsing fails
        return {c: {"voice": "Neutral", "base_emotion": "Neutral"} for c in characters}