# -*- coding: utf-8 -*-
import re
import spacy
import torch
from transformers import pipeline
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class NLPExtractor:
    def __init__(self, config):
        self.nlp = spacy.load(config["models"]["spacy_model"])
        self.emotion_model = config["models"]["emotion_model"]
        self._emotion_pipe = None
        self.sia = SentimentIntensityAnalyzer()
        
        self.sound_lemmas = {
            "drum", "hiss", "snap", "click", "scream", "whisper", "shout", 
            "thud", "bang", "ring", "crash", "creak", "rustle", "beep", "siren", 
            "slap", "gulp", "swallow", "breath", "step", "clatter", "buzz", "engine", "run"
        }

        # Blocklist for abstract non-visual nouns common in scripts
        self.entity_blocklist = {
            "way", "time", "idea", "series", "security", "side", "line", 
            "business", "risk", "reward", "moment", "hair", "hairs", "grace",
            "name", "internet", "signal", "fence", "air", "tension", "rhythm",
            "aggression", "bucket", "life", "spreadsheet", "account", "glow"
        }

    # ---------------- Emotion & Tone ----------------
    def load_emotion_model(self):
        if self._emotion_pipe is None:
            device = 0 if torch.cuda.is_available() else -1
            self._emotion_pipe = pipeline(
                "text-classification",
                model=self.emotion_model,
                top_k=None,
                device=device
            )

    def get_emotion(self, text, beat_type):
        if not text.strip(): 
            return {"label": "neutral", "intensity": 0.0}
            
        scores = self._emotion_pipe(text[:256])[0]
        scores = {s["label"].lower(): s["score"] for s in scores}
        label = max(scores, key=scores.get)
        intensity = min(float(scores[label]), 0.98)

        # --- GENRE-AWARE RE-MAPPING ---
        # In a Thriller/Noir context, "Joy" is rarely genuine joy.
        if label == "joy":
            sentiment = self.sia.polarity_scores(text)
            if sentiment['compound'] < 0.1: 
                # If sentiment is negative or neutral, 'joy' is a model error for 'relief' or 'manic'
                # Map to 'relief' if intensity is low, 'manic' if high, or fallback to 'neutral'
                label = "relief" if intensity < 0.8 else "manic"
            else:
                # Genuine positive sentiment in noir is usually 'satisfaction'
                label = "satisfaction"
        
        if label == "sadness":
            # Map 'sadness' to genre-appropriate 'resignation' or 'melancholy'
            label = "melancholy"

        return {"label": label, "intensity": round(intensity, 3)}

    # ---------------- SFX & Semantics ----------------
    def extract_sfx(self, text):
        doc = self.nlp(text)
        sfx_triggers = []
        for t in doc:
            if t.lemma_.lower() in self.sound_lemmas:
                sfx_triggers.append(t.lemma_.lower())
            elif t.ent_type_ == "EVENT" or t.text.lower() in ["rain", "thunder", "silence", "noise", "wind"]:
                 sfx_triggers.append(t.text.lower())
        return sorted(list(set(sfx_triggers)))

    def extract_svo(self, text):
        doc = self.nlp(text)
        for t in doc:
            if t.dep_ == "nsubj":
                action_token = t.head
                verb = action_token.lemma_
                
                # Semantic normalization for aux verbs
                if action_token.text.lower() in ["'re", "'m", "'s", "'ve"]:
                    if action_token.lemma_ == "be": verb = "is/are"
                
                # If verb is auxiliary, check if it's the main root
                if action_token.dep_ == "aux":
                    verb = action_token.head.lemma_

                obj = None
                # Check for direct objects, prepositional objects, or attributes (for "be" verbs)
                for c in action_token.children:
                    if c.dep_ in {"dobj", "pobj", "attr", "acomp"}:
                        obj = c.text
                        break
                
                return {"subject": t.text, "action": verb, "object": obj}
        return {"subject": None, "action": None, "object": None}

    # ---------------- Speaker Resolution ----------------
    def parse_scene_structure(self, text, characters):
        beats = []
        parts = re.split(r'(".*?")', text)
        last_speaker = None
        last_narration_subject = None # Track who acted in the last narration
        
        for p in parts:
            if not p.strip(): continue

            if p.startswith('"') and p.endswith('"'):
                clean_text = p.strip('"')
                # Pass the last narration subject as a strong hint
                speaker = self._resolve_speaker(text, p, characters, last_speaker, last_narration_subject)
                
                beats.append({
                    "type": "dialogue",
                    "speaker": speaker, 
                    "text": clean_text,
                    "duration": self._duration(clean_text, 2.5)
                })
                if speaker and speaker != "Unknown":
                    last_speaker = speaker
                last_narration_subject = None # Reset after dialogue
            else:
                # Narration: Find the subject (Actor)
                subj = self._find_narration_subject(p, characters)
                if subj: last_narration_subject = subj
                
                beats.append({
                    "type": "narration",
                    "text": p.strip(),
                    "duration": self._duration(p, 2.0)
                })

        active_chars = sorted({b["speaker"] for b in beats if b.get("speaker") and b["speaker"] in characters})
        return {"beats": beats, "active_chars": active_chars}

    def _find_narration_subject(self, text, characters):
        doc = self.nlp(text)
        for t in doc:
            if t.dep_ == "nsubj" and t.text in characters:
                return t.text
        return None

    def _resolve_speaker(self, full_text, quote, characters, last_speaker, last_narration_subject):
        idx = full_text.find(quote)
        if idx == -1: return last_speaker
        
        pre = full_text[max(0, idx-100):idx]
        post = full_text[idx+len(quote):min(len(full_text), idx+len(quote)+100)]
        
        combined_context = pre + " ... " + post
        doc = self.nlp(combined_context)
        
        speech_verbs = {"say", "ask", "whisper", "shout", "hiss", "mutter", "yell", "interrupt", "command", "reply", "warn", "snap"}
        
        # 1. Grammar Check: Subject of a speech verb in context
        for t in doc:
            if t.lemma_ in speech_verbs:
                for child in t.children:
                    if child.dep_ == "nsubj" and child.text in characters:
                        return child.text

        # 2. Preceding Action Heuristic (The "Silas took a sip" fix)
        # If the narration immediately before this quote had a character subject, they are likely speaking.
        if last_narration_subject:
            return last_narration_subject

        # 3. Proximity Check (Nearest character in pre-text)
        pre_doc = self.nlp(pre)
        for ent in reversed(pre_doc.ents):
            if ent.text in characters:
                return ent.text

        return last_speaker or "Unknown"

    # ---------------- Entity Extraction ----------------
    def extract_scene_entities(self, text, characters):
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.text in characters: continue
            
            # --- STRICT FILTER ---
            if ent.text.lower() in self.entity_blocklist: continue
            
            role = "background"
            etype = "prop"
            
            if ent.label_ in {"GPE", "LOC", "FAC"}:
                etype = "location"
            elif ent.label_ == "ORG":
                etype = "organization" 
            elif ent.label_ in {"PRODUCT", "WORK_OF_ART"}:
                etype = "prop"
            else:
                continue

            entities.append({"name": ent.text, "type": etype, "role": role})

        # WordNet fallback with Abstract Filter
        for t in doc:
            if t.text.lower() in self.entity_blocklist: continue
            if len(t.text) < 3: continue
            
            if t.pos_ == "NOUN" and not t.ent_type_:
                synsets = wn.synsets(t.lemma_, pos=wn.NOUN)
                is_phys = False
                is_abstract = False
                
                for s in synsets:
                    hypernyms = [p.name() for p in s.lowest_common_hypernyms(wn.synset('entity.n.01'))]
                    if "artifact.n.01" in hypernyms or "physical_entity.n.01" in hypernyms:
                        is_phys = True
                    if "abstraction.n.06" in hypernyms or "attribute.n.02" in hypernyms:
                        is_abstract = True
                
                # Only include if Physical AND NOT Abstract
                if is_phys and not is_abstract:
                    entities.append({"name": t.text.lower(), "type": "prop", "role": "background"})

        unique_entities = []
        seen = set()
        for e in entities:
            key = (e['name'].lower(), e['type'])
            if key not in seen:
                unique_entities.append(e)
                seen.add(key)
                
        return unique_entities

    def extract_characters_from_text(self, text):
        doc = self.nlp(text)
        candidates = set()
        BLOCKLIST = {"Board", "Formica", "Teflon", "Monday", "Sunday", "Rusty Anchor", "Anchor", "Adam", "Street", "Station"}
        active_deps = {"nsubj", "nsubjpass", "dobj", "iobj", "attr", "vocative", "appos"}
        
        for t in doc:
            if t.pos_ == "PROPN":
                clean_name = t.text.strip().strip('"').strip("'")
                if clean_name in BLOCKLIST: continue
                if t.ent_type_ in {"ORG", "GPE", "LOC", "FAC", "PRODUCT", "DATE", "TIME"}: continue
                
                if t.dep_ in active_deps:
                    if len(clean_name) > 2 and clean_name[0].isupper():
                        candidates.add(clean_name)
                elif t.ent_type_ == "PERSON":
                    if t.dep_ not in {"poss", "compound", "amod"}:
                         if len(clean_name) > 2 and clean_name[0].isupper():
                            candidates.add(clean_name)

        return sorted(list(candidates))

    def build_audio_prompt(self, beat, emo):
        speaker = beat.get("speaker", "Narrator") or "Narrator"
        return f"{speaker}, tone={emo['label']}, intensity={emo['intensity']}"

    def _duration(self, text, speed_factor):
        return round(max(1.0, len(text.split()) / speed_factor), 2)