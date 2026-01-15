# -*- coding: utf-8 -*-
import re
import spacy
import torch
from transformers import pipeline
from nltk.corpus import wordnet as wn

class NLPExtractor:
    def __init__(self, config):
        self.nlp = spacy.load(config["models"]["spacy_model"])
        self.emotion_model = config["models"]["emotion_model"]
        self._emotion_pipe = None
        
        # Acoustic triggers for SFX generation
        self.sound_lemmas = {
            "drum", "hiss", "snap", "click", "scream", "whisper", "shout", 
            "thud", "bang", "ring", "crash", "creak", "rustle", "beep", "siren", "slap"
        }

    # ---------------- Emotion ----------------
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
            
        # Truncate to model max length
        scores = self._emotion_pipe(text[:256])[0]
        scores = {s["label"].lower(): s["score"] for s in scores}

        label = max(scores, key=scores.get)
        intensity = min(float(scores[label]), 0.95)

        return {"label": label, "intensity": round(intensity, 3)}

    # ---------------- SFX Extraction ----------------
    def extract_sfx(self, text):
        doc = self.nlp(text)
        sfx_triggers = []
        for t in doc:
            # Check for sound verbs or nouns
            if t.lemma_.lower() in self.sound_lemmas:
                sfx_triggers.append(t.lemma_.lower())
            # Check for environmental nouns
            elif t.text.lower() in ["rain", "thunder", "silence", "noise", "traffic", "wind"]:
                 sfx_triggers.append(t.text.lower())
        return sorted(list(set(sfx_triggers)))

    # ---------------- Scene + Speaker Resolution ----------------
    def parse_scene_structure(self, text, characters):
        beats = []
        # Split by quotes to separate dialogue
        parts = re.split(r'(".*?")', text)
        last_speaker = None
        
        for p in parts:
            if not p.strip(): continue

            if p.startswith('"') and p.endswith('"'):
                clean_text = p.strip('"')
                speaker = self._resolve_speaker_lookaround(text, p, characters, last_speaker)
                
                beats.append({
                    "type": "dialogue",
                    "speaker": speaker, 
                    "text": clean_text,
                    "duration": self._duration(clean_text)
                })
                if speaker: last_speaker = speaker
            else:
                beats.append({
                    "type": "narration",
                    "text": p.strip(),
                    "duration": self._duration(p)
                })

        active_chars = sorted(
            {b["speaker"] for b in beats if b.get("speaker") and b["speaker"] in characters}
        )
        return {"beats": beats, "active_chars": active_chars}

    def _resolve_speaker_lookaround(self, full_text, quote, characters, last_speaker):
        # Locate the quote in the full text
        idx = full_text.find(quote)
        if idx == -1: return last_speaker
        
        # Look at surrounding text for context
        pre = full_text[max(0, idx-100):idx]
        post = full_text[idx+len(quote):min(len(full_text), idx+len(quote)+100)]
        
        context_doc = self.nlp(pre + " ... " + post)
        speech_verbs = {"say", "ask", "whisper", "shout", "hiss", "mutter", "yell", "interrupt", "command", "reply", "warn"}
        
        # 1. Dependency Check: Subject of a speech verb?
        for t in context_doc:
            if t.lemma_ in speech_verbs:
                for child in t.children:
                    if child.dep_ == "nsubj" and child.text in characters:
                        return child.text
        
        # 2. Proximity Check: Nearest character in pre-text
        for char in reversed(characters):
            if char in pre: return char

        return last_speaker

    # ---------------- Entity Typing ----------------
    def extract_scene_entities(self, text, characters):
        doc = self.nlp(text)
        entities = []
        for t in doc:
            if t.text in characters: continue
            if t.pos_ in {"NOUN", "PROPN"}:
                etype = self._classify_entity_fast(t)
                if etype in {"prop", "location", "organization"}:
                    entities.append({"name": t.lemma_.lower(), "type": etype, "role": "background"})

        # Deduplicate entities
        seen = set()
        final = []
        for e in entities:
            if e["name"] not in seen:
                final.append(e)
                seen.add(e["name"])
        return final

    def _classify_entity_fast(self, token):
        if token.ent_type_ == "ORG": return "organization"
        if token.ent_type_ in {"GPE", "LOC", "FAC"}: return "location"
        
        synsets = wn.synsets(token.lemma_, pos=wn.NOUN)
        for s in synsets:
            # Check if it's a physical artifact
            if "artifact.n.01" in [p.name() for p in s.lowest_common_hypernyms(wn.synset('artifact.n.01'))]:
                return "prop"
        return "abstract"

    # ---------------- Semantics ----------------
    def extract_svo(self, text):
        doc = self.nlp(text)
        for t in doc:
            if t.dep_ == "nsubj":
                # Ignore auxiliary verbs (e.g., "is" in "is running")
                if t.head.lemma_ in {"be", "have", "do"} and t.head.dep_ == "aux": continue
                
                verb = t.head.lemma_
                obj = None
                for c in t.head.children:
                    if c.dep_ in {"dobj", "pobj"}:
                        obj = c.text
                        break
                return {"subject": t.text, "action": verb, "object": obj}
        return {"subject": None, "action": None, "object": None}

    def build_audio_prompt(self, beat, emo):
        speaker = beat.get("speaker", "Narrator") or "Narrator"
        return f"{speaker}, tone={emo['label']}, intensity={emo['intensity']}"

    # ---------------- ROBUST CHARACTER EXTRACTION ----------------
    def extract_characters_from_text(self, text):
        """
        Extracts active characters while excluding Organizations (Board) and Objects.
        """
        doc = self.nlp(text)
        candidates = set()
        
        # Grammatical roles that imply an actor
        active_deps = {"nsubj", "nsubjpass", "dobj", "iobj", "attr", "vocative", "appos"}
        
        for t in doc:
            if t.pos_ == "PROPN":
                clean_name = t.text.strip().strip('"').strip("'")
                
                # Rule 1: Exclude known non-person entities
                if t.ent_type_ in {"ORG", "GPE", "LOC", "FAC", "PRODUCT", "DATE", "TIME"}:
                    continue
                
                # Rule 2: Must play an active role
                if t.dep_ in active_deps:
                    if len(clean_name) > 2 and clean_name[0].isupper():
                        candidates.add(clean_name)
                
                # Rule 3: Allow NER 'PERSON' if not caught by Rule 1
                elif t.ent_type_ == "PERSON":
                    if t.dep_ not in {"poss", "compound", "amod"}: # Exclude "Adam's"
                         if len(clean_name) > 2 and clean_name[0].isupper():
                            candidates.add(clean_name)

        return sorted(list(candidates))

    def _duration(self, text):
        # Approx 3 words per second for pacing
        return round(max(1.5, len(text.split()) / 3.0), 2)