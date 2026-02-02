import json
import os
import logging
from moviepy import ImageClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips, ColorClip
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [MOVIE] %(message)s")
logger = logging.getLogger("MAVIS_MOVIE")

EVENTS_FILE = "output/events.json"
OUTPUT_VIDEO = "output/movie.mp4"
IMG_DIR = "output/images"
AUDIO_DIR = "output/audio"
BGM_DIR = os.path.join(AUDIO_DIR, "bgm")
SFX_DIR = os.path.join(AUDIO_DIR, "sfx")

def generate_movie(events_path=EVENTS_FILE, output_file=OUTPUT_VIDEO):
    if not os.path.exists(events_path):
        logger.error(f"Events file not found at {events_path}")
        return

    with open(events_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    timeline = data.get("timeline", [])
    clips = []

    logger.info("--- ASSEMBLING MOVIE ---")

    for scene in timeline:
        scene_id = scene.get("id")
        logger.info(f"Processing Scene: {scene_id}")
        
        for beat in scene.get("beats", []):
            beat_id = beat.get("sub_scene_id")
            text = beat.get("text")
            duration = beat.get("duration", 2.0)
            
            # 1. VISUAL
            img_path = os.path.join(IMG_DIR, f"{beat_id}.png")
            if os.path.exists(img_path):
                clip = ImageClip(img_path).with_duration(duration)
            else:
                logger.warning(f"  Missing Image: {beat_id}. Using Black Placeholder.")
                clip = ColorClip(size=(1024, 1024), color=(0,0,0), duration=duration)
            
            # 2. AUDIO (Dialogue/Narration)
            audio_path = os.path.join(AUDIO_DIR, f"{beat_id}.wav")
            primary_audio = None
            if os.path.exists(audio_path):
                primary_audio = AudioFileClip(audio_path)
                # Adjust clip duration to audio if audio is longer?
                # Usually we want the visual to match the read time.
                # If audio is longer than estimated duration, extend clip.
                if primary_audio.duration > duration:
                    clip = clip.with_duration(primary_audio.duration)
            else:
                logger.warning(f"  Missing Audio: {beat_id}")
            
            # 3. COMPOSITE AUDIO (Primary + BGM + SFX)
            audio_layers = []
            if primary_audio:
                audio_layers.append(primary_audio)
            
            # BGM
            prod = beat.get("production", {})
            bgm_info = prod.get("bgm", {})
            style = bgm_info.get("style", "None")
            vol = bgm_info.get("volume", 0.1) * 1.5 # User requested boost
            
            if style and style.lower() not in ["none", "silence"]:
                key = style.replace(" ", "_").replace("/", "-").lower()
                bgm_path = os.path.join(BGM_DIR, f"{key}.wav")
                if os.path.exists(bgm_path):
                    bgm_clip = AudioFileClip(bgm_path)
                    # Loop or cut to fit beat
                    # Since BGM is usually longer, we take a subclip or loop if short
                    if bgm_clip.duration < clip.duration:
                        bgm_clip = bgm_clip.looped(duration=clip.duration)
                    else:
                        bgm_clip = bgm_clip.subclipped(0, clip.duration)
                    
                    try:
                        bgm_clip = bgm_clip.with_volume_scaled(vol)
                    except AttributeError:
                        # Fallback if with_volume_scaled doesn't exist (v2 beta vs release)
                        # v2 often uses effects differently
                        from moviepy.audio.fx import multiply_volume
                        bgm_clip = multiply_volume(bgm_clip, vol)

                    audio_layers.append(bgm_clip)
            
            # SFX
            sfx_list = prod.get("sfx", [])
            for s in sfx_list:
                name = s.get("name", "None")
                if name and name.lower() != "none":
                    key = name.replace(" ", "_").replace("/", "-").lower()
                    sfx_path = os.path.join(SFX_DIR, f"{key}.wav")
                    if os.path.exists(sfx_path):
                        sfx_clip = AudioFileClip(sfx_path)
                        # SFX Volume? Defaulting to 0.6
                        try:
                            sfx_clip = sfx_clip.with_volume_scaled(0.6)
                        except AttributeError:
                            from moviepy.audio.fx import multiply_volume
                            sfx_clip = multiply_volume(sfx_clip, 0.6)
                        
                        # Timing check
                        start_offset = 0 # Play immediately
                        timing = s.get("timing", {})
                        if "start" in timing:
                            # Relative start (0.0 - 1.0) -> seconds
                            start_offset = timing["start"] * clip.duration
                        
                        sfx_clip = sfx_clip.with_start(start_offset)
                        audio_layers.append(sfx_clip)

            # Combine Audio Layers
            if audio_layers:
                final_audio = CompositeAudioClip(audio_layers)
                # CompositeAudioClip duration extends to max end. Truncate to clip duration?
                # Narrator audio defines the rigid timing usually.
                final_audio = final_audio.with_duration(clip.duration)
                clip = clip.with_audio(final_audio)
            
            clips.append(clip)

    if not clips:
        logger.error("No clips generated!")
        return

    logger.info(f"Concatenating {len(clips)} beats...")
    final_video = concatenate_videoclips(clips, method="compose")
    
    logger.info(f"Writing video to {output_file}...")
    final_video.write_videofile(output_file, fps=24, codec='libx264', audio_codec='aac')
    logger.info("Movie Generation Complete.")

if __name__ == "__main__":
    import sys
    events = sys.argv[1] if len(sys.argv) > 1 else EVENTS_FILE
    generate_movie(events)
