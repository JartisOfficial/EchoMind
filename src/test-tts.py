from TTS.api import TTS

OUTPUT_PATH = "test.wav"

# Running a multi-speaker and multi-lingual model

# List available 🐸TTS models and choose the first one
model_name = TTS.list_models()[0]
for idx, name in enumerate(TTS.list_models()):
    print(idx, name)

# Init TTS
tts = TTS("tts_models/en/jenny/jenny")
tts.tts_to_file(text="This is the sexy voice you like.", file_path=OUTPUT_PATH, emotion="Sexy", speed=1.5)


from pygame import mixer
import time
mixer.init()

mixer.music.load(OUTPUT_PATH)
mixer.music.play()

while mixer.music.get_busy():  # wait for music to finish playing
    time.sleep(1)

mixer.music.unload()
mixer.quit()