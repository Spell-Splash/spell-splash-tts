# server.py
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import random
import uuid
from kokoro import KPipeline
import requests
import re
from difflib import SequenceMatcher
import os
from fastapi import HTTPException
import numpy as np
from scipy.io.wavfile import write as write_wav

app = FastAPI()
pipeline = KPipeline(lang_code='a')

class ChoiceResponse(BaseModel):
    correct: str
    choices: list[str]
    audio_file: str

def get_phonetic_similarity(word1, word2):
    word1, word2 = word1.lower(), word2.lower()

    def simplify_phonetics(word):
        phonetic = word
        replacements = [
            (r'ph', 'f'), (r'th', 't'), (r'sh', 's'), (r'ch', 'k'),
            (r'wh', 'w'), (r'qu', 'kw'), (r'gh', 'g'), (r'ck', 'k'),
            (r'kn', 'n'), (r'mb', 'm'), (r'ng', 'n'), (r'wr', 'r'),
            (r'mn', 'm'), (r'ps', 's'), (r'gn', 'n'), (r'rh', 'r'),
            (r'dg', 'j'), (r'ce', 's'), (r'ci', 's'), (r'cy', 's')
        ]
        for old, new in replacements:
            phonetic = re.sub(old, new, phonetic)
        phonetic = re.sub(r'([a-z])\1+', r'\1', phonetic)
        phonetic = re.sub(r'[aeiou]+', 'a', phonetic)
        return phonetic

    phonetic1 = simplify_phonetics(word1)
    phonetic2 = simplify_phonetics(word2)
    similarity = SequenceMatcher(None, phonetic1, phonetic2).ratio()
    return similarity

def generate_similar_words(word, word_list, n=2):
    if not word_list or len(word_list) < n:
        return ["no_similar_words_found"] * n
    similarities = [(w, get_phonetic_similarity(word, w)) for w in word_list if w != word]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in similarities[:n]]

@app.get("/generate", response_model=ChoiceResponse)
def generate_quiz():
    try:
        # Fetch 10 random words from the external API
        word_list = requests.get('https://random-word-api.vercel.app/api?words=10').json()
        target_word = random.choice(word_list)
        similar_words = generate_similar_words(target_word, word_list, n=3)
    except Exception as e:
        # Fallback in case of any API/network issue
        fallback_words = ["banana", "chocolate", "pineapple", "university", "information",
                          "creativity", "sensation", "tremendous", "technology", "generation"]
        target_word = random.choice(fallback_words)
        word_list = random.sample(fallback_words, len(fallback_words))
        similar_words = generate_similar_words(target_word, word_list, n=3)

    audio_filename = "current_question.wav"

    try:
        for _, _, audio_tensor in pipeline(target_word, voice='af_heart'):
            audio_np = audio_tensor.detach().cpu().numpy()
            audio_np = (audio_np * 32767).astype(np.int16)  # Convert to 16-bit PCM
            write_wav(audio_filename, 22000, audio_np)
            break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    choices = [target_word] + similar_words
    random.shuffle(choices)

    return ChoiceResponse(
        correct=target_word,
        choices=choices,
        audio_file=audio_filename
    )

@app.get("/audio/{file_name}")
def get_audio(file_name: str):
    if not os.path.isfile(file_name):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_name, media_type="audio/wav")