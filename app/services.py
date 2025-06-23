from fastapi import HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import numpy as np
import re
import subprocess
import whisper
import yt_dlp
import librosa
from typing import Optional

# --- Model Loading ---
# This is a heavy operation, so it's done once when the module is loaded.
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
whisper_model = whisper.load_model("base")

# --- Service Functions ---

def extract_instagram_video_url(instagram_url: str) -> str:
    """Extract the actual video URL from Instagram post"""
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'noplaylist': True,
            'extract_flat': False,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(instagram_url, download=False)
            if 'url' in info:
                return info['url']
            elif 'entries' in info and len(info['entries']) > 0:
                return info['entries'][0]['url']
            else:
                raise ValueError("Could not extract video URL")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract video URL: {str(e)}")

def download_instagram_video(instagram_url: str, output_path: str) -> str:
    """Download Instagram video using yt-dlp"""
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([instagram_url])
        return output_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

def extract_audio_from_video(video_path: str, audio_path: str) -> str:
    """Extract audio from video file using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ], check=True, capture_output=True, text=True)
        return audio_path
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found. Please ensure ffmpeg is installed and in your PATH."
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract audio: {e.stderr}")

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> dict:
    """Transcribe audio using Whisper"""
    try:
        result = whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )
        return {
            'text': result['text'].strip(),
            'language': result['language'],
            'segments': result.get('segments', []),
            'confidence': np.mean([seg.get('no_speech_prob', 0.5) for seg in result.get('segments', [{'no_speech_prob': 0.5}])])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

def analyze_text_sentiment(text: str) -> dict:
    """Analyze sentiment of transcribed text"""
    try:
        if not text or not text.strip():
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}

        cleaned_text = text.strip()
        max_length = 500
        if len(cleaned_text) > max_length:
            sentences = re.split(r'[.!?]+', cleaned_text)
            sentence_scores = []
            for sentence in sentences:
                if sentence.strip():
                    encoded_text = tokenizer(sentence.strip(), return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        output = model(**encoded_text)
                    scores = output[0][0].detach().numpy()
                    scores = softmax(scores)
                    sentence_scores.append({
                        'negative': float(scores[0]),
                        'neutral': float(scores[1]),
                        'positive': float(scores[2])
                    })
            if sentence_scores:
                avg_scores = {
                    'negative': float(np.mean([s['negative'] for s in sentence_scores])),
                    'neutral': float(np.mean([s['neutral'] for s in sentence_scores])),
                    'positive': float(np.mean([s['positive'] for s in sentence_scores]))
                }
                return avg_scores

        encoded_text = tokenizer(cleaned_text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2])
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33} 