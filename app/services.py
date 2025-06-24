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
from typing import Optional, List

# --- Model Loading ---
# This is a heavy operation, so it's done once when the module is loaded.
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
whisper_model = whisper.load_model("base")

# --- Service Functions ---

def extract_youtube_transcript(youtube_url: str, language: Optional[str] = None) -> dict:
    """Extract transcript from YouTube video using yt-dlp"""
    try:
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [language] if language else ['en'],
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            # Try to get automatic captions first
            if 'automatic_captions' in info and info['automatic_captions']:
                for lang_code, captions in info['automatic_captions'].items():
                    if language is None or lang_code == language:
                        for caption in captions:
                            if caption['ext'] == 'json3':
                                # Download the caption file
                                caption_url = caption['url']
                                import requests
                                response = requests.get(caption_url)
                                if response.status_code == 200:
                                    caption_data = response.json()
                                    return parse_youtube_caption_json(caption_data)
            
            # Try manual captions if automatic captions not available
            if 'subtitles' in info and info['subtitles']:
                for lang_code, captions in info['subtitles'].items():
                    if language is None or lang_code == language:
                        for caption in captions:
                            if caption['ext'] == 'json3':
                                caption_url = caption['url']
                                import requests
                                response = requests.get(caption_url)
                                if response.status_code == 200:
                                    caption_data = response.json()
                                    return parse_youtube_caption_json(caption_data)
            
            # If no captions available, fall back to audio extraction and transcription
            return extract_audio_and_transcribe(youtube_url, language)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract transcript: {str(e)}")

def parse_youtube_caption_json(caption_data: dict) -> dict:
    """Parse YouTube caption JSON format"""
    try:
        text_parts = []
        segments = []
        
        if 'events' in caption_data:
            for event in caption_data['events']:
                if 'segs' in event:
                    segment_text = ""
                    for seg in event['segs']:
                        if 'utf8' in seg:
                            segment_text += seg['utf8']
                    
                    if segment_text.strip():
                        start_time = event.get('tStartMs', 0) / 1000.0
                        duration = event.get('dDurationMs', 0) / 1000.0
                        end_time = start_time + duration
                        
                        text_parts.append(segment_text)
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'text': segment_text.strip(),
                            'confidence': 0.9  # High confidence for manual captions
                        })
        
        full_text = ' '.join(text_parts)
        
        return {
            'text': full_text.strip(),
            'language': 'en',  # Default, could be extracted from caption data
            'segments': segments,
            'confidence': 0.9 if segments else 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse caption data: {str(e)}")

def extract_audio_and_transcribe(youtube_url: str, language: Optional[str] = None) -> dict:
    """Fallback: Download video, extract audio, and transcribe"""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "video.mp4")
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Download video
        download_youtube_video(youtube_url, video_path)
        
        # Extract audio
        extract_audio_from_video(video_path, audio_path)
        
        # Transcribe audio
        return transcribe_audio(audio_path, language)

def download_youtube_video(youtube_url: str, output_path: str) -> str:
    """Download YouTube video using yt-dlp"""
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
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

def get_extreme_sentiment_scores(segments: List[dict]) -> List[dict]:
    """Get the 5 most extreme sentiment scores from text segments (very positive OR very negative)"""
    try:
        if not segments:
            return []
        
        segment_scores = []
        for segment in segments:
            text = segment.get('text', '').strip()
            if text:
                # Analyze sentiment for this segment
                scores = analyze_text_sentiment(text)
                
                # Find the most extreme sentiment (positive or negative)
                positive_score = scores['positive']
                negative_score = scores['negative']
                
                # Determine which is more extreme and calculate extremity
                if positive_score > negative_score:
                    extremity = positive_score
                    dominant_sentiment = 'positive'
                else:
                    extremity = negative_score
                    dominant_sentiment = 'negative'
                
                segment_scores.append({
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'text': text,
                    'sentiment_scores': scores,
                    'extremity': extremity,
                    'dominant_sentiment': dominant_sentiment
                })
        
        # Sort by extremity (most extreme first) and take top 5
        segment_scores.sort(key=lambda x: x['extremity'], reverse=True)
        extreme_scores = segment_scores[:5]
        
        # Format for response
        return [
            {
                'start': score['start'],
                'end': score['end'],
                'text': score['text'],
                'sentiment_scores': score['sentiment_scores'],
                'dominant_sentiment': score['dominant_sentiment'],
                'extremity_score': score['extremity']
            }
            for score in extreme_scores
        ]
    except Exception as e:
        print(f"Error getting extreme scores: {str(e)}")
        return [] 