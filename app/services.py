from fastapi import HTTPException, FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import numpy as np
import re
import os
from typing import Optional, List
from contextlib import asynccontextmanager
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Use local model paths for offline deployment
SENTIMENT_MODEL_PATH = "./models/sentiment-model"

# YouTube API configuration
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Set this in Azure App Service Configuration

# Global variables to store models (will be loaded during lifespan)
tokenizer = None
sentiment_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, sentiment_model
    print("Loading sentiment model from local files...")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
    print("Sentiment model loaded.")
    yield
    print("Cleaning up models...")
    tokenizer = None
    sentiment_model = None

# --- Service Functions ---

def extract_youtube_transcript(youtube_url: str, language: Optional[str] = None) -> dict:
    """Extract transcript from YouTube video using Google YouTube API"""
    
    # Extract video ID from URL
    video_id = extract_video_id_from_url(youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Check if API key is configured
    if not YOUTUBE_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="YouTube API key not configured. Please set YOUTUBE_API_KEY environment variable."
        )
    
    # Use Google YouTube API
    return extract_transcript_with_youtube_api(video_id, language)


def extract_video_id_from_url(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    import re
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_transcript_with_youtube_api(video_id: str, language: Optional[str] = None) -> dict:
    """Extract transcript using official Google YouTube API"""
    try:
        # Build the YouTube service
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
        
        # Get list of available captions
        caption_request = youtube.captions().list(
            part="snippet",
            videoId=video_id
        )
        caption_response = caption_request.execute()
        
        # Find the appropriate caption track
        caption_id = None
        target_language = language or "en"
        
        # First, try to find exact language match
        for item in caption_response.get("items", []):
            if item["snippet"]["language"] == target_language:
                caption_id = item["id"]
                break
        
        # If no exact match and no specific language requested, try English variants
        if not caption_id and language is None:
            for item in caption_response.get("items", []):
                lang = item["snippet"]["language"]
                if lang.startswith("en"):  # en, en-US, en-GB, etc.
                    caption_id = item["id"]
                    break
        
        # If still no match, take the first available caption
        if not caption_id and caption_response.get("items"):
            caption_id = caption_response["items"][0]["id"]
            print(f"Using first available caption: {caption_response['items'][0]['snippet']['language']}")
        
        if not caption_id:
            raise ValueError(f"No captions found for video {video_id}")
        
        # Download the transcript
        transcript_request = youtube.captions().download(
            id=caption_id,
            tfmt='srt'  # SubRip format
        )
        
        transcript_content = transcript_request.execute().decode('utf-8')
        
        # Parse SRT format and return structured data
        return parse_srt_transcript(transcript_content, target_language)
        
    except HttpError as e:
        if e.resp.status == 403:
            raise HTTPException(status_code=403, detail="YouTube API quota exceeded or invalid API key")
        elif e.resp.status == 404:
            raise HTTPException(status_code=404, detail="Video not found or captions not available")
        else:
            raise HTTPException(status_code=500, detail=f"YouTube API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract transcript via YouTube API: {str(e)}")


def parse_srt_transcript(srt_content: str, language: str) -> dict:
    """Parse SRT subtitle format into structured transcript data"""
    try:
        segments = []
        text_parts = []
        
        # Split SRT into blocks
        blocks = srt_content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Parse timestamp line (format: 00:00:01,000 --> 00:00:04,000)
                timestamp_line = lines[1]
                if ' --> ' in timestamp_line:
                    start_str, end_str = timestamp_line.split(' --> ')
                    start_time = parse_srt_timestamp(start_str)
                    end_time = parse_srt_timestamp(end_str)
                    
                    # Join text lines (lines 2 onwards)
                    text = ' '.join(lines[2:]).strip()
                    # Clean up HTML tags and formatting
                    text = re.sub(r'<[^>]+>', '', text)
                    text = text.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                    
                    if text:
                        text_parts.append(text)
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text,
                            'confidence': 0.95  # High confidence for official captions
                        })
        
        full_text = ' '.join(text_parts)
        
        return {
            'text': full_text.strip(),
            'language': language,
            'segments': segments,
            'confidence': 0.95 if segments else 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse SRT transcript: {str(e)}")


def parse_srt_timestamp(timestamp_str: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""
    try:
        # Format: 00:00:01,000
        time_part, ms_part = timestamp_str.split(',')
        hours, minutes, seconds = map(int, time_part.split(':'))
        milliseconds = int(ms_part)
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds
    except:
        return 0.0






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
                        output = sentiment_model(**encoded_text)
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
            output = sentiment_model(**encoded_text)
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