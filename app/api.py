from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import os
import tempfile
import shutil
import numpy as np
import asyncio

from app.models import YouTubeVideoRequest, VideoSentimentResponse
from app.services import (
    extract_youtube_transcript,
    extract_audio_from_video,
    transcribe_audio,
    analyze_text_sentiment,
    get_extreme_sentiment_scores,
    MODEL
)

router = APIRouter(
    prefix="/youtube",
    tags=["youtube-video-sentiment"]
)

@router.get("/")
async def root():
    return {
        "message": "YouTube Video Sentiment Analysis API is running!",
        "description": "Analyze sentiment of YouTube videos by extracting their transcripts",
        "endpoints": {
            "/analyze_video_url": "Analyze YouTube video from URL",
            "/analyze_uploaded_video": "Analyze uploaded video file",
        },
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "sentiment_model": MODEL,
        "transcription_model": "whisper-base",
    }

@router.post("/analyze_video_url", response_model=VideoSentimentResponse)
async def analyze_youtube_video(request: YouTubeVideoRequest):
    """Analyze sentiment of YouTube video by extracting its transcript"""
    try:
        # Run blocking I/O and CPU-bound operations in a separate thread
        transcription_result = await asyncio.to_thread(extract_youtube_transcript, request.video_url, request.transcribe_language)

        if not transcription_result['text']:
            raise HTTPException(status_code=400, detail="No transcript available for the video")

        sentiment_scores = await asyncio.to_thread(analyze_text_sentiment, transcription_result['text'])
        
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_labels[np.argmax(list(sentiment_scores.values()))]

        # Get extreme sentiment scores from segments
        extreme_scores = await asyncio.to_thread(get_extreme_sentiment_scores, transcription_result.get('segments', []))

        return VideoSentimentResponse(
            video_url=request.video_url,
            transcription=transcription_result['text'],
            transcription_confidence=transcription_result['confidence'],
            sentiment_scores=sentiment_scores,
            predicted_sentiment=predicted_sentiment,
            extreme_scores=extreme_scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@router.post("/analyze_uploaded_video", response_model=VideoSentimentResponse)
async def analyze_uploaded_video(
    file: UploadFile = File(...),
    transcribe_language: Optional[str] = Form(None)
):
    """Analyze sentiment of uploaded video file"""
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Please upload a video file")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            video_path = os.path.join(temp_dir, f"uploaded_video_{file.filename}")
            audio_path = os.path.join(temp_dir, "audio.wav")

            # Save uploaded file
            with open(video_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Run blocking I/O and CPU-bound operations in a separate thread
            await asyncio.to_thread(extract_audio_from_video, video_path, audio_path)
            transcription_result = await asyncio.to_thread(transcribe_audio, audio_path, transcribe_language)

            if not transcription_result['text']:
                raise HTTPException(status_code=400, detail="No speech detected in the video")

            sentiment_scores = await asyncio.to_thread(analyze_text_sentiment, transcription_result['text'])
            
            sentiment_labels = ['negative', 'neutral', 'positive']
            predicted_sentiment = sentiment_labels[np.argmax(list(sentiment_scores.values()))]

            # Get extreme sentiment scores from segments
            extreme_scores = await asyncio.to_thread(get_extreme_sentiment_scores, transcription_result.get('segments', []))

            return VideoSentimentResponse(
                video_url=f"uploaded:{file.filename}",
                transcription=transcription_result['text'],
                transcription_confidence=1.0 - transcription_result['confidence'],
                sentiment_scores=sentiment_scores,
                predicted_sentiment=predicted_sentiment,
                extreme_scores=extreme_scores
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing uploaded video: {str(e)}") 