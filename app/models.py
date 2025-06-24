from pydantic import BaseModel, Field
from typing import List, Optional

class YouTubeVideoRequest(BaseModel):
    video_url: str = Field(description="YouTube video URL")
    transcribe_language: Optional[str] = Field(default=None, description="Language for transcription (auto-detect if None)")

class VideoSentimentResponse(BaseModel):
    video_url: Optional[str]
    transcription: str
    transcription_confidence: float
    sentiment_scores: dict
    predicted_sentiment: str
    extreme_scores: List[dict] = Field(description="5 most extreme sentiment scores from text segments")

class UploadVideoRequest(BaseModel):
    transcribe_language: Optional[str] = Field(default=None, description="Language for transcription") 