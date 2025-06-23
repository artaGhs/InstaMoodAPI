from pydantic import BaseModel, Field
from typing import List, Optional

class InstagramVideoRequest(BaseModel):
    video_url: str = Field(description="Instagram video/reel URL")
    transcribe_language: Optional[str] = Field(default=None, description="Language for transcription (auto-detect if None)")

class VideoSentimentResponse(BaseModel):
    video_url: Optional[str]
    transcription: str
    transcription_confidence: float
    sentiment_scores: dict
    predicted_sentiment: str
    text_segments: List[dict] = Field(description="Timestamped transcription segments")

class UploadVideoRequest(BaseModel):
    transcribe_language: Optional[str] = Field(default=None, description="Language for transcription") 