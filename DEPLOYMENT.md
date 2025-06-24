# Deployment Guide

## Local Model Setup

This application uses pre-downloaded models for faster startup and reliable offline deployment.

### Initial Setup (Development)

1. **Download Models Locally:**
   ```bash
   python scripts/download_models.py
   ```
   This downloads:
   - Sentiment analysis model: `cardiffnlp/twitter-roberta-base-sentiment`

**Note:** Models are excluded from Git due to size (>100MB). They're downloaded during Azure deployment.

2. **Models Directory Structure:**
   ```
   models/
   └── sentiment-model/
       ├── config.json
       ├── model.safetensors
       ├── tokenizer.json
       ├── vocab.json
       └── ...
   ```

### Deployment Options

#### Option 1: Include Models in Deployment (Recommended for Azure)

**Pros:**
- ✅ Faster startup (no downloads)
- ✅ Works offline
- ✅ Consistent across environments
- ✅ No dependency on external APIs

**Cons:**
- ❌ Larger deployment package (~600MB)

**Setup:**
1. Keep models in your deployment package
2. Ensure models/ directory is included in your build

#### Option 2: Download at Runtime (Original approach)

**Pros:**
- ✅ Smaller deployment package

**Cons:**
- ❌ Slower startup (downloads on first run)
- ❌ Requires internet connection
- ❌ May fail if Hugging Face is down
- ❌ Uses more bandwidth

### Azure App Service Deployment

For Azure deployment with local models:

1. **Update your startup command** to ensure models are available:
   ```bash
   # If models are included in deployment
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   
   # If you need to download models first
   python download_models.py && python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Environment Variables:**
   - `YOUTUBE_API_KEY`: Your YouTube Data API v3 key (required for reliable transcript extraction)
   - Optional: `TRANSFORMERS_OFFLINE=1` to prevent any internet calls to Hugging Face

### Docker Deployment

If using Docker, add to your Dockerfile:

```dockerfile
# Copy models into container
COPY models/ /app/models/

# Or download during build (slower but smaller image)
# RUN python download_models.py
```

### YouTube API Setup

For reliable transcript extraction, configure the YouTube Data API v3:

1. **Get API Key:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable "YouTube Data API v3"
   - Create credentials (API key)
   - Restrict the API key to YouTube Data API v3 for security

2. **Configure Environment Variable:**
   ```bash
   # Local development (.env file)
   YOUTUBE_API_KEY=your_api_key_here
   
   # Azure App Service (Configuration > Application settings)
   Name: YOUTUBE_API_KEY
   Value: your_api_key_here
   ```

3. **API Quotas:**
   - Default quota: 10,000 units/day
   - Each transcript request costs ~3-7 units
   - Monitor usage in Google Cloud Console

4. **API Requirement:**
   - ✅ **YouTube API Key Required:** Application only uses official YouTube Data API v3
   - 🚫 **No API Key:** Application will return error (API key required)
   - 🎯 **Benefits:** Reliable, fast, no bot detection issues

### Benefits of Local Models

1. **Faster Cold Starts:** No download time on startup
2. **Reliability:** No dependency on external model repositories
3. **Cost Savings:** No egress charges from repeated downloads
4. **Offline Capability:** Works without internet access
5. **Consistent Performance:** Same model versions across all environments

### Troubleshooting

**Model Issues:**
- **Model files missing:** Run `python download_models.py`
- **Permission errors:** Ensure the web app has read access to models/ directory
- **Path issues:** Verify `SENTIMENT_MODEL_PATH` and `WHISPER_MODEL_PATH` in `app/services.py`

**YouTube API Issues:**
- **403 Forbidden:** Check API key is valid and YouTube Data API v3 is enabled
- **404 Not Found:** Video doesn't exist or has no captions available
- **Quota exceeded:** Monitor usage in Google Cloud Console, consider upgrading quota
- **API key missing:** App will fall back to yt-dlp method (check logs)

**Testing API Integration:**
```python
# Test with a video ID (e.g., "dQw4w9WgXcQ")
from app.services import extract_transcript_with_youtube_api
result = extract_transcript_with_youtube_api("dQw4w9WgXcQ")
print(f"Transcript extracted: {len(result['text'])} characters")
```

**Note:** This application now only supports YouTube URL-based transcript extraction using the official YouTube Data API v3. yt-dlp fallback and file upload functionality have been removed to ensure reliability and avoid bot detection issues. 