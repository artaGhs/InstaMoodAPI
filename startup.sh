#!/bin/bash

echo "🚀 Starting InstaMood API deployment..."

# Download models if they don't exist
echo "📦 Checking for models..."
if [ ! -f "./models/sentiment-model/config.json" ]; then
    echo "📥 Downloading models..."
    python scripts/download_models.py
    if [ $? -ne 0 ]; then
        echo "❌ Model download failed!"
        exit 1
    fi
else
    echo "✅ Models already available"
fi

# Start the FastAPI application
echo "🌟 Starting FastAPI server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 