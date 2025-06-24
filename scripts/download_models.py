#!/usr/bin/env python3
"""
Azure deployment script to download models.
This runs during the build process in Azure, not locally.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_sentiment_model():
    """Download sentiment model for Azure deployment."""
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    MODEL_DIR = "./models/sentiment-model"
    
    print(f"📥 Downloading sentiment model: {MODEL_NAME}")
    print(f"📁 Saving to: {MODEL_DIR}")
    
    # Create directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        # Download tokenizer and model
        print("🔄 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print("🔄 Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Save locally
        print("💾 Saving tokenizer...")
        tokenizer.save_pretrained(MODEL_DIR)
        
        print("💾 Saving model...")
        model.save_pretrained(MODEL_DIR)
        
        print("✅ Sentiment model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def main():
    """Main function for Azure deployment."""
    print("🚀 Starting model download for Azure deployment...")
    
    # Check if models already exist
    model_path = "./models/sentiment-model/config.json"
    if os.path.exists(model_path):
        print("✅ Models already exist, skipping download.")
        return 0
    
    # Download models
    success = download_sentiment_model()
    
    if success:
        print("🎉 All models ready for deployment!")
        return 0
    else:
        print("💥 Model download failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 