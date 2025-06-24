from fastapi import FastAPI
from app.api import router
from app.services import lifespan

app = FastAPI(
    title="YouTubeMood API",
    description="Analyze sentiment of YouTube videos by extracting their transcripts.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to YouTubeMood API"} 