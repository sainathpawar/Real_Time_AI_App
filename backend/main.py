# Fast API Code

import asyncio
import logging
import os
import uuid
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from .graph import app_graph


# Logging Setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  )

logger = logging.getLogger("realtime_ai_backend")

# Settings

class Settings(BaseSettings):
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8501"]  # Default allowed origins
    TIMEOUT_SECONDS: int = 20  # Default timeout for graph execution
    MAX_INPUT_LENGTH: int = 2000  # Max length for input messages
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)

    class Config:
        env_file = ".env"

settings = Settings()

# FastAPI App Initialization

app = FastAPI(
    title = "Real-Time AI Support Assistant API",
    version = "1.0.0"
)

# CORS Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True, #Cookies, Tokens
    allow_methods=["POST", "OPTIONS"], # Options send by borwser for preflight
    allow_headers=["*"],
)

# Models

class Query(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    request_id: str

# Healthy Check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query, request: Request):
    request_id = str(uuid.uuid4())
    logger.info(f"Received request {request_id} with message: {query.message}")

    text = query.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    if len(text) > settings.MAX_INPUT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Message exceeds maximum length of {Settings.MAX_INPUT_LENGTH} characters.")
    
    # Try Invoke the Graph
    try:
        if hasattr(app_graph, "ainvoke"):
            result = await asyncio.wait_for(
                app_graph.ainvoke({"message": text}), # ainvoke is async
                timeout=settings.TIMEOUT_SECONDS
            )
        else:
            raise HTTPException(status_code=500, detail="Graph does not support asynchronous invocation.")
        
    except asyncio.TimeoutError:
        logger.error(f"Request {request_id} timed out after {Settings.TIMEOUT_SECONDS} seconds.")
        raise HTTPException(status_code=504, detail="Processing timed out. Please try again later.")
    
    except Exception:
        logger.exception(f"Request %s failed internally", request_id)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")   
    
    # Extract Response
    response_text: Optional[str] = None

    if isinstance(result, dict):
        response_text = result.get("response")
    else:
        response_text = getattr(result, "response", None)

    if not response_text:
        logger.error("Request %s returned empty response", request_id)
        raise HTTPException(status_code=500, detail="Failed to generate response. Please try again later.")
    
    logger.info(f"Request {request_id} processed successfully.")

    return ChatResponse(
        response=response_text.strip(), 
        request_id=request_id
    )

