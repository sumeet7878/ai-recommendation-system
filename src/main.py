from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
import redis.asyncio as redis
from typing import Optional
import os

from src.api.endpoints import router
from src.utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
redis_client: Optional[redis.Redis] = None
metrics_collector = MetricsCollector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global redis_client
    
    # Startup
    logger.info("🚀 Starting Recommendation Engine...")
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        redis_client = None
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")
    if redis_client:
        await redis_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="🎬 AI-Powered Recommendation Engine",
    description="Advanced ML-powered recommendation system with real-time predictions",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"⚠️ Static files mount warning: {e}")

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Recommendations"])


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page - serve static HTML"""
    try:
        with open("templates/dashboard.html", "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        return f"<h1>Error loading dashboard: {str(e)}</h1>"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_status = "connected" if redis_client else "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": redis_status,
        "version": "1.0.0"
    }


@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    return await metrics_collector.get_metrics()


def format_number(value):
    """Make numbers look pretty like: 1,000,000"""
    if isinstance(value, (int, float)):
        return "{:,}".format(int(value))
    return value

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )