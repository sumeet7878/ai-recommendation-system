from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import asyncio
from datetime import datetime
import numpy as np
from pydantic import ConfigDict

router = APIRouter()

# Pydantic models for request/response
class UserPreferences(BaseModel):
    user_id: int = Field(..., description="User ID")
    preferences: Dict[str, Any] = Field(default={}, description="User preferences")

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    num_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")
    context: Optional[Dict[str, Any]] = Field(None, description="Context information (time, device, etc)")
    filter_watched: bool = Field(True, description="Filter already watched items")

class RecommendationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    user_id: int
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    model_used: str
    timestamp: str

class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    rating: float = Field(..., ge=0, le=5)
    interaction_type: str = Field(..., description="Type: click, watch, like, etc")

# In-memory storage (replace with real DB in production)
users_db = {}
items_db = {}
interactions_db = []

# Simulated model predictions
async def get_collaborative_recommendations(user_id: int, n: int) -> List[Dict]:
    """Collaborative filtering recommendations"""
    await asyncio.sleep(0.01)  # Simulate model inference
    
    recommendations = []
    for i in range(n):
        recommendations.append({
            "item_id": 1000 + i,
            "title": f"Movie {1000 + i}",
            "score": round(np.random.uniform(0.7, 0.99), 3),
            "genres": ["Action", "Thriller"],
            "year": 2023,
            "rating": round(np.random.uniform(7.0, 9.5), 1),
            "method": "collaborative_filtering"
        })
    
    return recommendations

async def get_content_based_recommendations(user_id: int, n: int) -> List[Dict]:
    """Content-based recommendations"""
    await asyncio.sleep(0.01)
    
    recommendations = []
    for i in range(n):
        recommendations.append({
            "item_id": 2000 + i,
            "title": f"Series {2000 + i}",
            "score": round(np.random.uniform(0.65, 0.95), 3),
            "genres": ["Drama", "Comedy"],
            "year": 2024,
            "rating": round(np.random.uniform(7.5, 9.0), 1),
            "method": "content_based"
        })
    
    return recommendations

async def get_neural_recommendations(user_id: int, n: int) -> List[Dict]:
    """Neural network recommendations"""
    await asyncio.sleep(0.015)
    
    recommendations = []
    for i in range(n):
        recommendations.append({
            "item_id": 3000 + i,
            "title": f"Documentary {3000 + i}",
            "score": round(np.random.uniform(0.75, 0.98), 3),
            "genres": ["Documentary", "Nature"],
            "year": 2024,
            "rating": round(np.random.uniform(8.0, 9.5), 1),
            "method": "neural_network"
        })
    
    return recommendations

async def get_hybrid_recommendations(user_id: int, n: int, context: Optional[Dict] = None) -> List[Dict]:
    """Hybrid recommendations combining all methods"""
    start_time = time.time()
    
    # Get recommendations from all models in parallel
    cf_recs, cb_recs, nn_recs = await asyncio.gather(
        get_collaborative_recommendations(user_id, n // 3),
        get_content_based_recommendations(user_id, n // 3),
        get_neural_recommendations(user_id, n // 3)
    )
    
    # Combine and diversify
    all_recs = cf_recs + cb_recs + nn_recs
    
    # Sort by score and add diversity
    all_recs.sort(key=lambda x: x['score'], reverse=True)
    
    # Add metadata
    for rec in all_recs[:n]:
        rec['latency_ms'] = round((time.time() - start_time) * 1000, 2)
        rec['timestamp'] = datetime.now().isoformat()
    
    return all_recs[:n]


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    🎯 Get personalized recommendations for a user
    
    - **user_id**: User identifier
    - **num_recommendations**: Number of items to recommend (1-50)
    - **context**: Optional context (time, device, location)
    - **filter_watched**: Remove already watched items
    """
    start_time = time.time()
    
    try:
        # Get hybrid recommendations
        recommendations = await get_hybrid_recommendations(
            request.user_id,
            request.num_recommendations,
            request.context
        )
        
        latency = round((time.time() - start_time) * 1000, 2)
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            latency_ms=latency,
            model_used="hybrid_v1",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@router.get("/recommend/{user_id}")
async def get_recommendations_simple(
    user_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    🎯 Simple GET endpoint for recommendations
    """
    start_time = time.time()
    
    recommendations = await get_hybrid_recommendations(user_id, n)
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "latency_ms": latency,
        "count": len(recommendations)
    }


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    📊 Submit user feedback (ratings, clicks, watches)
    """
    interaction = {
        "user_id": feedback.user_id,
        "item_id": feedback.item_id,
        "rating": feedback.rating,
        "interaction_type": feedback.interaction_type,
        "timestamp": datetime.now().isoformat()
    }
    
    interactions_db.append(interaction)
    
    return {
        "status": "success",
        "message": "Feedback recorded",
        "interaction": interaction
    }


@router.get("/trending")
async def get_trending_items(
    n: int = Query(20, ge=1, le=100),
    timeframe: str = Query("24h", description="Time frame: 24h, 7d, 30d")
):
    """
    🔥 Get trending items
    """
    trending = []
    for i in range(n):
        trending.append({
            "item_id": 5000 + i,
            "title": f"Trending Item {5000 + i}",
            "trend_score": round(np.random.uniform(0.8, 1.0), 3),
            "views": np.random.randint(10000, 1000000),
            "genres": ["Thriller", "Mystery"],
            "year": 2024
        })
    
    return {
        "timeframe": timeframe,
        "trending_items": trending,
        "count": len(trending),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/similar/{item_id}")
async def get_similar_items(
    item_id: int,
    n: int = Query(10, ge=1, le=50)
):
    """
    🔗 Get items similar to a specific item
    """
    similar = []
    for i in range(n):
        similar.append({
            "item_id": 6000 + i,
            "title": f"Similar to {item_id} - Item {i+1}",
            "similarity_score": round(np.random.uniform(0.6, 0.95), 3),
            "genres": ["Drama", "Romance"],
            "year": 2023
        })
    
    return {
        "source_item_id": item_id,
        "similar_items": similar,
        "count": len(similar)
    }


@router.post("/batch-recommend")
async def batch_recommendations(
    user_ids: List[int] = Body(..., description="List of user IDs")
):
    """
    📦 Batch recommendations for multiple users
    """
    start_time = time.time()
    
    results = []
    for user_id in user_ids[:100]:  # Limit to 100 users
        recs = await get_hybrid_recommendations(user_id, 5)
        results.append({
            "user_id": user_id,
            "recommendations": recs[:5]
        })
    
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "results": results,
        "total_users": len(results),
        "total_latency_ms": latency,
        "avg_latency_per_user_ms": round(latency / len(results), 2)
    }


@router.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int):
    """
    👤 Get user profile and preferences
    """
    profile = {
        "user_id": user_id,
        "preferences": {
            "favorite_genres": ["Action", "Sci-Fi", "Thriller"],
            "preferred_languages": ["English", "Spanish"],
            "watch_time": "evening"
        },
        "stats": {
            "total_watched": np.random.randint(50, 500),
            "avg_rating": round(np.random.uniform(3.5, 4.8), 1),
            "member_since": "2022-01-15"
        }
    }
    
    return profile


@router.get("/stats")
async def get_system_stats():
    """
    📈 Get system statistics
    """
    return {
        "total_users": 150000,
        "total_items": 25000,
        "total_interactions": len(interactions_db),
        "recommendations_served_today": 2500000,
        "avg_latency_ms": 28.5,
        "cache_hit_rate": 85.5,
        "models": {
            "collaborative_filtering": "active",
            "content_based": "active",
            "neural_network": "active",
            "hybrid": "active"
        },
        "timestamp": datetime.now().isoformat()
    }