import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_recommendations():
    """Test recommendations endpoint"""
    response = client.get("/api/v1/recommend/123?n=10")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 10

def test_submit_feedback():
    """Test feedback endpoint"""
    feedback = {
        "user_id": 123,
        "item_id": 456,
        "rating": 4.5,
        "interaction_type": "watch"
    }
    response = client.post("/api/v1/feedback", json=feedback)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_trending_items():
    """Test trending endpoint"""
    response = client.get("/api/v1/trending?n=20")
    assert response.status_code == 200
    data = response.json()
    assert "trending_items" in data
    assert len(data["trending_items"]) == 20

def test_similar_items():
    """Test similar items endpoint"""
    response = client.get("/api/v1/similar/100?n=10")
    assert response.status_code == 200
    data = response.json()
    assert "similar_items" in data

def test_batch_recommendations():
    """Test batch recommendations"""
    user_ids = [1, 2, 3, 4, 5]
    response = client.post("/api/v1/batch-recommend", json=user_ids)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == len(user_ids)

def test_user_profile():
    """Test user profile endpoint"""
    response = client.get("/api/v1/user/123/profile")
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert "preferences" in data

def test_system_stats():
    """Test stats endpoint"""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_users" in data
    assert "total_items" in data