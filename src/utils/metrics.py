import numpy as np
from typing import List, Dict, Set
import time
from datetime import datetime
import asyncio

class MetricsCollector:
    """
    Collect and track system metrics
    """
    
    def __init__(self):
        self.request_count = 0
        self.total_latency = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        
    async def record_request(self, latency_ms: float):
        """Record a request"""
        self.request_count += 1
        self.total_latency += latency_ms
        
    async def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
        
    async def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
        
    async def get_metrics(self) -> Dict:
        """Get current metrics"""
        uptime = time.time() - self.start_time
        avg_latency = (self.total_latency / self.request_count 
                      if self.request_count > 0 else 0)
        
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100 
                         if cache_total > 0 else 0)
        
        return {
            "requests_total": self.request_count,
            "avg_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "uptime_seconds": round(uptime, 2),
            "timestamp": datetime.now().isoformat()
        }

def calculate_precision_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    """Calculate Precision@K"""
    if k == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    hits = len([p for p in predicted_k if p in actual])
    
    return hits / k

def calculate_recall_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    """Calculate Recall@K"""
    if len(actual) == 0:
        return 0.0
    
    predicted_k = predicted[:k]
    hits = len([p for p in predicted_k if p in actual])
    
    return hits / len(actual)

def calculate_ndcg_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    """Calculate NDCG@K"""
    predicted_k = predicted[:k]
    
    # DCG
    dcg = sum([1 / np.log2(i + 2) if pred in actual else 0 
               for i, pred in enumerate(predicted_k)])
    
    # IDCG
    ideal_k = min(k, len(actual))
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_k)])
    
    return dcg / idcg if idcg > 0 else 0.0