import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ContentBasedFiltering:
    """
    Content-based recommendations using item features
    """
    
    def __init__(self, n_components: int = 50, similarity_metric: str = "cosine"):
        self.n_components = n_components
        self.similarity_metric = similarity_metric
        
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.svd = TruncatedSVD(n_components=n_components)
        
        self.item_features = None
        self.item_ids = None
        self.item_metadata = {}
        
    def fit(self, items_data: List[Dict]):
        """
        Train the content-based model
        
        items_data: List of dicts with 'item_id', 'title', 'genres', 'description'
        """
        logger.info(f"Training content-based model with {len(items_data)} items")
        
        self.item_ids = [item['item_id'] for item in items_data]
        
        # Create text features
        texts = []
        for item in items_data:
            text = f"{item.get('title', '')} {item.get('genres', '')} {item.get('description', '')}"
            texts.append(text)
            self.item_metadata[item['item_id']] = item
        
        # TF-IDF vectorization
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Dimensionality reduction
        self.item_features = self.svd.fit_transform(tfidf_matrix)
        
        logger.info(f"Content-based model trained! Feature shape: {self.item_features.shape}")
        
    def get_user_profile(self, user_interactions: List[Tuple[int, float]]) -> np.ndarray:
        """
        Build user profile from their interactions
        
        user_interactions: List of (item_id, rating) tuples
        """
        profile = np.zeros(self.n_components)
        total_weight = 0
        
        for item_id, rating in user_interactions:
            if item_id in self.item_ids:
                idx = self.item_ids.index(item_id)
                weight = rating  # Use rating as weight
                profile += self.item_features[idx] * weight
                total_weight += weight
        
        if total_weight > 0:
            profile /= total_weight
        
        return profile
    
    def recommend(self, user_profile: np.ndarray, n: int = 10, 
                  exclude_items: List[int] = None) -> List[Tuple[int, float]]:
        """
        Get recommendations based on user profile
        """
        if exclude_items is None:
            exclude_items = []
        
        # Compute similarities
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get top-N items
        recommendations = []
        for idx in np.argsort(similarities)[::-1]:
            item_id = self.item_ids[idx]
            if item_id not in exclude_items:
                recommendations.append((item_id, float(similarities[idx])))
                if len(recommendations) >= n:
                    break
        
        return recommendations
    
    def get_similar_items(self, item_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find items similar to a given item
        """
        if item_id not in self.item_ids:
            return []
        
        idx = self.item_ids.index(item_id)
        item_vector = self.item_features[idx].reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(item_vector, self.item_features)[0]
        
        # Get top-N similar items (excluding itself)
        similar_items = []
        for sim_idx in np.argsort(similarities)[::-1]:
            if sim_idx != idx:
                similar_items.append((
                    self.item_ids[sim_idx],
                    float(similarities[sim_idx])
                ))
                if len(similar_items) >= n:
                    break
        
        return similar_items
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'tfidf': self.tfidf,
            'svd': self.svd,
            'item_features': self.item_features,
            'item_ids': self.item_ids,
            'item_metadata': self.item_metadata,
            'params': {
                'n_components': self.n_components,
                'similarity_metric': self.similarity_metric
            }
        }, filepath)
        logger.info(f"Content-based model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.tfidf = data['tfidf']
        self.svd = data['svd']
        self.item_features = data['item_features']
        self.item_ids = data['item_ids']
        self.item_metadata = data['item_metadata']
        
        params = data['params']
        self.n_components = params['n_components']
        self.similarity_metric = params['similarity_metric']
        
        logger.info(f"Content-based model loaded from {filepath}")