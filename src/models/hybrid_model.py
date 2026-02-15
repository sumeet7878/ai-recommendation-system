import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.content_based import ContentBasedFiltering

logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    Hybrid recommendation system combining multiple approaches
    """
    
    def __init__(self, cf_weight: float = 0.6, content_weight: float = 0.4):
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        
        self.cf_model = CollaborativeFiltering()
        self.content_model = ContentBasedFiltering()
        
        self.is_trained = False
        
    def train(self, user_item_matrix, user_ids: List[int], 
              item_ids: List[int], items_data: List[Dict]):
        """
        Train both models
        """
        logger.info("Training hybrid model...")
        
        # Train collaborative filtering
        self.cf_model.fit(user_item_matrix, user_ids, item_ids)
        
        # Train content-based
        self.content_model.fit(items_data)
        
        self.is_trained = True
        logger.info("Hybrid model training completed!")
        
    def recommend(self, user_id: int, user_interactions: List[Tuple[int, float]] = None,
                  n: int = 10, diversity_weight: float = 0.2) -> List[Dict]:
        """
        Get hybrid recommendations
        """
        if not self.is_trained:
            logger.warning("Model not trained yet!")
            return []
        
        # Get CF recommendations
        cf_recs = self.cf_model.predict(user_id, n=n*2)
        cf_dict = {item_id: score for item_id, score in cf_recs}
        
        # Get content-based recommendations
        cb_recs = []
        if user_interactions:
            user_profile = self.content_model.get_user_profile(user_interactions)
            cb_recs = self.content_model.recommend(user_profile, n=n*2)
        cb_dict = {item_id: score for item_id, score in cb_recs}
        
        # Combine scores
        all_items = set(cf_dict.keys()) | set(cb_dict.keys())
        hybrid_scores = {}
        
        for item_id in all_items:
            cf_score = cf_dict.get(item_id, 0)
            cb_score = cb_dict.get(item_id, 0)
            
            # Weighted combination
            hybrid_score = (self.cf_weight * cf_score + 
                          self.content_weight * cb_score)
            
            hybrid_scores[item_id] = hybrid_score
        
        # Sort by score
        sorted_items = sorted(hybrid_scores.items(), 
                            key=lambda x: x[1], reverse=True)
        
        # Apply diversity
        final_recommendations = self._apply_diversity(
            sorted_items[:n*2], n, diversity_weight
        )
        
        # Format output
        recommendations = []
        for item_id, score in final_recommendations[:n]:
            item_data = self.content_model.item_metadata.get(item_id, {})
            recommendations.append({
                'item_id': item_id,
                'score': round(score, 4),
                'title': item_data.get('title', f'Item {item_id}'),
                'genres': item_data.get('genres', ''),
                'method': 'hybrid'
            })
        
        return recommendations
    
    def _apply_diversity(self, items: List[Tuple[int, float]], 
                        n: int, diversity_weight: float) -> List[Tuple[int, float]]:
        """
        Apply diversity to recommendations
        """
        if diversity_weight == 0 or len(items) <= n:
            return items[:n]
        
        selected = [items[0]]  # Start with top item
        remaining = items[1:]
        
        while len(selected) < n and remaining:
            max_score = -float('inf')
            best_idx = 0
            
            for idx, (item_id, score) in enumerate(remaining):
                # Calculate diversity bonus
                diversity_bonus = self._calculate_diversity(
                    item_id, [s[0] for s in selected]
                )
                
                # Combined score
                combined_score = score + diversity_weight * diversity_bonus
                
                if combined_score > max_score:
                    max_score = combined_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _calculate_diversity(self, item_id: int, 
                            selected_items: List[int]) -> float:
        """
        Calculate diversity bonus for an item
        """
        if not selected_items:
            return 1.0
        
        # Get item genres
        item_genres = set(self.content_model.item_metadata.get(
            item_id, {}
        ).get('genres', '').split())
        
        # Calculate average dissimilarity
        dissimilarities = []
        for selected_id in selected_items:
            selected_genres = set(self.content_model.item_metadata.get(
                selected_id, {}
            ).get('genres', '').split())
            
            if not item_genres or not selected_genres:
                dissimilarities.append(0.5)
            else:
                # Jaccard distance
                intersection = len(item_genres & selected_genres)
                union = len(item_genres | selected_genres)
                dissimilarity = 1 - (intersection / union if union > 0 else 0)
                dissimilarities.append(dissimilarity)
        
        return np.mean(dissimilarities) if dissimilarities else 0.5
    
    def save(self, cf_path: str, content_path: str):
        """Save both models"""
        self.cf_model.save(cf_path)
        self.content_model.save(content_path)
        logger.info("Hybrid model saved!")
    
    def load(self, cf_path: str, content_path: str):
        """Load both models"""
        self.cf_model.load(cf_path)
        self.content_model.load(content_path)
        self.is_trained = True
        logger.info("Hybrid model loaded!")