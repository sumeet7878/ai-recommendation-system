import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """
    Matrix Factorization using Alternating Least Squares (ALS)
    """
    
    def __init__(self, n_factors: int = 100, regularization: float = 0.01, 
                 iterations: int = 15, alpha: float = 40):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        
        self.user_factors = None
        self.item_factors = None
        self.user_index = {}
        self.item_index = {}
        
    def fit(self, user_item_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        """
        Train the model using ALS
        """
        logger.info(f"Training CF model with {len(user_ids)} users and {len(item_ids)} items")
        
        n_users, n_items = user_item_matrix.shape
        
        # Create index mappings
        self.user_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_index = {iid: idx for idx, iid in enumerate(item_ids)}
        
        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # ALS iterations
        for iteration in range(self.iterations):
            # Update user factors
            self.user_factors = self._als_step(
                user_item_matrix, 
                self.item_factors, 
                self.regularization
            )
            
            # Update item factors
            self.item_factors = self._als_step(
                user_item_matrix.T, 
                self.user_factors, 
                self.regularization
            )
            
            if (iteration + 1) % 5 == 0:
                loss = self._calculate_loss(user_item_matrix)
                logger.info(f"Iteration {iteration + 1}/{self.iterations}, Loss: {loss:.4f}")
        
        logger.info("CF model training completed!")
        
    def _als_step(self, ratings: csr_matrix, factors: np.ndarray, 
                  regularization: float) -> np.ndarray:
        """
        One step of ALS
        """
        n_entities, n_factors = factors.shape
        new_factors = np.zeros_like(factors)
        
        for i in range(n_entities):
            # Get non-zero entries
            items = ratings[i].indices
            values = ratings[i].data
            
            if len(items) == 0:
                continue
            
            # Compute factors
            A = factors[items]
            b = values
            
            # Add confidence weighting
            confidence = 1 + self.alpha * np.abs(b)
            
            # Solve least squares with regularization
            AtA = A.T @ (A * confidence[:, np.newaxis])
            Atb = A.T @ (b * confidence)
            
            # Add regularization
            AtA += regularization * np.eye(n_factors)
            
            # Solve
            new_factors[i] = np.linalg.solve(AtA, Atb)
        
        return new_factors
    
    def _calculate_loss(self, user_item_matrix: csr_matrix) -> float:
        """
        Calculate reconstruction loss
        """
        predictions = self.user_factors @ self.item_factors.T
        mask = user_item_matrix.toarray() > 0
        
        loss = np.sum((user_item_matrix.toarray()[mask] - predictions[mask]) ** 2)
        loss += self.regularization * (np.sum(self.user_factors ** 2) + 
                                       np.sum(self.item_factors ** 2))
        
        return loss
    
    def predict(self, user_id: int, item_ids: Optional[List[int]] = None, 
                n: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user
        """
        if user_id not in self.user_index:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_index[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Compute scores for all items
        scores = self.item_factors @ user_vector
        
        # Get top-N items
        if item_ids is None:
            top_indices = np.argsort(scores)[-n:][::-1]
            recommendations = [
                (list(self.item_index.keys())[idx], float(scores[idx]))
                for idx in top_indices
            ]
        else:
            # Filter by specific items
            item_indices = [self.item_index[iid] for iid in item_ids 
                           if iid in self.item_index]
            filtered_scores = [(iid, float(scores[self.item_index[iid]])) 
                              for iid in item_ids if iid in self.item_index]
            recommendations = sorted(filtered_scores, key=lambda x: x[1], 
                                    reverse=True)[:n]
        
        return recommendations
    
    def get_similar_items(self, item_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items
        """
        if item_id not in self.item_index:
            return []
        
        item_idx = self.item_index[item_id]
        item_vector = self.item_factors[item_idx].reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(item_vector, self.item_factors)[0]
        
        # Get top-N similar items (excluding itself)
        top_indices = np.argsort(similarities)[-(n+1):-1][::-1]
        
        similar_items = [
            (list(self.item_index.keys())[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return similar_items
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_index': self.user_index,
            'item_index': self.item_index,
            'params': {
                'n_factors': self.n_factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.user_factors = data['user_factors']
        self.item_factors = data['item_factors']
        self.user_index = data['user_index']
        self.item_index = data['item_index']
        
        params = data['params']
        self.n_factors = params['n_factors']
        self.regularization = params['regularization']
        self.iterations = params['iterations']
        self.alpha = params['alpha']
        
        logger.info(f"Model loaded from {filepath}")