import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Load and preprocess data for recommendation system
    """
    
    @staticmethod
    def load_movielens_sample() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create sample MovieLens-style data
        """
        logger.info("Creating sample data...")
        
        # Sample ratings
        np.random.seed(42)
        n_users = 1000
        n_items = 500
        n_ratings = 10000
        
        ratings_data = {
            'user_id': np.random.randint(1, n_users+1, n_ratings),
            'item_id': np.random.randint(1, n_items+1, n_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings, 
                                      p=[0.05, 0.1, 0.2, 0.35, 0.3]),
            'timestamp': pd.date_range('2023-01-01', periods=n_ratings, freq='1H')
        }
        ratings_df = pd.DataFrame(ratings_data)
        
        # Sample items
        genres_list = ['Action', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi', 
                      'Romance', 'Horror', 'Documentary']
        
        items_data = []
        for i in range(1, n_items+1):
            items_data.append({
                'item_id': i,
                'title': f'Movie {i}',
                'genres': ' '.join(np.random.choice(genres_list, 
                                  size=np.random.randint(1, 4), replace=False)),
                'year': np.random.randint(2000, 2024),
                'description': f'Description for movie {i}'
            })
        
        items_df = pd.DataFrame(items_data)
        
        logger.info(f"Created {len(ratings_df)} ratings and {len(items_df)} items")
        
        return ratings_df, items_df
    
    @staticmethod
    def create_user_item_matrix(ratings_df: pd.DataFrame) -> Tuple[csr_matrix, List[int], List[int]]:
        """
        Create sparse user-item matrix
        """
        # Get unique users and items
        user_ids = sorted(ratings_df['user_id'].unique())
        item_ids = sorted(ratings_df['item_id'].unique())
        
        # Create mappings
        user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        item_map = {iid: idx for idx, iid in enumerate(item_ids)}
        
        # Create matrix
        rows = ratings_df['user_id'].map(user_map)
        cols = ratings_df['item_id'].map(item_map)
        data = ratings_df['rating'].values
        
        matrix = csr_matrix((data, (rows, cols)), 
                           shape=(len(user_ids), len(item_ids)))
        
        logger.info(f"Created matrix: {matrix.shape}, Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")
        
        return matrix, user_ids, item_ids
    
    @staticmethod
    def train_test_split(ratings_df: pd.DataFrame, 
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        """
        # Sort by timestamp
        ratings_df = ratings_df.sort_values('timestamp')
        
        # Split
        split_idx = int(len(ratings_df) * (1 - test_size))
        train_df = ratings_df.iloc[:split_idx]
        test_df = ratings_df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df