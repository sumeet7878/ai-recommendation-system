import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for recommendation system
    """
    
    @staticmethod
    def create_user_features(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level features
        """
        logger.info("Creating user features...")
        
        user_features = ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'item_id': 'nunique',
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_features.columns = ['user_id', 'rating_count', 'avg_rating', 
                                'rating_std', 'unique_items', 'first_rating', 'last_rating']
        
        # Calculate user activity span
        user_features['activity_span_days'] = (
            user_features['last_rating'] - user_features['first_rating']
        ).dt.days
        
        # User engagement score
        user_features['engagement_score'] = (
            user_features['rating_count'] * user_features['avg_rating']
        )
        
        logger.info(f"Created {len(user_features)} user feature vectors")
        
        return user_features
    
    @staticmethod
    def create_item_features(ratings_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create item-level features
        """
        logger.info("Creating item features...")
        
        # Rating statistics
        item_stats = ratings_df.groupby('item_id').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        }).reset_index()
        
        item_stats.columns = ['item_id', 'rating_count', 'avg_rating', 
                             'rating_std', 'unique_users']
        
        # Merge with item metadata
        item_features = items_df.merge(item_stats, on='item_id', how='left')
        
        # Fill missing values
        item_features.fillna({
            'rating_count': 0,
            'avg_rating': 0,
            'rating_std': 0,
            'unique_users': 0
        }, inplace=True)
        
        # Popularity score
        item_features['popularity_score'] = (
            item_features['rating_count'] * item_features['avg_rating']
        )
        
        logger.info(f"Created {len(item_features)} item feature vectors")
        
        return item_features
    
    @staticmethod
    def create_interaction_features(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-item interaction features
        """
        logger.info("Creating interaction features...")
        
        interactions = ratings_df.copy()
        
        # Time-based features
        interactions['hour'] = interactions['timestamp'].dt.hour
        interactions['day_of_week'] = interactions['timestamp'].dt.dayofweek
        interactions['is_weekend'] = interactions['day_of_week'].isin([5, 6]).astype(int)
        
        # Rating recency
        interactions['days_since_rating'] = (
            interactions['timestamp'].max() - interactions['timestamp']
        ).dt.days
        
        # Normalized rating (z-score per user)
        user_means = interactions.groupby('user_id')['rating'].transform('mean')
        user_stds = interactions.groupby('user_id')['rating'].transform('std')
        interactions['rating_normalized'] = (
            (interactions['rating'] - user_means) / (user_stds + 1e-8)
        )
        
        logger.info(f"Created interaction features for {len(interactions)} interactions")
        
        return interactions