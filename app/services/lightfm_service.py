"""
LightFM recommendation model service
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .database_service import CouponDataLoader

class LightFMRecommendationSystem:
    def __init__(self, data_loader: CouponDataLoader):
        self.data_loader = data_loader
        self.dataset = None
        self.model = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.train_interactions = None
        self.test_interactions = None
        self.user_features_matrix = None
        self.item_features_matrix = None
        
    def prepare_dataset(self):
        """Prepare LightFM dataset"""
        print("\nPreparing LightFM dataset...")
        
        # Initialize dataset
        self.dataset = Dataset()
        
        # Include ALL active coupons, not just those with interactions
        interaction_users = set(self.data_loader.interactions_df['user_id'].unique())
        feature_users = set(self.data_loader.user_features_df['user_id'].unique())
        users = list(interaction_users.union(feature_users))
        
        # Use ALL active coupons from all_coupons_df
        all_active_items = set(self.data_loader.all_coupons_df['coupon_id'].unique())
        interaction_items = set(self.data_loader.interactions_df['coupon_id'].unique())
        
        # Ensure we include coupons from item_features_df as well
        feature_items = set(self.data_loader.item_features_df['coupon_id'].unique())
        all_active_items = all_active_items.union(feature_items)
        
        items = list(all_active_items)
        
        print(f"Total unique users: {len(users)}")
        print(f"Total active coupons: {len(all_active_items)}")
        print(f"Coupons with interactions: {len(interaction_items)}")
        print(f"Coupons without interactions: {len(all_active_items - interaction_items)}")
        
        # Fit dataset with ALL active coupons
        print("Fitting dataset with ALL active coupons (including zero-interaction ones)...")
        self.dataset.fit(users=users, items=items)
        
        # Verify mappings
        temp_user_feature_map, temp_user_id_map, temp_item_feature_map, temp_item_id_map = self.dataset.mapping()
        
        print(f"Clean mappings created:")
        print(f"   Users: {len(temp_user_id_map)} (expected: {len(users)})")
        print(f"   Items: {len(temp_item_id_map)} (expected: {len(items)})")
        
        print(f"Dataset prepared with {len(users)} users and {len(items)} items")
        print(f"Now ALL active coupons can be recommended!")
        
    def build_interaction_matrix(self):
        """Build interaction matrices"""
        print("\nBuilding interaction matrices...")
        
        interactions_with_weights = []
        
        for _, row in self.data_loader.interactions_df.iterrows():
            user_id = row['user_id']
            coupon_id = row['coupon_id']
            weight = row['total_weight']
            interactions_with_weights.append((user_id, coupon_id, weight))
            
            if len(interactions_with_weights) <= 5:
                print(f"Added interaction: user={user_id}, item={coupon_id}, weight={weight}")
        
        print(f"Total interactions for training: {len(interactions_with_weights)}")
        
        # Build interaction matrix
        interaction_matrix, weights_matrix = self.dataset.build_interactions(interactions_with_weights)
        
        # Get the clean mappings
        user_feature_map, self.user_id_map, item_feature_map, self.item_id_map = self.dataset.mapping()
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        
        print(f"Final LightFM mappings:")
        print(f"   Users: {len(self.user_id_map)} mapped")
        print(f"   Items: {len(self.item_id_map)} mapped")
        
        # Split into train/test
        self.train_interactions, self.test_interactions = random_train_test_split(
            interaction_matrix, test_percentage=0.2, random_state=42
        )
        
        print(f"Built interaction matrix: {interaction_matrix.shape}")
        print(f"Train interactions: {self.train_interactions.nnz}")
        print(f"Test interactions: {self.test_interactions.nnz}")
        
    def build_feature_matrices(self):
        """Build feature matrices"""
        print("\nBuilding feature matrices...")
        
        # Use identity matrices to ensure the system works
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)
        
        # Create identity matrices (each user/item has a unique feature)
        self.user_features_matrix = identity(num_users, format='csr')
        self.item_features_matrix = identity(num_items, format='csr')
        
        print(f"User features matrix: {self.user_features_matrix.shape}")
        print(f"Item features matrix: {self.item_features_matrix.shape}")
        print("Using identity matrices to avoid feature confusion")
        
    def train_model(self, loss='bpr', learning_rate=0.01, no_components=20, epochs=30):
        """Train the LightFM model with optimized parameters for sparse data"""
        print(f"\nTraining LightFM model with sparse data optimizations...")
        print(f"   Loss: {loss} (BPR works better for sparse data)")
        print(f"   Learning rate: {learning_rate} (reduced for stability)")
        print(f"   Components: {no_components} (reduced to prevent overfitting)")
        print(f"   Epochs: {epochs} (reduced for sparse data)")
        
        self.model = LightFM(
            loss=loss,
            learning_rate=learning_rate,
            no_components=no_components,
            item_alpha=0.0001,  # Item regularization to prevent overfitting
            user_alpha=0.0001,  # User regularization to prevent overfitting
            random_state=42
        )
        
        self.model.fit(
            interactions=self.train_interactions,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            epochs=epochs,
            verbose=True
        )
        
        print("Model training completed!")
        
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate model performance"""
        print("\nEvaluating model performance...")
        
        train_precision = precision_at_k(
            self.model, self.train_interactions,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            k=5
        ).mean()
        
        test_precision = precision_at_k(
            self.model, self.test_interactions,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            k=5
        ).mean()
        
        train_auc = auc_score(
            self.model, self.train_interactions,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix
        ).mean()
        
        test_auc = auc_score(
            self.model, self.test_interactions,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix
        ).mean()
        
        print(f"Training Precision@5: {train_precision:.4f}")
        print(f"Test Precision@5: {test_precision:.4f}")
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        return {
            'train_precision': float(train_precision),
            'test_precision': float(test_precision),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc)
        }
    
    def get_recommendations(self, user_id: str, num_recommendations: int = 10, filter_seen: bool = True) -> List[Dict[str, Any]]:
        """Get recommendations for a user with improved cold start handling"""
        if user_id not in self.user_id_map:
            print(f"User {user_id} not found in training data, using cold start recommendations")
            return self._get_cold_start_recommendations(num_recommendations)
        
        user_idx = self.user_id_map[user_id]
        all_item_indices = np.array(list(self.item_id_map.values()))
        
        scores = self.model.predict(
            user_ids=user_idx,
            item_ids=all_item_indices,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix
        )
        
        # Filter seen items
        if filter_seen:
            user_interactions = self.data_loader.interactions_df[
                self.data_loader.interactions_df['user_id'] == user_id
            ]
            for _, interaction in user_interactions.iterrows():
                if interaction['coupon_id'] in self.item_id_map:
                    seen_item_idx = self.item_id_map[interaction['coupon_id']]
                    if seen_item_idx < len(scores):
                        scores[seen_item_idx] = -999
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:num_recommendations]
        
        recommendations = []
        for idx in top_indices:
            if idx < len(all_item_indices):
                item_idx = all_item_indices[idx]
                coupon_id = self.reverse_item_map[item_idx]
                score = float(scores[idx])
                
                # Try to get coupon details from item_features_df first
                coupon_details = self.data_loader.item_features_df[
                    self.data_loader.item_features_df['coupon_id'] == coupon_id
                ]
                
                if len(coupon_details) > 0:
                    coupon_info = coupon_details.iloc[0]
                    recommendations.append({
                        'coupon_id': coupon_id,
                        'score': score,
                        'category': coupon_info['category'],
                        'has_image': bool(coupon_info['has_image']),
                        'days_until_expiry': int(coupon_info['days_until_expiry']) if pd.notna(coupon_info['days_until_expiry']) else None,
                        'view_count': int(coupon_info['view_count']),
                        'save_count': int(coupon_info['save_count']),
                        'vote_score': float(coupon_info['vote_score'])
                    })
                else:
                    # Fallback to all_coupons_df
                    coupon_details = self.data_loader.all_coupons_df[
                        self.data_loader.all_coupons_df['coupon_id'] == coupon_id
                    ]
                    if len(coupon_details) > 0:
                        coupon_info = coupon_details.iloc[0]
                        recommendations.append({
                            'coupon_id': coupon_id,
                            'score': score,
                            'title': coupon_info['title'],
                            'category': coupon_info['category'],
                            'coupon_type': coupon_info['coupon_type'],
                            'has_image': False,
                            'days_until_expiry': None,
                            'view_count': 0,
                            'save_count': 0,
                            'vote_score': 0.0
                        })
        
        print(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    
    def _get_cold_start_recommendations(self, num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Improved fallback for new users - return diverse popular coupons"""
        try:
            # Get popular coupons with category diversity
            popular_df = self.data_loader.popular_coupons_df.copy()
            
            # Ensure we have the required columns and convert to proper format
            recommendations = []
            categories_used = set()
            
            # First pass: get one from each category to ensure diversity
            for _, row in popular_df.iterrows():
                if len(recommendations) >= num_recommendations:
                    break
                    
                category = row.get('category', 'Unknown')
                if category not in categories_used:
                    recommendations.append({
                        'coupon_id': str(row.get('coupon_id', '')),
                        'score': float(row.get('popularity_score', 0)) / 10.0,  # Normalize to reasonable range
                        'category': category,
                        'title': row.get('title'),
                        'description': row.get('description'),
                        'discount_percentage': row.get('discount_percentage'),
                        'expires_at': row.get('expires_at'),
                        'has_image': False,
                        'days_until_expiry': row.get('days_until_expiry'),
                        'view_count': int(row.get('view_count', 0)),
                        'save_count': int(row.get('save_count', 0)),
                        'vote_score': float(row.get('vote_score', 0)),
                        'coupon_type': row.get('coupon_type', 'regular'),
                        'fresh_data': None
                    })
                    categories_used.add(category)
            
            # Second pass: fill remaining slots with highest popularity
            for _, row in popular_df.iterrows():
                if len(recommendations) >= num_recommendations:
                    break
                    
                coupon_id = str(row.get('coupon_id', ''))
                if not any(rec['coupon_id'] == coupon_id for rec in recommendations):
                    recommendations.append({
                        'coupon_id': coupon_id,
                        'score': float(row.get('popularity_score', 0)) / 10.0,
                        'category': row.get('category', 'Unknown'),
                        'title': row.get('title'),
                        'description': row.get('description'),
                        'discount_percentage': row.get('discount_percentage'),
                        'expires_at': row.get('expires_at'),
                        'has_image': False,
                        'days_until_expiry': row.get('days_until_expiry'),
                        'view_count': int(row.get('view_count', 0)),
                        'save_count': int(row.get('save_count', 0)),
                        'vote_score': float(row.get('vote_score', 0)),
                        'coupon_type': row.get('coupon_type', 'regular'),
                        'fresh_data': None
                    })
            
            print(f"Generated {len(recommendations)} cold start recommendations with category diversity")
            return recommendations
            
        except Exception as e:
            print(f"Error in cold start recommendations: {e}")
            # Fallback to simple approach
            return self.data_loader.popular_coupons_df.head(num_recommendations).to_dict('records')
    
    def get_similar_items(self, coupon_id: str, num_similar: int = 5) -> List[Dict[str, Any]]:
        """Get similar coupons"""
        if coupon_id not in self.item_id_map:
            print(f"Coupon {coupon_id} not found")
            return []
        
        item_idx = self.item_id_map[coupon_id]
        item_embeddings = self.model.item_embeddings
        
        target_embedding = item_embeddings[item_idx].reshape(1, -1)
        similarities = cosine_similarity(target_embedding, item_embeddings)[0]
        
        similar_indices = np.argsort(similarities)[::-1]
        
        similar_items = []
        for idx in similar_indices:
            if idx == item_idx:
                continue
                
            if idx in self.reverse_item_map:
                similar_coupon_id = self.reverse_item_map[idx]
                similarity_score = float(similarities[idx])
                
                # Try to get details from item_features_df first
                coupon_details = self.data_loader.item_features_df[
                    self.data_loader.item_features_df['coupon_id'] == similar_coupon_id
                ]
                
                if len(coupon_details) > 0:
                    coupon_info = coupon_details.iloc[0]
                    similar_items.append({
                        'coupon_id': similar_coupon_id,
                        'similarity_score': similarity_score,
                        'category': coupon_info['category'],
                        'view_count': int(coupon_info['view_count']),
                        'save_count': int(coupon_info['save_count'])
                    })
                else:
                    # Fallback to all_coupons_df
                    coupon_details = self.data_loader.all_coupons_df[
                        self.data_loader.all_coupons_df['coupon_id'] == similar_coupon_id
                    ]
                    if len(coupon_details) > 0:
                        coupon_info = coupon_details.iloc[0]
                        similar_items.append({
                            'coupon_id': similar_coupon_id,
                            'similarity_score': similarity_score,
                            'title': coupon_info['title'],
                            'category': coupon_info['category'],
                            'coupon_type': coupon_info['coupon_type'],
                            'view_count': 0,
                            'save_count': 0
                        })
                        
                if len(similar_items) >= num_similar:
                    break
        
        return similar_items