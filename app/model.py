import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import identity
from sklearn.metrics.pairwise import cosine_similarity

class LightFMRecommendationSystem:
    def __init__(self, data_loader):
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
        self.dataset = Dataset()
        interaction_users = set(self.data_loader.interactions_df['user_id'].unique())
        feature_users = set(self.data_loader.user_features_df['user_id'].unique())
        users = list(interaction_users.union(feature_users))
        
        interaction_items = set(self.data_loader.interactions_df['coupon_id'].unique())
        feature_items = set(self.data_loader.item_features_df['coupon_id'].unique())
        items = list(interaction_items.union(feature_items))
        
        self.dataset.fit(users=users, items=items)
        
    def build_interaction_matrix(self):
        interactions_with_weights = []
        for _, row in self.data_loader.interactions_df.iterrows():
            interactions_with_weights.append((
                row['user_id'], 
                row['coupon_id'], 
                row['total_weight']
            ))
        
        interaction_matrix, _ = self.dataset.build_interactions(
            interactions_with_weights
        )
        
        _, self.user_id_map, _, self.item_id_map = self.dataset.mapping()
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        
        self.train_interactions, self.test_interactions = random_train_test_split(
            interaction_matrix, test_percentage=0.2, random_state=42
        )
    
    def build_feature_matrices(self):
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)
        self.user_features_matrix = identity(num_users, format='csr')
        self.item_features_matrix = identity(num_items, format='csr')
    
    def train_model(self, loss='warp', learning_rate=0.05, no_components=50, epochs=100):
        self.model = LightFM(
            loss=loss,
            learning_rate=learning_rate,
            no_components=no_components,
            random_state=42
        )
        
        self.model.fit(
            interactions=self.train_interactions,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
            epochs=epochs,
            verbose=False
        )
    
    def get_recommendations(self, user_id, num_recommendations=10):
        if user_id not in self.user_id_map:
            return self.data_loader.popular_coupons_df.head(
                num_recommendations
            ).to_dict('records')
        
        user_idx = self.user_id_map[user_id]
        all_item_indices = np.array(list(self.item_id_map.values()))
        
        scores = self.model.predict(
            user_ids=user_idx,
            item_ids=all_item_indices,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix
        )
        
        top_indices = np.argsort(scores)[::-1][:num_recommendations]
        recommendations = []
        
        for idx in top_indices:
            item_idx = all_item_indices[idx]
            coupon_id = self.reverse_item_map[item_idx]
            coupon_details = self.data_loader.item_features_df[
                self.data_loader.item_features_df['coupon_id'] == coupon_id
            ]
            
            if not coupon_details.empty:
                rec = coupon_details.iloc[0].to_dict()
                rec['score'] = float(scores[idx])
                recommendations.append(rec)
                
        return recommendations
    
    def get_similar_items(self, coupon_id, num_similar=5):
        if coupon_id not in self.item_id_map:
            return []
        
        item_idx = self.item_id_map[coupon_id]
        target_embedding = self.model.item_embeddings[item_idx].reshape(1, -1)
        similarities = cosine_similarity(
            target_embedding, 
            self.model.item_embeddings
        )[0]
        
        similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]
        similar_items = []
        
        for idx in similar_indices:
            similar_coupon_id = self.reverse_item_map[idx]
            coupon_details = self.data_loader.item_features_df[
                self.data_loader.item_features_df['coupon_id'] == similar_coupon_id
            ]
            
            if not coupon_details.empty:
                item = coupon_details.iloc[0].to_dict()
                item['similarity_score'] = float(similarities[idx])
                similar_items.append(item)
                
        return similar_items

    def refresh_and_retrain(self):
        """Refresh data and retrain model"""
        # Close existing connection
        if self.data_loader.db.client:
            self.data_loader.db.close()
        
        # Load fresh data
        if not self.data_loader.refresh_data():
            return False
        
        # Retrain with new data
        self.prepare_dataset()
        self.build_interaction_matrix()
        self.build_feature_matrices()
        self.train_model(epochs=50)
        return True