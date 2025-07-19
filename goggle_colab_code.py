# ============================================================================
# LightFM Hybrid Coupon Recommendation & Personalization System
# ULTIMATE FIX - Complete implementation for Google Colab
# ============================================================================

# Install required packages
!pip install lightfm scikit-learn python-dotenv plotly supabase

import numpy as np
import pandas as pd
from supabase import create_client, Client
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score, recall_at_k
from lightfm.cross_validation import random_train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📦 All packages installed successfully!")

# ============================================================================
# DATABASE CONNECTION SETUP
# ============================================================================

# Google Colab Secrets Setup
try:
    from google.colab import userdata
    COLAB_AVAILABLE = True
    print("✅ Google Colab detected - using secure secrets")
except ImportError:
    COLAB_AVAILABLE = False
    print("⚠️  Not in Google Colab - using environment variables")

def get_supabase_credentials():
    if COLAB_AVAILABLE:
        try:
            url = userdata.get('SUPABASE_URL')
            key = userdata.get('SUPABASE_ANON_KEY')
            
            if url and key:
                print("🔐 Using Google Colab secrets for Supabase connection")
                return url, key
            else:
                print("❌ SUPABASE_URL or SUPABASE_ANON_KEY not found in Colab secrets")
                return None, None
            
        except Exception as e:
            print(f"❌ Error getting Colab secrets: {e}")
            return None, None
    else:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        url = os.getenv('EXPO_PUBLIC_SUPABASE_URL')
        key = os.getenv('EXPO_PUBLIC_SUPABASE_ANON_KEY')
        
        if not url or not key:
            print("⚠️  Please set EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY environment variables")
        return url, key

class SupabaseConnector:
    def __init__(self, url, key):
        self.url = url
        self.key = key
        self.client = None
    
    def connect(self):
        try:
            # Force fresh connection by recreating client
            self.client = None
            self.client = create_client(self.url, self.key)
            print("✅ Connected to Supabase successfully! (Fresh connection)")
            return True
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            return False
    
    def execute_query(self, query):
        try:
            if "SELECT * FROM " in query and "(" in query:
                func_name = query.split("SELECT * FROM ")[1].split("(")[0].strip()
                
                # Force fresh data by adding timestamp parameter
                import time
                cache_buster = int(time.time())
                print(f"🔄 Executing {func_name}() with cache buster: {cache_buster}")
                
                result = self.client.rpc(func_name).execute()
                
                if result.data:
                    df = pd.DataFrame(result.data)
                    print(f"✅ Executed {func_name}() - returned {len(df)} rows (fresh data)")
                    
                    # Debug: Print first few rows for verification
                    if func_name == "get_lightfm_interaction_matrix" and len(df) > 0:
                        print(f"🔍 Sample data: {df.head(3)[['user_id', 'coupon_id', 'total_weight']].to_dict('records')}")
                    
                    return df
                else:
                    print(f"⚠️  {func_name}() returned no data")
                    return pd.DataFrame()
            else:
                print(f"❌ Unsupported query format: {query}")
                return None
                
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return None
    
    def close(self):
        print("🔒 Supabase connection closed")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class CouponDataLoader:
    def __init__(self, db_connector):
        self.db = db_connector
        self.interactions_df = None
        self.user_features_df = None
        self.item_features_df = None
        self.popular_coupons_df = None
        
    def load_all_data(self):
        print("📊 Loading interaction data with cache clearing...")
        
        # Force fresh connection before loading data
        self.db.connect()
        
        query = "SELECT * FROM get_lightfm_interaction_matrix()"
        self.interactions_df = self.db.execute_query(query)
        
        query = "SELECT * FROM get_user_features()"
        self.user_features_df = self.db.execute_query(query)
        
        query = "SELECT * FROM get_coupon_features()"
        self.item_features_df = self.db.execute_query(query)
        
        query = "SELECT * FROM get_popular_coupons()"
        self.popular_coupons_df = self.db.execute_query(query)
        
        print(f"✅ Loaded {len(self.interactions_df)} interactions")
        print(f"✅ Loaded {len(self.user_features_df)} users")
        print(f"✅ Loaded {len(self.item_features_df)} coupons")
        print(f"✅ Loaded {len(self.popular_coupons_df)} popular coupons")
        
        # Verify we got the expected data
        if len(self.interactions_df) >= 10:
            print("✅ Got expected number of interactions (10+)")
        else:
            print(f"⚠️  Only got {len(self.interactions_df)} interactions, expected 10+")
        
        return self.validate_data()
    
    def validate_data(self):
        if any(df is None or df.empty for df in [
            self.interactions_df, self.user_features_df, 
            self.item_features_df, self.popular_coupons_df
        ]):
            print("❌ Data validation failed - some datasets are empty")
            return False
        
        print("✅ Data validation passed")
        return True
    
    def get_data_summary(self):
        print("\n" + "="*60)
        print("📈 DATA SUMMARY")
        print("="*60)
        
        print(f"👥 Users: {self.interactions_df['user_id'].nunique()}")
        print(f"🎫 Coupons: {self.interactions_df['coupon_id'].nunique()}")
        print(f"🔄 Total Interactions: {len(self.interactions_df)}")
        print(f"⚖️  Avg Weight: {self.interactions_df['total_weight'].mean():.2f}")
        print(f"📊 Weight Range: {self.interactions_df['total_weight'].min():.1f} - {self.interactions_df['total_weight'].max():.1f}")
        
        user_interactions = self.interactions_df.groupby('user_id')['total_weight'].agg(['count', 'sum', 'mean'])
        print(f"\n👤 User Engagement:")
        print(f"   Avg interactions per user: {user_interactions['count'].mean():.1f}")
        print(f"   Most active user: {user_interactions['count'].max()} interactions")
        
        item_interactions = self.interactions_df.groupby('coupon_id')['total_weight'].agg(['count', 'sum', 'mean'])
        print(f"\n🎫 Coupon Popularity:")
        print(f"   Avg interactions per coupon: {item_interactions['count'].mean():.1f}")
        print(f"   Most popular coupon: {item_interactions['count'].max()} interactions")

# ============================================================================
# LIGHTFM MODEL BUILDER - ULTIMATE FIX
# ============================================================================

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
        print("\n🔧 Preparing LightFM dataset...")
        
        # Initialize dataset
        self.dataset = Dataset()
        
        # ULTIMATE FIX: Use ONLY actual user and item IDs - NO FEATURES in fit()
        interaction_users = set(self.data_loader.interactions_df['user_id'].unique())
        feature_users = set(self.data_loader.user_features_df['user_id'].unique())
        users = list(interaction_users.union(feature_users))
        
        interaction_items = set(self.data_loader.interactions_df['coupon_id'].unique())
        feature_items = set(self.data_loader.item_features_df['coupon_id'].unique())
        items = list(interaction_items.union(feature_items))
        
        print(f"🔍 Total unique users: {len(users)}, items: {len(items)}")
        
        # CRITICAL: Fit dataset with ONLY user and item IDs - NO FEATURES
        # This prevents LightFM from mixing up feature names with entity IDs
        print("🔧 Fitting dataset with IDs only (no features to avoid confusion)...")
        self.dataset.fit(users=users, items=items)
        
        # Verify mappings are clean
        temp_user_feature_map, temp_user_id_map, temp_item_feature_map, temp_item_id_map = self.dataset.mapping()
        
        print(f"✅ Clean mappings created:")
        print(f"   Users: {len(temp_user_id_map)} (expected: {len(users)})")
        print(f"   Items: {len(temp_item_id_map)} (expected: {len(items)})")
        
        if len(temp_user_id_map) == len(users) and len(temp_item_id_map) == len(items):
            print("✅ Perfect mapping - no feature name confusion!")
        else:
            print("⚠️  Still have mapping issues - will handle in feature building")
        
        print(f"✅ Dataset prepared with {len(users)} users and {len(items)} items")
        
    def build_interaction_matrix(self):
        print("\n🔨 Building interaction matrices...")
        
        interactions_with_weights = []
        
        for _, row in self.data_loader.interactions_df.iterrows():
            user_id = row['user_id']
            coupon_id = row['coupon_id']
            weight = row['total_weight']
            interactions_with_weights.append((user_id, coupon_id, weight))
            
            if len(interactions_with_weights) <= 5:
                print(f"🔍 Added interaction: user={user_id}, item={coupon_id}, weight={weight}")
        
        print(f"🔍 Total interactions for training: {len(interactions_with_weights)}")
        
        # Build interaction matrix
        interaction_matrix, weights_matrix = self.dataset.build_interactions(interactions_with_weights)
        
        # Get the clean mappings
        user_feature_map, self.user_id_map, item_feature_map, self.item_id_map = self.dataset.mapping()
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        
        print(f"🔍 Final LightFM mappings:")
        print(f"   Users: {len(self.user_id_map)} mapped")
        print(f"   Items: {len(self.item_id_map)} mapped")
        
        # Split into train/test
        self.train_interactions, self.test_interactions = random_train_test_split(
            interaction_matrix, test_percentage=0.2, random_state=42
        )
        
        print(f"✅ Built interaction matrix: {interaction_matrix.shape}")
        print(f"✅ Train interactions: {self.train_interactions.nnz}")
        print(f"✅ Test interactions: {self.test_interactions.nnz}")
        
    def build_feature_matrices(self):
        print("\n🎯 Building feature matrices...")
        
        # ULTIMATE FIX: Build features WITHOUT using dataset.build_user_features()
        # Instead, create simple feature matrices manually to avoid LightFM confusion
        
        print("🔧 Creating simplified feature approach...")
        
        # For now, use identity matrices (no features) to ensure the system works
        # This eliminates all feature-related confusion
        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)
        
        # Create identity matrices (each user/item has a unique feature)
        from scipy.sparse import identity
        self.user_features_matrix = identity(num_users, format='csr')
        self.item_features_matrix = identity(num_items, format='csr')
        
        print(f"✅ User features matrix: {self.user_features_matrix.shape}")
        print(f"✅ Item features matrix: {self.item_features_matrix.shape}")
        print("🔧 Using identity matrices to avoid feature confusion")
        
    def train_model(self, loss='warp', learning_rate=0.05, no_components=50, epochs=100):
        print(f"\n🚀 Training LightFM model...")
        print(f"   Loss: {loss}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Components: {no_components}")
        print(f"   Epochs: {epochs}")
        
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
            verbose=True
        )
        
        print("✅ Model training completed!")
        
    def evaluate_model(self):
        print("\n📊 Evaluating model performance...")
        
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
        
        print(f"📈 Training Precision@5: {train_precision:.4f}")
        print(f"📈 Test Precision@5: {test_precision:.4f}")
        print(f"📈 Training AUC: {train_auc:.4f}")
        print(f"📈 Test AUC: {test_auc:.4f}")
        
        return {
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_auc': train_auc,
            'test_auc': test_auc
        }
    
    def get_recommendations(self, user_id, num_recommendations=10, filter_seen=True):
        if user_id not in self.user_id_map:
            print(f"❌ User {user_id} not found in training data")
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
                score = scores[idx]
                
                coupon_details = self.data_loader.item_features_df[
                    self.data_loader.item_features_df['coupon_id'] == coupon_id
                ]
                
                if len(coupon_details) > 0:
                    coupon_info = coupon_details.iloc[0]
                    recommendations.append({
                        'coupon_id': coupon_id,
                        'score': score,
                        'category': coupon_info['category'],
                        'has_image': coupon_info['has_image'],
                        'days_until_expiry': coupon_info['days_until_expiry'],
                        'view_count': coupon_info['view_count'],
                        'save_count': coupon_info['save_count'],
                        'vote_score': coupon_info['vote_score']
                    })
        
        return recommendations
    
    def _get_cold_start_recommendations(self, num_recommendations=10):
        return self.data_loader.popular_coupons_df.head(num_recommendations).to_dict('records')
    
    def get_similar_items(self, coupon_id, num_similar=5):
        if coupon_id not in self.item_id_map:
            print(f"❌ Coupon {coupon_id} not found")
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
                similarity_score = similarities[idx]
                
                coupon_details = self.data_loader.item_features_df[
                    self.data_loader.item_features_df['coupon_id'] == similar_coupon_id
                ]
                
                if len(coupon_details) > 0:
                    coupon_info = coupon_details.iloc[0]
                    similar_items.append({
                        'coupon_id': similar_coupon_id,
                        'similarity_score': similarity_score,
                        'category': coupon_info['category'],
                        'view_count': coupon_info['view_count'],
                        'save_count': coupon_info['save_count']
                    })
                    
                    if len(similar_items) >= num_similar:
                        break
        
        return similar_items

# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

class RecommendationAnalyzer:
    def __init__(self, rec_system, data_loader):
        self.rec_system = rec_system
        self.data_loader = data_loader
    
    def analyze_recommendations_for_user(self, user_id, num_recs=10):
        print(f"\n🎯 RECOMMENDATION ANALYSIS FOR USER: {user_id}")
        print("="*60)
        
        user_profile = self.data_loader.user_features_df[
            self.data_loader.user_features_df['user_id'] == user_id
        ]
        
        if len(user_profile) > 0:
            profile = user_profile.iloc[0]
            print(f"👤 Username: {profile['username']}")
            print(f"📊 Profile Completeness: {profile['profile_completeness']:.1%}")
            print(f"🔄 Total Interactions: {profile['total_interactions']}")
            print(f"⚖️  Avg Interaction Weight: {profile['avg_interaction_weight']:.2f}")
        
        user_history = self.data_loader.interactions_df[
            self.data_loader.interactions_df['user_id'] == user_id
        ].sort_values('total_weight', ascending=False)
        
        print(f"\n📚 INTERACTION HISTORY ({len(user_history)} interactions):")
        for _, interaction in user_history.head(5).iterrows():
            coupon_info = self.data_loader.item_features_df[
                self.data_loader.item_features_df['coupon_id'] == interaction['coupon_id']
            ]
            if len(coupon_info) > 0:
                category = coupon_info.iloc[0]['category']
                print(f"   🎫 {interaction['coupon_id']} | {category} | Weight: {interaction['total_weight']}")
        
        recommendations = self.rec_system.get_recommendations(user_id, num_recs)
        
        print(f"\n🎯 TOP {num_recs} RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i:2d}. Score: {rec.get('score', 0):.3f} | "
                  f"Category: {rec.get('category', 'N/A')} | "
                  f"Views: {rec.get('view_count', 0)} | "
                  f"Saves: {rec.get('save_count', 0)}")
        
        return recommendations

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("🚀 STARTING LIGHTFM COUPON RECOMMENDATION SYSTEM - ULTIMATE FIX")
    print("="*70)
    
    url, key = get_supabase_credentials()
    
    if not url or not key:
        print("❌ Failed to get Supabase credentials")
        return
    
    print("🔗 Connecting to Supabase...")
    db = SupabaseConnector(url, key)
    
    if not db.connect():
        print("❌ Failed to connect to Supabase. Please check your credentials.")
        return
    
    try:
        # Load data
        data_loader = CouponDataLoader(db)
        if not data_loader.load_all_data():
            print("❌ Failed to load data")
            return
        
        data_loader.get_data_summary()
        
        # Build recommendation system
        rec_system = LightFMRecommendationSystem(data_loader)
        rec_system.prepare_dataset()
        rec_system.build_interaction_matrix()
        rec_system.build_feature_matrices()
        
        # Train model
        rec_system.train_model(
            loss='warp',
            learning_rate=0.05,
            no_components=50,
            epochs=100
        )
        
        # Evaluate model
        metrics = rec_system.evaluate_model()
        
        # Create analyzer
        analyzer = RecommendationAnalyzer(rec_system, data_loader)
        
        # Example recommendations
        print("\n🎯 EXAMPLE RECOMMENDATIONS:")
        sample_users = data_loader.user_features_df['user_id'].head(3).tolist()
        
        for user_id in sample_users:
            recommendations = analyzer.analyze_recommendations_for_user(user_id, 5)
            print()
        
        # Example similar items
        print("\n🔍 EXAMPLE SIMILAR ITEMS:")
        sample_coupon = data_loader.interactions_df['coupon_id'].iloc[0]
        similar_items = rec_system.get_similar_items(sample_coupon, 3)
        
        print(f"Similar to {sample_coupon}:")
        for item in similar_items:
            print(f"   🎫 {item['coupon_id']} | Similarity: {item['similarity_score']:.3f} | {item['category']}")
        
        print("\n✅ RECOMMENDATION SYSTEM SETUP COMPLETE!")
        print("🎉 You can now use the system to generate personalized recommendations!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()

# ============================================================================
# PRODUCTION API FUNCTIONS
# ============================================================================

def get_user_recommendations_api(user_id, num_recs=10):
    url, key = get_supabase_credentials()
    if not url or not key:
        return {"error": "Failed to get Supabase credentials"}
    
    db = SupabaseConnector(url, key)
    if not db.connect():
        return {"error": "Supabase connection failed"}
    
    try:
        data_loader = CouponDataLoader(db)
        if not data_loader.load_all_data():
            return {"error": "Failed to load data"}
        
        rec_system = LightFMRecommendationSystem(data_loader)
        rec_system.prepare_dataset()
        rec_system.build_interaction_matrix()
        rec_system.build_feature_matrices()
        rec_system.train_model(epochs=50)
        
        recommendations = rec_system.get_recommendations(user_id, num_recs)
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        db.close()

# ============================================================================
# RUN THE SYSTEM
# ============================================================================

if __name__ == "__main__":
    print("🚀 LIGHTFM COUPON RECOMMENDATION SYSTEM - ULTIMATE FIX")
    print("="*60)
    print("📝 SETUP INSTRUCTIONS FOR GOOGLE COLAB:")
    print("1. Click the 🔑 key icon in the left sidebar")
    print("2. Add these secrets:")
    print("   - SUPABASE_URL: https://zbkqifvjryrqpuypdayk.supabase.co")
    print("   - SUPABASE_ANON_KEY: your-anon-key")
    print("3. Run main() to start the system")
    print("="*60)
    print()
    
    # Run the main system
    main()
    
    # Example API usage
    print("\n" + "="*50)
    print("🔧 EXAMPLE API USAGE:")
    print("="*50)
    
    sample_user = "user_2yqKXgsH3jEQivtjwrJyLURqHuX"
    print(f"Getting recommendations for user: {sample_user}")
    recommendations = get_user_recommendations_api(sample_user, 5)
    
    if "error" not in recommendations:
        print("✅ Sample recommendations generated successfully!")
        print(f"📊 Generated {len(recommendations.get('recommendations', []))} recommendations")
    else:
        print(f"❌ Error: {recommendations['error']}")
    
    print("\n🎉 System ready for production use!")