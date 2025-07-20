"""
Main recommendation service that orchestrates the entire system
"""

import os
import json
import pickle
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio

from .database_service import SupabaseConnector, CouponDataLoader, get_supabase_credentials
from .lightfm_service import LightFMRecommendationSystem

class RecommendationService:
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = cache_dir
        self.model_cache_file = os.path.join(cache_dir, "lightfm_model.pkl")
        self.metadata_file = os.path.join(cache_dir, "model_metadata.json")
        self.rec_system = None
        self.data_loader = None
        self.last_trained = None
        self.model_lock = threading.Lock()
        
        # Configuration
        self.RETRAIN_INTERVAL_HOURS = 6
        self.FORCE_RETRAIN_THRESHOLD = 100
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
    async def initialize(self) -> bool:
        """Initialize the recommendation service"""
        try:
            # Try to load cached model first
            if not await self.load_cached_model():
                print("🔄 No cached model found, training fresh model...")
                return await self.train_fresh_model()
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            # Check if model is loaded
            if self.rec_system is None:
                return False
            
            # Check if we can get credentials
            url, key = get_supabase_credentials()
            if not url or not key:
                return False
            
            # Try a simple database connection
            db = SupabaseConnector(url, key)
            if not db.connect():
                return False
            
            db.close()
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        status = {
            "is_trained": self.rec_system is not None,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "cache_status": "cached" if os.path.exists(self.model_cache_file) else "no_cache",
            "timestamp": datetime.now().isoformat()
        }
        
        if self.data_loader:
            status.update({
                "total_users": len(self.data_loader.user_features_df) if self.data_loader.user_features_df is not None else 0,
                "total_coupons": len(self.data_loader.all_coupons_df) if self.data_loader.all_coupons_df is not None else 0,
                "total_interactions": len(self.data_loader.interactions_df) if self.data_loader.interactions_df is not None else 0,
                "model_version": "1.0"
            })
        
        return status
    
    async def should_retrain(self) -> bool:
        """Check if model needs retraining"""
        if not os.path.exists(self.model_cache_file):
            return True
            
        # Check time-based retraining
        if self.last_trained:
            time_diff = datetime.now() - self.last_trained
            if time_diff > timedelta(hours=self.RETRAIN_INTERVAL_HOURS):
                print(f"🕒 Time-based retrain needed (last trained: {time_diff} ago)")
                return True
        
        return False
    
    async def load_cached_model(self) -> bool:
        """Load model from cache if available"""
        try:
            if os.path.exists(self.model_cache_file):
                with open(self.model_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.rec_system = cached_data['rec_system']
                self.data_loader = cached_data['data_loader']
                
                # Load metadata
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.last_trained = datetime.fromisoformat(metadata['last_trained'])
                
                print(f"✅ Loaded cached model from {self.last_trained}")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load cached model: {e}")
            return False
        
        return False
    
    async def save_model_cache(self):
        """Save trained model to cache"""
        try:
            with self.model_lock:
                # Create a clean copy of data_loader without database connection
                clean_data_loader = CouponDataLoader(None)
                clean_data_loader.interactions_df = self.data_loader.interactions_df.copy()
                clean_data_loader.user_features_df = self.data_loader.user_features_df.copy()
                clean_data_loader.item_features_df = self.data_loader.item_features_df.copy()
                clean_data_loader.popular_coupons_df = self.data_loader.popular_coupons_df.copy()
                clean_data_loader.all_coupons_df = self.data_loader.all_coupons_df.copy()
                
                # Create a clean copy of rec_system
                clean_rec_system = LightFMRecommendationSystem(clean_data_loader)
                clean_rec_system.dataset = self.rec_system.dataset
                clean_rec_system.model = self.rec_system.model
                clean_rec_system.user_id_map = self.rec_system.user_id_map.copy()
                clean_rec_system.item_id_map = self.rec_system.item_id_map.copy()
                clean_rec_system.reverse_user_map = self.rec_system.reverse_user_map.copy()
                clean_rec_system.reverse_item_map = self.rec_system.reverse_item_map.copy()
                clean_rec_system.train_interactions = self.rec_system.train_interactions
                clean_rec_system.test_interactions = self.rec_system.test_interactions
                clean_rec_system.user_features_matrix = self.rec_system.user_features_matrix
                clean_rec_system.item_features_matrix = self.rec_system.item_features_matrix
                
                # Save only the clean objects
                cached_data = {
                    'rec_system': clean_rec_system,
                    'data_loader': clean_data_loader
                }
                
                with open(self.model_cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
                
                # Save metadata
                metadata = {
                    'last_trained': datetime.now().isoformat(),
                    'model_version': '1.0',
                    'total_users': len(self.data_loader.user_features_df),
                    'total_coupons': len(self.data_loader.all_coupons_df),
                    'total_interactions': len(self.data_loader.interactions_df)
                }
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.last_trained = datetime.now()
                print("✅ Model cached successfully")
                
        except Exception as e:
            print(f"❌ Failed to cache model: {e}")
    
    async def get_recommendations(self, user_id: str, num_recommendations: int = 10, fresh_coupon_data: bool = True) -> Dict[str, Any]:
        """Get recommendations with caching and optimization"""
        with self.model_lock:
            # Check if we need to retrain
            if await self.should_retrain() or self.rec_system is None:
                print("🔄 Model needs retraining...")
                success = await self.train_fresh_model()
                if not success:
                    return {"error": "Failed to train model"}
            
            # Generate recommendations
            try:
                recommendations = self.rec_system.get_recommendations(user_id, num_recommendations)
                
                result = {
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "model_last_trained": self.last_trained.isoformat() if self.last_trained else None,
                    "total_active_coupons": len(self.data_loader.all_coupons_df),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Optionally enrich with fresh coupon data
                if fresh_coupon_data:
                    result = await self._enrich_with_fresh_coupon_data(result)
                
                return result
                
            except Exception as e:
                return {"error": f"Recommendation generation failed: {str(e)}"}
    
    async def get_similar_items(self, coupon_id: str, num_similar: int = 5) -> List[Dict[str, Any]]:
        """Get similar items"""
        if self.rec_system:
            return self.rec_system.get_similar_items(coupon_id, num_similar)
        return []
    
    async def get_popular_coupons(self, limit: int = 20, category: Optional[str] = None) -> Dict[str, Any]:
        """Get popular/trending coupons"""
        try:
            # Ensure we have data loaded
            if self.data_loader is None or self.data_loader.popular_coupons_df is None:
                # Try to get fresh popular coupons data
                url, key = get_supabase_credentials()
                if not url or not key:
                    return {"error": "Database credentials not available"}
                
                db = SupabaseConnector(url, key)
                if not db.connect():
                    return {"error": "Database connection failed"}
                
                # Load just popular coupons data
                query = "SELECT * FROM get_popular_coupons()"
                popular_df = db.execute_query(query)
                db.close()
                
                if popular_df is None or popular_df.empty:
                    return {"error": "No popular coupons data available"}
            else:
                popular_df = self.data_loader.popular_coupons_df.copy()
            
            # Filter by category if specified
            if category:
                popular_df = popular_df[popular_df['category'].str.lower() == category.lower()]
            
            # Sort by popularity score (assuming higher is better)
            if 'popularity_score' in popular_df.columns:
                popular_df = popular_df.sort_values('popularity_score', ascending=False)
            elif 'view_count' in popular_df.columns:
                popular_df = popular_df.sort_values('view_count', ascending=False)
            elif 'save_count' in popular_df.columns:
                popular_df = popular_df.sort_values('save_count', ascending=False)
            
            # Limit results
            popular_df = popular_df.head(limit)
            
            # Convert to list of dictionaries
            popular_coupons = []
            for _, row in popular_df.iterrows():
                coupon_data = {
                    'coupon_id': str(row.get('coupon_id', row.get('id', ''))),
                    'title': row.get('title'),
                    'category': row.get('category'),
                    'description': row.get('description'),
                    'discount_percentage': row.get('discount_percentage'),
                    'expires_at': row.get('expires_at'),
                    'popularity_score': row.get('popularity_score'),
                    'view_count': row.get('view_count'),
                    'save_count': row.get('save_count'),
                    'vote_score': row.get('vote_score'),
                    'coupon_type': row.get('coupon_type', 'regular'),
                    'days_until_expiry': row.get('days_until_expiry'),
                    'is_trending': row.get('is_trending', False)
                }
                popular_coupons.append(coupon_data)
            
            return {
                "popular_coupons": popular_coupons,
                "total_count": len(popular_coupons),
                "category_filter": category,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get popular coupons: {str(e)}"}
    
    async def _enrich_popular_coupons_with_fresh_data(self, popular_coupons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich popular coupons with fresh data from database"""
        try:
            url, key = get_supabase_credentials()
            if not url or not key:
                return popular_coupons
            
            db = SupabaseConnector(url, key)
            if not db.connect():
                return popular_coupons
            
            # Get coupon IDs
            coupon_ids = [coupon['coupon_id'] for coupon in popular_coupons]
            
            if coupon_ids:
                # Fetch fresh coupon details
                fresh_coupons = db.client.table('coupons').select(
                    'id, title, category, description, discount_percentage, is_active, expires_at'
                ).in_('id', coupon_ids).eq('is_active', True).execute()
                
                if fresh_coupons.data:
                    fresh_coupon_dict = {c['id']: c for c in fresh_coupons.data}
                    
                    # Enrich popular coupons with fresh data
                    for coupon in popular_coupons:
                        coupon_id = coupon['coupon_id']
                        if coupon_id in fresh_coupon_dict:
                            fresh_data = fresh_coupon_dict[coupon_id]
                            coupon.update({
                                'title': fresh_data.get('title') or coupon.get('title'),
                                'description': fresh_data.get('description') or coupon.get('description'),
                                'discount_percentage': fresh_data.get('discount_percentage') or coupon.get('discount_percentage'),
                                'expires_at': fresh_data.get('expires_at') or coupon.get('expires_at'),
                                'fresh_data': True
                            })
            
            db.close()
            
        except Exception as e:
            print(f"⚠️ Failed to enrich popular coupons with fresh data: {e}")
        
        return popular_coupons
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if self.rec_system:
            metrics = self.rec_system.evaluate_model()
            metrics['timestamp'] = datetime.now().isoformat()
            return metrics
        return {"error": "Model not trained"}
    
    async def force_retrain(self) -> bool:
        """Force retrain the model"""
        return await self.train_fresh_model()
    
    async def train_fresh_model(self) -> bool:
        """Train a fresh model"""
        try:
            print("🚀 Training fresh model...")
            
            # Get database connection
            url, key = get_supabase_credentials()
            if not url or not key:
                return False
            
            db = SupabaseConnector(url, key)
            if not db.connect():
                return False
            
            # Load fresh data
            self.data_loader = CouponDataLoader(db)
            if not self.data_loader.load_all_data():
                return False
            
            # Build and train model
            self.rec_system = LightFMRecommendationSystem(self.data_loader)
            self.rec_system.prepare_dataset()
            self.rec_system.build_interaction_matrix()
            self.rec_system.build_feature_matrices()
            self.rec_system.train_model(epochs=50)  # Reduced epochs for faster training
            
            # Cache the model
            await self.save_model_cache()
            
            db.close()
            return True
            
        except Exception as e:
            print(f"❌ Fresh model training failed: {e}")
            return False
    
    async def _enrich_with_fresh_coupon_data(self, recommendations_response: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich recommendations with fresh coupon data from database"""
        try:
            url, key = get_supabase_credentials()
            if not url or not key:
                return recommendations_response
            
            db = SupabaseConnector(url, key)
            if not db.connect():
                return recommendations_response
            
            # Get fresh coupon data
            coupon_ids = [rec['coupon_id'] for rec in recommendations_response['recommendations']]
            
            if coupon_ids:
                # Fetch fresh coupon details
                fresh_coupons = db.client.table('coupons').select(
                    'id, title, category, description, discount_percentage, is_active, expires_at'
                ).in_('id', coupon_ids).eq('is_active', True).execute()
                
                if fresh_coupons.data:
                    fresh_coupon_dict = {c['id']: c for c in fresh_coupons.data}
                    
                    # Enrich recommendations with fresh data
                    for rec in recommendations_response['recommendations']:
                        coupon_id = rec['coupon_id']
                        if coupon_id in fresh_coupon_dict:
                            fresh_data = fresh_coupon_dict[coupon_id]
                            rec.update({
                                'title': fresh_data.get('title'),
                                'description': fresh_data.get('description'),
                                'discount_percentage': fresh_data.get('discount_percentage'),
                                'expires_at': fresh_data.get('expires_at'),
                                'fresh_data': True
                            })
            
            db.close()
            
        except Exception as e:
            print(f"⚠️ Failed to enrich with fresh data: {e}")
        
        return recommendations_response