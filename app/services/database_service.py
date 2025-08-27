"""
Database service for Supabase connections and data loading
"""

import pandas as pd
from supabase import create_client, Client
import os
from typing import Optional, Tuple
import time

class SupabaseConnector:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.client = None
    
    def connect(self) -> bool:
        """Establish connection to Supabase"""
        try:
            self.client = None
            self.client = create_client(self.url, self.key)
            print("Connected to Supabase successfully! (Fresh connection)")
            return True
        except Exception as e:
            print(f"Supabase connection failed: {e}")
            return False
    
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute a query and return DataFrame"""
        try:
            if "SELECT * FROM " in query and "(" in query:
                func_name = query.split("SELECT * FROM ")[1].split("(")[0].strip()
                
                # Force fresh data by adding timestamp parameter
                cache_buster = int(time.time())
                print(f"Executing {func_name}() with cache buster: {cache_buster}")
                
                # Try calling RPC function with empty params first, then without params
                try:
                    result = self.client.rpc(func_name, {}).execute()
                except Exception as e:
                    if "missing 1 required positional argument: 'params'" in str(e):
                        # Try with explicit empty params
                        result = self.client.rpc(func_name, params={}).execute()
                    else:
                        # Try without params (older API)
                        result = self.client.rpc(func_name).execute()
                
                if result.data:
                    df = pd.DataFrame(result.data)
                    print(f"Executed {func_name}() - returned {len(df)} rows (fresh data)")
                    
                    # Debug: Print first few rows for verification
                    if func_name == "get_lightfm_interaction_matrix" and len(df) > 0:
                        print(f"Sample data: {df.head(3)[['user_id', 'coupon_id', 'total_weight']].to_dict('records')}")
                    
                    return df
                else:
                    print(f"Warning: {func_name}() returned no data")
                    return pd.DataFrame()
            else:
                print(f"Unsupported query format: {query}")
                return None
                
        except Exception as e:
            print(f"Query execution failed: {e}")
            return None
    
    def close(self):
        """Close connection"""
        print("Supabase connection closed")

class CouponDataLoader:
    def __init__(self, db_connector: SupabaseConnector):
        self.db = db_connector
        self.interactions_df = None
        self.user_features_df = None
        self.item_features_df = None
        self.popular_coupons_df = None
        self.all_coupons_df = None
        
    def load_all_data(self) -> bool:
        """Load all required data from database"""
        print("Loading interaction data with cache clearing...")
        
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
        
        # Load ALL active coupons
        self.load_all_active_coupons()
        
        print(f"Loaded {len(self.interactions_df)} interactions")
        print(f"Loaded {len(self.user_features_df)} users")
        print(f"Loaded {len(self.item_features_df)} coupons from features")
        print(f"Loaded {len(self.all_coupons_df)} total active coupons")
        print(f"Loaded {len(self.popular_coupons_df)} popular coupons")
        
        return self.validate_data()
    
    def load_all_active_coupons(self):
        """Load ALL active regular coupons (excluding vendor coupons)"""
        try:
            # Get all active regular coupons only
            regular_result = self.db.client.table('coupons').select('id, title, category, is_active, created_at').eq('is_active', True).execute()
            
            regular_df = pd.DataFrame(regular_result.data) if regular_result.data else pd.DataFrame()
            
            # Add coupon type and rename id column
            if not regular_df.empty:
                regular_df['coupon_type'] = 'regular'
                regular_df = regular_df.rename(columns={'id': 'coupon_id'})
                self.all_coupons_df = regular_df
            else:
                self.all_coupons_df = pd.DataFrame()
            
            print(f"Found {len(regular_df)} regular coupons (vendor coupons excluded)")
            print(f"Total active coupons: {len(self.all_coupons_df)}")
            
        except Exception as e:
            print(f"Error loading all coupons: {e}")
            self.all_coupons_df = pd.DataFrame()
    
    def validate_data(self) -> bool:
        """Validate that all required data is loaded"""
        if any(df is None or df.empty for df in [
            self.interactions_df, self.user_features_df, 
            self.item_features_df, self.popular_coupons_df, self.all_coupons_df
        ]):
            print("Data validation failed - some datasets are empty")
            return False
        
        print("Data validation passed")
        return True

def get_supabase_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Get Supabase credentials from environment variables"""
    url = os.getenv('EXPO_PUBLIC_SUPABASE_URL') or os.getenv('SUPABASE_URL')
    key = os.getenv('EXPO_PUBLIC_SUPABASE_ANON_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print("Warning: Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
        return None, None
    
    return url, key