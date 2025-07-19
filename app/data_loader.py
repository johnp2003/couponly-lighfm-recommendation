import pandas as pd
from .database import SupabaseConnector

class CouponDataLoader:
    def __init__(self):
        self.db = SupabaseConnector()
        self.interactions_df = None
        self.user_features_df = None
        self.item_features_df = None
        self.popular_coupons_df = None
        
    def refresh_data(self):
        """Load fresh data from database"""
        self.db.connect()
        
        self.interactions_df = self.db.execute_query(
            "SELECT * FROM get_lightfm_interaction_matrix()"
        )
        self.user_features_df = self.db.execute_query(
            "SELECT * FROM get_user_features()"
        )
        self.item_features_df = self.db.execute_query(
            "SELECT * FROM get_coupon_features()"
        )
        self.popular_coupons_df = self.db.execute_query(
            "SELECT * FROM get_popular_coupons()"
        )
        
        return self.validate_data()
    
    def validate_data(self):
        return all(df is not None and not df.empty for df in [
            self.interactions_df, self.user_features_df, 
            self.item_features_df, self.popular_coupons_df
        ])