"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    fresh_coupon_data: bool = Field(default=True, description="Whether to fetch fresh coupon data")

class CouponRecommendation(BaseModel):
    coupon_id: str
    score: float
    category: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    discount_percentage: Optional[float] = None
    expires_at: Optional[str] = None
    has_image: Optional[bool] = None
    days_until_expiry: Optional[int] = None
    view_count: Optional[int] = None
    save_count: Optional[int] = None
    vote_score: Optional[float] = None
    coupon_type: Optional[str] = None
    fresh_data: Optional[bool] = None

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[CouponRecommendation]
    model_last_trained: Optional[str] = None
    total_active_coupons: Optional[int] = None
    timestamp: str

class SimilarItemsRequest(BaseModel):
    coupon_id: str = Field(..., description="Coupon ID to find similar items for")
    num_similar: int = Field(default=5, ge=1, le=20, description="Number of similar items to return")

class SimilarItem(BaseModel):
    coupon_id: str
    similarity_score: float
    category: Optional[str] = None
    title: Optional[str] = None
    view_count: Optional[int] = None
    save_count: Optional[int] = None
    coupon_type: Optional[str] = None

class SimilarItemsResponse(BaseModel):
    coupon_id: str
    similar_items: List[SimilarItem]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class ModelStatusResponse(BaseModel):
    is_trained: bool
    last_trained: Optional[str] = None
    total_users: Optional[int] = None
    total_coupons: Optional[int] = None
    total_interactions: Optional[int] = None
    model_version: Optional[str] = None
    cache_status: str
    timestamp: str

class ModelMetrics(BaseModel):
    train_precision: Optional[float] = None
    test_precision: Optional[float] = None
    train_auc: Optional[float] = None
    test_auc: Optional[float] = None
    timestamp: str