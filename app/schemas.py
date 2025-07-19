from pydantic import BaseModel

class RecommendationResponse(BaseModel):
    coupon_id: str
    score: float
    category: str
    has_image: bool
    days_until_expiry: int
    view_count: int
    save_count: int
    vote_score: float

class SimilarItemResponse(BaseModel):
    coupon_id: str
    similarity_score: float
    category: str
    view_count: int
    save_count: int