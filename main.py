"""
FastAPI LightFM Coupon Recommendation System
Production-ready API for coupon recommendations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
from datetime import datetime

from app.services.recommendation_service import RecommendationService
from app.models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    SimilarItemsRequest,
    SimilarItemsResponse,
    HealthResponse,
    ModelStatusResponse,
    PopularCouponsRequest,
    PopularCouponsResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="LightFM Coupon Recommendation API",
    description="Production-ready coupon recommendation system using LightFM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation service
recommendation_service = RecommendationService()

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup"""
    print("🚀 Starting LightFM Recommendation API...")
    await recommendation_service.initialize()
    print("✅ Recommendation service initialized")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="LightFM Coupon Recommendation API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    is_healthy = await recommendation_service.health_check()
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        message="Service is operational" if is_healthy else "Service has issues",
        timestamp=datetime.now().isoformat()
    )

@app.head("/health")
async def health_check_head():
    """Health check for HEAD requests (no body returned)"""
    is_healthy = await recommendation_service.health_check()
    # Return 200 if healthy, 503 if not
    if not is_healthy:
        return Response(status_code=503)
    return

@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get current model status and statistics"""
    status = await recommendation_service.get_model_status()
    return ModelStatusResponse(**status)

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized coupon recommendations for a user"""
    try:
        recommendations = await recommendation_service.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            fresh_coupon_data=request.fresh_coupon_data,
            categories=request.categories,
            exclude_categories=request.exclude_categories
        )
        
        return RecommendationResponse(**recommendations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@app.post("/similar-items", response_model=SimilarItemsResponse)
async def get_similar_items(request: SimilarItemsRequest):
    """Get similar coupons for a given coupon"""
    try:
        similar_items = await recommendation_service.get_similar_items(
            coupon_id=request.coupon_id,
            num_similar=request.num_similar
        )
        
        return SimilarItemsResponse(
            coupon_id=request.coupon_id,
            similar_items=similar_items,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar items generation failed: {str(e)}")

@app.post("/model/retrain")
async def force_retrain(background_tasks: BackgroundTasks):
    """Force retrain the model (admin endpoint)"""
    try:
        background_tasks.add_task(recommendation_service.force_retrain)
        return {
            "message": "Model retraining initiated in background",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain initiation failed: {str(e)}")

@app.get("/model/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    try:
        metrics = await recommendation_service.get_model_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.post("/popular-coupons", response_model=PopularCouponsResponse)
async def get_popular_coupons(request: PopularCouponsRequest):
    """Get popular/trending coupons"""
    try:
        popular_coupons = await recommendation_service.get_popular_coupons(
            limit=request.limit,
            category=request.category
        )
        
        if "error" in popular_coupons:
            raise HTTPException(status_code=500, detail=popular_coupons["error"])
        
        return PopularCouponsResponse(**popular_coupons)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Popular coupons retrieval failed: {str(e)}")

@app.get("/popular-coupons", response_model=PopularCouponsResponse)
async def get_popular_coupons_get(
    limit: int = 20,
    category: Optional[str] = None
):
    """Get popular/trending coupons (GET endpoint)"""
    try:
        popular_coupons = await recommendation_service.get_popular_coupons(
            limit=limit,
            category=category
        )
        
        if "error" in popular_coupons:
            raise HTTPException(status_code=500, detail=popular_coupons["error"])
        
        return PopularCouponsResponse(**popular_coupons)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Popular coupons retrieval failed: {str(e)}")

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations_get(
    user_id: str,
    num_recommendations: int = 10,
    fresh_coupon_data: bool = True,
    categories: Optional[str] = None,
    exclude_categories: Optional[str] = None
):
    """Get personalized coupon recommendations for a user (GET endpoint)"""
    try:
        # Parse comma-separated categories
        categories_list = [cat.strip() for cat in categories.split(',')] if categories else None
        exclude_categories_list = [cat.strip() for cat in exclude_categories.split(',')] if exclude_categories else None
        
        recommendations = await recommendation_service.get_recommendations(
            user_id=user_id,
            num_recommendations=num_recommendations,
            fresh_coupon_data=fresh_coupon_data,
            categories=categories_list,
            exclude_categories=exclude_categories_list
        )
        
        return RecommendationResponse(**recommendations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@app.get("/categories")
async def get_available_categories():
    """Get list of available coupon categories"""
    return {
        "categories": [
            "Food & Drink",
            "Fashion", 
            "Tech",
            "Beauty",
            "Home & Living",
            "Travel",
            "E-commerce"
        ],
        "description": "Available categories for filtering recommendations",
        "usage": "Use these category names (case-insensitive) in your requests",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )