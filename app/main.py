import os
import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from .database import SupabaseConnector
from .data_loader import CouponDataLoader
from .model import LightFMRecommendationSystem
from .schemas import RecommendationResponse, SimilarItemResponse

# Global model and training state
model_instance = None
training_lock = threading.Lock()
last_trained = None

def initialize_system():
    """Initialize the recommendation system"""
    global model_system
    data_loader = CouponDataLoader()
    if data_loader.refresh_data():
        model_system = LightFMRecommendationSystem(data_loader)
        model_system.prepare_dataset()
        model_system.build_interaction_matrix()
        model_system.build_feature_matrices()
        model_system.train_model(epochs=50)
        return True
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup"""
    initialize_model()
    yield
    
    # Cleanup on shutdown
    if model_instance and model_instance.data_loader.db:
        model_instance.data_loader.db.close()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_ready": model_instance is not None,
        "last_trained": last_trained
    }

@app.get("/recommendations/{user_id}", response_model=list[RecommendationResponse])
def get_recommendations(user_id: str, num_recs: int = 10):
    if not model_instance:
        raise HTTPException(503, "Model not initialized")
    
    try:
        return model_instance.get_recommendations(user_id, num_recs)
    except Exception as e:
        raise HTTPException(500, f"Recommendation failed: {str(e)}")

@app.get("/similar/{coupon_id}", response_model=list[SimilarItemResponse])
def get_similar_items(coupon_id: str, num_similar: int = 5):
    if not model_instance:
        raise HTTPException(503, "Model not initialized")
    
    try:
        return model_instance.get_similar_items(coupon_id, num_similar)
    except Exception as e:
        raise HTTPException(500, f"Similar items failed: {str(e)}")

@app.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining in background"""
    if model_instance and model_instance.data_loader.db:
        model_instance.data_loader.db.close()
    
    background_tasks.add_task(initialize_model)
    return {"message": "Retraining started in background", "status": "processing"}

@app.post("/refresh")
def refresh_recommendations():
    """Refresh data and retrain model"""
    if not model_system:
        raise HTTPException(503, "System not initialized")
    
    try:
        success = model_system.refresh_and_retrain()
        return {"status": "success" if success else "failed"}
    except Exception as e:
        raise HTTPException(500, f"Refresh failed: {str(e)}")

@app.get("/training-status")
def training_status():
    return {
        "status": "ready" if model_instance else "initializing",
        "last_trained": last_trained
    }