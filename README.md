# LightFM Coupon Recommendation API

A production-ready FastAPI application for coupon recommendations using LightFM collaborative filtering.

## Features

- **Personalized Recommendations**: Get tailored coupon recommendations for users
- **Similar Items**: Find similar coupons based on user behavior
- **Model Caching**: Intelligent model caching with automatic retraining
- **Production Ready**: Optimized for deployment on Render
- **Health Monitoring**: Built-in health checks and metrics

## API Endpoints

### Core Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /recommendations` - Get user recommendations
- `POST /similar-items` - Get similar coupons
- `GET /model/status` - Model status and statistics
- `GET /model/metrics` - Model performance metrics
- `POST /model/retrain` - Force model retraining (admin)

### API Documentation

- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Quick Start

### Local Development

1. **Clone and setup**:

   ```bash
   cd fastapi
   pip install -r requirements.txt
   ```

2. **Environment setup**:

   ```bash
   cp .env.example .env
   # Edit .env with your Supabase credentials
   ```

3. **Run the server**:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Test the API**:
   ```bash
   curl http://localhost:8000/health
   ```

### Example Usage

**Get Recommendations**:

```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_2yqKXgsH3jEQivtjwrJyLURqHuX",
    "num_recommendations": 5,
    "fresh_coupon_data": true
  }'
```

**Get Similar Items**:

```bash
curl -X POST "http://localhost:8000/similar-items" \
  -H "Content-Type: application/json" \
  -d '{
    "coupon_id": "some-coupon-id",
    "num_similar": 3
  }'
```

## Deployment on Render

### Method 1: Using render.yaml (Recommended)

1. **Push to GitHub**:

   ```bash
   git add .
   git commit -m "Add FastAPI recommendation system"
   git push origin main
   ```

2. **Deploy on Render**:

   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`

3. **Set Environment Variables**:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_ANON_KEY`: Your Supabase anonymous key

### Method 2: Manual Web Service

1. **Create Web Service**:

   - Go to Render Dashboard
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Set root directory to `fastapi`

2. **Configure Build & Start**:

   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**:
   - Add `SUPABASE_URL` and `SUPABASE_ANON_KEY`

### Method 3: Docker Deployment

```bash
# Build the image
docker build -t lightfm-api .

# Run locally
docker run -p 8000:8000 \
  -e SUPABASE_URL=your-url \
  -e SUPABASE_ANON_KEY=your-key \
  lightfm-api
```

## Configuration

### Environment Variables

| Variable                  | Description                 | Required |
| ------------------------- | --------------------------- | -------- |
| `SUPABASE_URL`            | Supabase project URL        | Yes      |
| `SUPABASE_ANON_KEY`       | Supabase anonymous key      | Yes      |
| `PORT`                    | Server port (default: 8000) | No       |
| `RETRAIN_INTERVAL_HOURS`  | Model retrain interval      | No       |
| `FORCE_RETRAIN_THRESHOLD` | Force retrain threshold     | No       |

### Model Configuration

The system automatically:

- Loads cached models on startup
- Retrains every 6 hours
- Handles cold-start users with popular coupons
- Includes ALL active coupons in recommendations

## Architecture

```
fastapi/
├── main.py                 # FastAPI application entry point
├── app/
│   ├── models/
│   │   └── schemas.py      # Pydantic models
│   └── services/
│       ├── database_service.py      # Supabase connection
│       ├── lightfm_service.py       # LightFM model
│       └── recommendation_service.py # Main service
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── render.yaml            # Render deployment config
└── README.md              # This file
```

## Performance

- **Cold Start**: ~2-3 seconds (cached model)
- **Recommendations**: ~100-200ms per request
- **Model Training**: ~30-60 seconds
- **Memory Usage**: ~200-500MB
- **Concurrent Users**: 100+ (with proper scaling)

## Monitoring

### Health Checks

- `GET /health` - Service health status
- `GET /model/status` - Model training status
- `GET /model/metrics` - Performance metrics

### Logs

The application provides detailed logging:

- Model training progress
- Cache hit/miss rates
- Database connection status
- Error tracking

## Troubleshooting

### Common Issues

1. **Model Training Fails**:

   - Check Supabase credentials
   - Verify database functions exist
   - Check data availability

2. **Slow Responses**:

   - Model may need retraining
   - Check database connection
   - Monitor memory usage

3. **Cache Issues**:
   - Clear model cache: `rm -rf model_cache/*`
   - Force retrain: `POST /model/retrain`

### Debug Mode

Run with debug logging:

```bash
uvicorn main:app --log-level debug
```

## Production Considerations

1. **Scaling**: Use Render's auto-scaling features
2. **Monitoring**: Set up alerts for health endpoints
3. **Caching**: Model cache persists across deployments
4. **Security**: Implement API authentication if needed
5. **Rate Limiting**: Consider adding rate limiting for production

## Support

For issues and questions:

1. Check the logs for detailed error messages
2. Verify environment variables are set correctly
3. Test database connectivity with `/health` endpoint
4. Monitor model status with `/model/status`

## License

This project is part of the Couponly application.
