from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth_router import router as auth_router
from filters_router import router as filters_router
from recommendation_router import router as fashion_router
from fashion_analytics_router import router as analytics_router
from fashion_likes_router import router as likes_router

app = FastAPI(title="Fashion recommendation APIs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

app.include_router(auth_router)
app.include_router(filters_router)
app.include_router(fashion_router)
app.include_router(analytics_router)
app.include_router(likes_router)
@app.get("/health")
def health():
    return {"status": "ok"}
