from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth_router import router as auth_router
from filters_router import router as filters_router

app = FastAPI(title="Fashion recommendation APIs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

app.include_router(auth_router)
app.include_router(filters_router)

@app.get("/health")
def health():
    return {"status": "ok"}
