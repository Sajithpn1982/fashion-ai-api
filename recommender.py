import torch
import clip
import faiss
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
import uvicorn
import os

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIM_WEIGHT = 0.6
TREND_WEIGHT = 0.25
RECENCY_WEIGHT = 0.15

TOP_K_CANDIDATES = 20
TOP_K_RESULTS = 5

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Fashion Recommendation API")

# =====================================================
# LOAD MODEL
# =====================================================
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(META_PATH)
df["release_date"] = pd.to_datetime(df["release_date"])

# =====================================================
# SCORING FEATURES
# =====================================================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

df["trend_score"] = normalize(df["popularity"])
df["recency_days"] = (datetime.now() - df["release_date"]).dt.days
df["recency_score"] = 1 - normalize(df["recency_days"])

# =====================================================
# BUILD EMBEDDINGS
# =====================================================
def build_embeddings():
    vectors = []
    for _, row in df.iterrows():
        image = preprocess(
            Image.open(os.path.join(IMAGE_DIR, row["image"]))
        ).unsqueeze(0).to(DEVICE)

        text = clip.tokenize([row["description"]]).to(DEVICE)

        with torch.no_grad():
            img_vec = model.encode_image(image)
            txt_vec = model.encode_text(text)

        img_vec /= img_vec.norm(dim=-1, keepdim=True)
        txt_vec /= txt_vec.norm(dim=-1, keepdim=True)

        combined = (img_vec + txt_vec) / 2
        vectors.append(combined.cpu().numpy()[0])

    return np.array(vectors).astype("float32")

# =====================================================
# FAISS INDEX
# =====================================================
embeddings = build_embeddings()
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# =====================================================
# RANKING
# =====================================================
def faiss_candidates(query_vec):
    scores, ids = index.search(query_vec, TOP_K_CANDIDATES)
    candidates = df.iloc[ids[0]].copy()
    candidates["similarity"] = scores[0]
    return candidates

def rank_results(candidates):
    candidates["final_score"] = (
        SIM_WEIGHT * candidates["similarity"] +
        TREND_WEIGHT * candidates["trend_score"] +
        RECENCY_WEIGHT * candidates["recency_score"]
    )
    return candidates.sort_values(
        "final_score", ascending=False
    ).head(TOP_K_RESULTS)

# =====================================================
# API ROUTES
# =====================================================

@app.get("/trending")
def get_trending():
    results = df.sort_values(
        "trend_score", ascending=False
    ).head(TOP_K_RESULTS)
    return results.to_dict(orient="records")


@app.get("/latest")
def get_latest():
    results = df.sort_values(
        "recency_score", ascending=False
    ).head(TOP_K_RESULTS)
    return results.to_dict(orient="records")


@app.get("/recommend/text")
def recommend_text(query: str):
    tokens = clip.tokenize([query]).to(DEVICE)

    with torch.no_grad():
        query_vec = model.encode_text(tokens)

    query_vec /= query_vec.norm(dim=-1, keepdim=True)
    query_vec = query_vec.cpu().numpy().astype("float32")

    candidates = faiss_candidates(query_vec)
    ranked = rank_results(candidates)

    return ranked[[
        "description",
        "brand",
        "price",
        "final_score"
    ]].to_dict(orient="records")


@app.post("/recommend/image")
async def recommend_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_vec = model.encode_image(image)

    query_vec /= query_vec.norm(dim=-1, keepdim=True)
    query_vec = query_vec.cpu().numpy().astype("float32")

    candidates = faiss_candidates(query_vec)
    ranked = rank_results(candidates)

    return ranked[[
        "description",
        "brand",
        "price",
        "final_score"
    ]].to_dict(orient="records")

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
