import torch
import clip
import faiss
import pandas as pd
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime

from azure_blob import download_blob
from config import *

DEVICE = "cpu"

router = APIRouter(
    prefix="/fashion",
    tags=["Fashion Recommendation"]
)

# =====================================================
#  GLOBALS (start empty – VERY IMPORTANT)
# =====================================================

model = None
preprocess = None
df = None
index = None

# =====================================================
#  LAZY LOADER
# =====================================================

def load_resources():
    global model, preprocess, df, index

    # Prevent double-loading
    if model is not None:
        return

    print("🔄 Loading ML resources...")

    # ---- CLIP ----
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    # ---- Metadata ----
    # df = pd.read_csv(download_blob(METADATA_BLOB))
    # df["release_date"] = pd.to_datetime(df["release_date"])

    # ---- FAISS ----
    # index = faiss.read_index(
    #     download_blob(f"{ARTIFACT_PREFIX}faiss.index")
    # )

    # ---- Scoring ----
    # def normalize(x):
    #     return (x - x.min()) / (x.max() - x.min() + 1e-6)

    # df["trend_score"] = normalize(df["popularity"])
    # df["recency_days"] = (datetime.now() - df["release_date"]).dt.days
    # df["recency_score"] = 1 - normalize(df["recency_days"])

    print("ML resources loaded")

# =====================================================
#  HELPERS
# =====================================================

# def rank(vec):
#     scores, ids = index.search(vec, TOP_K_CANDIDATES)
#     c = df.iloc[ids[0]].copy()
#     c["similarity"] = scores[0]

#     c["final_score"] = (
#         SIM_WEIGHT * c["similarity"]
#         + TREND_WEIGHT * c["trend_score"]
#         + RECENCY_WEIGHT * c["recency_score"]
#     )

#     return (
#         c.sort_values("final_score", ascending=False)
#          .head(TOP_K_RESULTS)
#          .to_dict("records")
#     )

# =====================================================
#  ROUTES
# =====================================================

@router.get("/trending")
def trending():
    load_resources()
    return (
        df.sort_values("trend_score", ascending=False)
          .head(TOP_K_RESULTS)
          .to_dict("records")
    )


@router.get("/latest")
def latest():
    load_resources()
    return (
        df.sort_values("recency_score", ascending=False)
          .head(TOP_K_RESULTS)
          .to_dict("records")
    )


@router.get("/recommend/text")
def recommend_text(query: str):
    load_resources()

    tokens = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        vec = model.encode_text(tokens)

    vec /= vec.norm(dim=-1, keepdim=True)
    return rank(vec.cpu().numpy().astype("float32"))


@router.post("/recommend/image")
async def recommend_image(file: UploadFile = File(...)):
    load_resources()

    image = preprocess(
        Image.open(file.file).convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        vec = model.encode_image(image)

    vec /= vec.norm(dim=-1, keepdim=True)
    return rank(vec.cpu().numpy().astype("float32"))


@router.post("/recommend/hybrid")
async def recommend_hybrid(
    file: UploadFile = File(...),
    query: str = ""
):
    load_resources()

    image = preprocess(
        Image.open(file.file).convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    tokens = clip.tokenize([query]).to(DEVICE)

    with torch.no_grad():
        iv = model.encode_image(image)
        tv = model.encode_text(tokens)

    iv /= iv.norm(dim=-1, keepdim=True)
    tv /= tv.norm(dim=-1, keepdim=True)

    vec = (iv + tv) / 2
    return rank(vec.cpu().numpy().astype("float32"))
