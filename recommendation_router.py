import os
import uuid
import time
import pickle
import numpy as np
import torch
import faiss
import open_clip

from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form
from azure_blob import download_blob, image_url_from_local_path
from config import TOP_K_RESULTS

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cpu"
VECTOR_DIM = 512
STM_TTL = 1800  # 30 minutes

DATA_DIR = "data"
LTM_INDEX_PATH = f"{DATA_DIR}/ltm.index"
LTM_META_PATH = f"{DATA_DIR}/ltm.meta.pkl"

os.makedirs(DATA_DIR, exist_ok=True)

router = APIRouter(
    prefix="/fashion",
    tags=["Fashion Recommendation"]
)

# =====================================================
# GLOBALS (LAZY)
# =====================================================

_model = None
_preprocess = None
_tokenizer = None

# Product index
_product_index = None
_product_meta = None

# STM / LTM
_stm_index = None
_stm_meta = []
_stm_ts = []

_ltm_index = None
_ltm_meta = []

# =====================================================
# LOAD EVERYTHING
# =====================================================

def load_resources():
    global _model, _preprocess, _tokenizer
    global _product_index, _product_meta
    global _stm_index, _ltm_index, _ltm_meta

    if _model is not None:
        return

    print("🔄 Loading CLIP + Product FAISS + Memory...")

    # ---- CLIP ----
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _model.eval().to(DEVICE)
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # ---- PRODUCT FAISS ----
    index_path = download_blob("image_embeddings.faiss")
    meta_path = download_blob("metadata.pkl")

    _product_index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        _product_meta = pickle.load(f)

    # ---- STM ----
    _stm_index = faiss.IndexFlatIP(VECTOR_DIM)

    # ---- LTM ----
    if os.path.exists(LTM_INDEX_PATH):
        _ltm_index = faiss.read_index(LTM_INDEX_PATH)
        with open(LTM_META_PATH, "rb") as f:
            _ltm_meta = pickle.load(f)
    else:
        _ltm_index = faiss.IndexFlatIP(VECTOR_DIM)
        _ltm_meta = []

    print(f"✅ Products loaded: {_product_index.ntotal}")

# =====================================================
# EMBEDDINGS
# =====================================================

def _normalize(x: torch.Tensor) -> np.ndarray:
    x = x / torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-6)
    return x.cpu().numpy().astype("float32")

def encode_text(text: str) -> np.ndarray:
    tokens = _tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        emb = _model.encode_text(tokens)
    return _normalize(emb)

def encode_image(image: Image.Image) -> np.ndarray:
    img = _preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _model.encode_image(img)
    return _normalize(emb)

# =====================================================
# STM / LTM HELPERS
# =====================================================

def stm_cleanup():
    global _stm_index, _stm_meta, _stm_ts
    now = time.time()
    keep = [i for i,t in enumerate(_stm_ts) if now - t < STM_TTL]
    if len(keep) == len(_stm_ts):
        return

    new_idx = faiss.IndexFlatIP(VECTOR_DIM)
    new_meta, new_ts = [], []

    for i in keep:
        new_idx.add(_stm_index.reconstruct(i).reshape(1,-1))
        new_meta.append(_stm_meta[i])
        new_ts.append(_stm_ts[i])

    _stm_index, _stm_meta, _stm_ts = new_idx, new_meta, new_ts

def store_stm(vec, text):
    _stm_index.add(vec)
    _stm_meta.append(text)
    _stm_ts.append(time.time())

def store_ltm(vec, text):
    _ltm_index.add(vec)
    _ltm_meta.append(text)
    faiss.write_index(_ltm_index, LTM_INDEX_PATH)
    with open(LTM_META_PATH, "wb") as f:
        pickle.dump(_ltm_meta, f)

# =====================================================
# PRODUCT RANKING (SAME OUTPUT)
# =====================================================

def rank_products(vec: np.ndarray, k: int):
    k = min(k, _product_index.ntotal)
    scores, ids = _product_index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue

        item = _product_meta[idx].copy()
        item["similarity"] = float(score)
        item["image_url"] = [
            image_url_from_local_path(item["image_path"])
        ]
        results.append(item)

    return results

# =====================================================
# ROUTES (IDENTICAL OUTPUT)
# =====================================================

@router.get("/recommend/text")
def recommend_text(query: str, k: int = TOP_K_RESULTS):
    load_resources()
    stm_cleanup()

    vec = encode_text(query)
    store_stm(vec, query)

    if "remember" in query.lower() or "favorite" in query.lower():
        store_ltm(vec, query)

    return rank_products(vec, k)

@router.post("/recommend/image")
async def recommend_image(
    file: UploadFile = File(...),
    k: int = TOP_K_RESULTS
):
    load_resources()
    stm_cleanup()

    image = Image.open(file.file).convert("RGB")
    vec = encode_image(image)

    store_stm(vec, "image_query")
    return rank_products(vec, k)

@router.post("/recommend/hybrid")
async def recommend_hybrid(
    file: UploadFile = File(...),
    query: str = Form(...),
    k: int = TOP_K_RESULTS
):
    load_resources()
    stm_cleanup()

    image = Image.open(file.file).convert("RGB")
    iv = encode_image(image)
    tv = encode_text(query)

    vec = (iv + tv) / 2
    vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-6, None)

    store_stm(vec, query)
    if "remember" in query.lower():
        store_ltm(vec, query)

    return rank_products(vec.astype("float32"), k)
