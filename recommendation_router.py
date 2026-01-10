import torch
import faiss
import numpy as np
import pickle
import open_clip
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form

from azure_blob import download_blob, image_url_from_local_path
from config import TOP_K_RESULTS

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cpu"

router = APIRouter(
    prefix="/fashion",
    tags=["Fashion Recommendation"]
)

# =====================================================
# GLOBALS (LAZY-LOADED)
# =====================================================

_model = None
_preprocess = None
_tokenizer = None
_index = None
_metadata = None

# =====================================================
# LAZY LOADER
# =====================================================

def load_resources():
    global _model, _preprocess, _tokenizer, _index, _metadata

    if _model is not None:
        return

    print("🔄 Lazy loading CLIP + FAISS...")

    # ---- OpenCLIP ----
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="openai"
    )
    _model = _model.to(DEVICE)
    _model.eval()

    _tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # ---- FAISS + METADATA ----
    index_path = download_blob("image_embeddings.faiss")
    meta_path = download_blob("metadata.pkl")

    _index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        _metadata = pickle.load(f)

    if _index.ntotal != len(_metadata):
        raise RuntimeError(
            f"FAISS index ({_index.ntotal}) != metadata ({len(_metadata)})"
        )

    print(f"✅ Loaded {_index.ntotal} embeddings")

# =====================================================
# EMBEDDING HELPERS
# =====================================================

def _normalize(vec: torch.Tensor) -> np.ndarray:
    norm = vec.norm(dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-6)
    vec = vec / norm
    return vec.cpu().numpy().astype("float32")

def _encode_text(text: str) -> np.ndarray:
    tokens = _tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        emb = _model.encode_text(tokens)
    return _normalize(emb)

def _encode_image(image: Image.Image) -> np.ndarray:
    img = _preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = _model.encode_image(img)
    return _normalize(emb)

# =====================================================
# RANKING (SAFE)
# =====================================================

def _rank(vec: np.ndarray, k: int):
    # 🚨 No data in index
    if _index.ntotal == 0:
        return []

    k = min(k, _index.ntotal)

    scores, ids = _index.search(vec, k)
    results = []

    for score, idx in zip(scores[0], ids[0]):
        # 🚨 FAISS uses -1 when no match
        if idx < 0:
            continue

        item = _metadata[idx].copy()
        item["similarity"] = float(score)

        # Azure Blob URL
        item["image_url"] = [
            image_url_from_local_path(item["image_path"])
        ]

        results.append(item)

    return results

# =====================================================
# ROUTES
# =====================================================

@router.get("/recommend/text")
def recommend_text(query: str, k: int = TOP_K_RESULTS):
    load_resources()

    if _index.ntotal == 0:
        return []

    vec = _encode_text(query)
    return _rank(vec, k)

@router.post("/recommend/image")
async def recommend_image(
    file: UploadFile = File(...),
    k: int = TOP_K_RESULTS
):
    load_resources()

    if _index.ntotal == 0:
        return []

    image = Image.open(file.file).convert("RGB")
    vec = _encode_image(image)
    return _rank(vec, k)

@router.post("/recommend/hybrid")
async def recommend_hybrid(
    file: UploadFile = File(...),
    query: str = Form(...),
    k: int = TOP_K_RESULTS
):
    load_resources()

    if _index.ntotal == 0:
        return []

    image = Image.open(file.file).convert("RGB")

    iv = _encode_image(image)
    tv = _encode_text(query)

    vec = (iv + tv) / 2

    # Safe normalization
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm = np.clip(norm, a_min=1e-6, a_max=None)
    vec = vec / norm

    return _rank(vec.astype("float32"), k)
