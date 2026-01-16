from typing import Optional
import os
import json

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
)

from PIL import Image
import numpy as np
import faiss
import pickle

from openai import OpenAI

from auth import get_current_user, get_thread_id
from filters import parse_filters
from clip_model import load_model, encode_text, encode_image
from memory import (
    store_stm,
    stm_cleanup,
    latest_vec,
    _stm_index,
    _stm_text,
)
from config import TOP_K_RESULTS
from azure_blob import image_url_from_local_path, download_blob

# -------------------------------------------------
# OpenAI client
# -------------------------------------------------

_openai_client = OpenAI(
    api_key=""  # os.getenv("OPENAI_API_KEY")
)

# -------------------------------------------------
# Router
# -------------------------------------------------

router = APIRouter(
    prefix="/fashionrec",
    tags=["Fashion Recommendation"],
)

# -------------------------------------------------
# Product index (kept local on purpose)
# -------------------------------------------------

_product_index = None
_product_meta = None


def load_products():
    global _product_index, _product_meta
    if _product_index is not None:
        return

    idx_path = download_blob("image_embeddings.faiss")
    meta_path = download_blob("metadata.pkl")

    _product_index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        _product_meta = pickle.load(f)


# -------------------------------------------------
# Filter mask (NEW, SAFE)
# -------------------------------------------------

def build_id_mask(filters_dict):
    if not filters_dict:
        return None

    mask = np.zeros(len(_product_meta), dtype=bool)

    for idx, meta in enumerate(_product_meta):
        keep = True
        for k, v in filters_dict.items():
            if meta.get(k) != v:
                keep = False
                break
        mask[idx] = keep

    return mask


# -------------------------------------------------
# FAISS search (SAFE FOR ALL BUILDS)
# -------------------------------------------------

def rank_products(vec, k, id_mask=None):
    search_k = min(k * 5, _product_index.ntotal)

    scores, ids = _product_index.search(vec, search_k)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        if id_mask is not None and not id_mask[i]:
            continue

        item = _product_meta[i].copy()
        item["similarity"] = float(s)
        item["image_url"] = image_url_from_local_path(
            item["image_path"]
        )
        results.append(item)

        if len(results) >= k:
            break

    return results


def build_query_vector(input_vec, stm_vec):
    vec = 0.8 * input_vec
    if stm_vec is not None:
        vec += 0.2 * stm_vec

    vec /= np.clip(
        np.linalg.norm(vec, axis=1, keepdims=True), 1e-6, None
    )
    return vec.astype("float32")


# -------------------------------------------------
# STM diagnostics builder
# -------------------------------------------------

def build_stm_summary(
    had_stm_before: bool,
    cleanup_performed: bool,
    intent: Optional[dict] = None,
):
    return {
        "used_for_query": had_stm_before,
        "stored_this_turn": intent["store_stm"] if intent else True,
        "vectors_before": 1 if had_stm_before else 0,
        "vectors_after": _stm_index.ntotal,
        "cleanup_performed": cleanup_performed,
        "latest_text": _stm_text[-1] if _stm_text else None,
        "text_history": list(_stm_text),
        "blend_weight": 0.2,
        "llm_reason": intent.get("reason") if intent else None,
    }


# -------------------------------------------------
# OpenAI STM intent classifier
# -------------------------------------------------

def classify_memory_intent(query: str) -> dict:
    prompt = f"""
You are a memory controller for an AI fashion assistant.

Rules:
- STM: short-term conversational context (searches, refinements)
- Greetings, confirmations, filler → do NOT store

Return ONLY valid JSON:
{{
  "store_stm": true | false,
  "reason": "short explanation"
}}

Message:
\"\"\"{query}\"\"\"
"""

    try:
        response = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "store_stm": True,
            "reason": f"fallback_no_llm: {str(e)}",
        }


# -------------------------------------------------
# Routes
# -------------------------------------------------

@router.post("/recommend/text")
def recommend_text(
    query: Optional[str] = Form(None),
    filters: Optional[str] = Form(None),
    k: int = TOP_K_RESULTS,
    user_id: str = Depends(get_current_user),
    thread_id: str = Depends(get_thread_id),
):
    filters_dict = parse_filters(filters)

    load_model()
    load_products()

    had_stm_before = latest_vec() is not None
    cleanup_performed = stm_cleanup()

    stm_vec = latest_vec()

    if query:
        iv = encode_text(query)
        qv = build_query_vector(iv, stm_vec)

        intent = classify_memory_intent(query)
        if intent["store_stm"]:
            store_stm(iv, query)
    else:
        if stm_vec is None:
            return {
                "thread_id": thread_id,
                "results": [],
                "error": "query is required when no STM context exists",
            }

        qv = stm_vec.astype("float32")
        intent = {
            "store_stm": False,
            "reason": "no_query_provided_used_stm_only",
        }

    id_mask = build_id_mask(filters_dict)
    results = rank_products(qv, k, id_mask=id_mask)

    return {
        "thread_id": thread_id,
        "results": results,
        "memory": {
            "stm": build_stm_summary(
                had_stm_before=had_stm_before,
                cleanup_performed=cleanup_performed,
                intent=intent,
            ),
            "ltm": {
                "used": False,
                "stored_this_turn": False,
                "reason": "ltm_not_enabled_in_this_router",
            },
        },
    }


@router.post("/recommend/image")
async def recommend_image(
    file: UploadFile = File(...),
    filters: Optional[str] = Form(None),
    k: int = TOP_K_RESULTS,
    user_id: str = Depends(get_current_user),
    thread_id: str = Depends(get_thread_id),
):
    filters_dict = parse_filters(filters)

    load_model()
    load_products()

    had_stm_before = latest_vec() is not None
    cleanup_performed = stm_cleanup()

    img = Image.open(file.file).convert("RGB")
    iv = encode_image(img)
    stm_vec = latest_vec()
    qv = build_query_vector(iv, stm_vec)

    store_stm(iv, None)

    id_mask = build_id_mask(filters_dict)
    results = rank_products(qv, k, id_mask=id_mask)

    return {
        "thread_id": thread_id,
        "results": results,
        "memory": {
            "stm": build_stm_summary(
                had_stm_before=had_stm_before,
                cleanup_performed=cleanup_performed,
                intent={"store_stm": True, "reason": "image_query"},
            ),
            "ltm": {
                "used": False,
                "stored_this_turn": False,
                "reason": "ltm_not_enabled_in_this_router",
            },
        },
    }


@router.post("/recommend/hybrid")
async def recommend_hybrid(
    query: str = Form(...),
    file: UploadFile = File(...),
    filters: Optional[str] = Form(None),
    k: int = TOP_K_RESULTS,
    user_id: str = Depends(get_current_user),
    thread_id: str = Depends(get_thread_id),
):
    filters_dict = parse_filters(filters)

    load_model()
    load_products()

    had_stm_before = latest_vec() is not None
    cleanup_performed = stm_cleanup()

    img = Image.open(file.file).convert("RGB")
    iv = encode_image(img)
    tv = encode_text(query)

    input_vec = (iv + tv) / 2
    input_vec /= np.clip(
        np.linalg.norm(input_vec, axis=1, keepdims=True), 1e-6, None
    )

    stm_vec = latest_vec()
    qv = build_query_vector(input_vec, stm_vec)

    intent = classify_memory_intent(query)
    if intent["store_stm"]:
        store_stm(input_vec, query)

    id_mask = build_id_mask(filters_dict)
    results = rank_products(qv, k, id_mask=id_mask)

    return {
        "thread_id": thread_id,
        "results": results,
        "memory": {
            "stm": build_stm_summary(
                had_stm_before=had_stm_before,
                cleanup_performed=cleanup_performed,
                intent=intent,
            ),
            "ltm": {
                "used": False,
                "stored_this_turn": False,
                "reason": "ltm_not_enabled_in_this_router",
            },
        },
    }
