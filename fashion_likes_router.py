import os
from fastapi import APIRouter, HTTPException
from azure.data.tables import TableServiceClient
from datetime import datetime, timezone

from azure_blob import image_url_from_local_path

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# =====================================================
# CONFIG
# =====================================================

PRODUCT_LIKES_TABLE = "ProductLikes"
USER_LIKES_TABLE = "UserLikes"

router = APIRouter(
    prefix="/fashion/likes",
    tags=["Fashion Likes"]
)

# =====================================================
# AZURE TABLE CLIENTS
# =====================================================

service = TableServiceClient.from_connection_string(
    CONNECTION_STRING
)

service.create_table_if_not_exists(PRODUCT_LIKES_TABLE)
service.create_table_if_not_exists(USER_LIKES_TABLE)

product_likes_table = service.get_table_client(PRODUCT_LIKES_TABLE)
user_likes_table = service.get_table_client(USER_LIKES_TABLE)

# =====================================================
# UTILITIES
# =====================================================

def iso_now():
    return datetime.now(timezone.utc).isoformat()

# =====================================================
# LIKE PRODUCT
# =====================================================

@router.post("/{user_id}/{product_id}")
def like_product(user_id: str, product_id: str, image_path: str):
    """
    User likes a product
    """

    # 1️⃣ Prevent duplicate likes
    try:
        user_likes_table.get_entity(user_id, product_id)
        return {"status": "already_liked"}
    except:
        pass

    # 2️⃣ Add to UserLikes
    user_likes_table.create_entity({
        "PartitionKey": user_id,
        "RowKey": product_id,
        "liked_at": iso_now(),
        "image_path": image_path
    })

    # 3️⃣ Increment ProductLikes
    try:
        e = product_likes_table.get_entity("PRODUCT", product_id)
        e["like_count"] += 1
        product_likes_table.update_entity(e, mode="replace")
    except:
        product_likes_table.create_entity({
            "PartitionKey": "PRODUCT",
            "RowKey": product_id,
            "like_count": 1
        })

    return {"status": "liked"}

# =====================================================
# UNLIKE PRODUCT
# =====================================================

@router.delete("/{user_id}/{product_id}")
def unlike_product(user_id: str, product_id: str):
    """
    User removes a like
    """

    # 1️⃣ Remove from UserLikes
    try:
        user_likes_table.delete_entity(user_id, product_id)
    except:
        raise HTTPException(status_code=404, detail="Like not found")

    # 2️⃣ Decrement ProductLikes
    try:
        e = product_likes_table.get_entity("PRODUCT", product_id)
        e["like_count"] = max(0, e["like_count"] - 1)
        product_likes_table.update_entity(e, mode="replace")
    except:
        pass

    return {"status": "unliked"}

# =====================================================
# GET MY LIKED PRODUCTS
# =====================================================

@router.get("/{user_id}")
def get_my_likes(user_id: str, limit: int = 50):
    """
    Get all products liked by a user
    """

    entities = user_likes_table.query_entities(
        f"PartitionKey eq '{user_id}'"
    )

    items = sorted(
        entities,
        key=lambda e: e["liked_at"],
        reverse=True
    )[:limit]

    return [
        {
            "product_id": e["RowKey"],
            "liked_at": e["liked_at"],
            "image_url": image_url_from_local_path(e["image_path"])
        }
        for e in items
    ]

# =====================================================
# GET LIKE COUNT FOR A PRODUCT
# =====================================================

@router.get("/count/{product_id}")
def get_like_count(product_id: str):
    """
    Get total likes for a product
    """
    try:
        e = product_likes_table.get_entity("PRODUCT", product_id)
        return {
            "product_id": product_id,
            "like_count": e["like_count"]
        }
    except:
        return {
            "product_id": product_id,
            "like_count": 0
        }
