import os
from fastapi import APIRouter
from datetime import datetime, timezone
from azure.data.tables import TableServiceClient
import math

from azure_blob import image_url_from_local_path
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# =====================================================
# CONFIG
# =====================================================

TABLE_NAME = "ProductAnalytics"

router = APIRouter(
    prefix="/analytics",
    tags=["Fashion Analytics"]
)

# =====================================================
# AZURE TABLE STORAGE CLIENT
# =====================================================

table_service = TableServiceClient.from_connection_string(
    CONNECTION_STRING
)

table = table_service.get_table_client(TABLE_NAME)
table.create_table_if_not_exists()

# =====================================================
# UTILITIES
# =====================================================

def utc_now():
    return datetime.now(timezone.utc)

def iso_now():
    return utc_now().isoformat()

def hours_since(iso_time: str) -> float:
    past = datetime.fromisoformat(iso_time)
    return (utc_now() - past).total_seconds() / 3600

# =====================================================
# VIEW TRACKING
# =====================================================

@router.post("/view/{product_id}")
def record_view(product_id: str, image_path: str):
    """
    Call this when a user views or clicks a product
    """
    try:
        entity = table.get_entity("PRODUCT", product_id)
        entity["view_count"] += 1
        entity["last_viewed_at"] = iso_now()
        table.update_entity(entity, mode="replace")

    except Exception:
        table.create_entity({
            "PartitionKey": "PRODUCT",
            "RowKey": product_id,
            "image_path": image_path,
            "view_count": 1,
            "created_at": iso_now(),
            "last_viewed_at": iso_now()
        })

    return {"status": "ok"}

# =====================================================
# MOST VIEWED PRODUCTS
# =====================================================

@router.get("/most-viewed")
def most_viewed(limit: int = 20):
    entities = table.query_entities(
        "PartitionKey eq 'PRODUCT'"
    )

    items = sorted(
        entities,
        key=lambda e: e.get("view_count", 0),
        reverse=True
    )[:limit]

    return [
        {
            "product_id": e["RowKey"],
            "view_count": e["view_count"],
            "image_url": image_url_from_local_path(e["image_path"])
        }
        for e in items
    ]

# =====================================================
# RECENT PRODUCTS (NEW ARRIVALS)
# =====================================================

@router.get("/recent")
def recent_products(limit: int = 20):
    entities = table.query_entities(
        "PartitionKey eq 'PRODUCT'"
    )

    items = sorted(
        entities,
        key=lambda e: e["created_at"],
        reverse=True
    )[:limit]

    return [
        {
            "product_id": e["RowKey"],
            "created_at": e["created_at"],
            "image_url": image_url_from_local_path(e["image_path"])
        }
        for e in items
    ]

# =====================================================
# TRENDING PRODUCTS (VIEWS × RECENCY)
# =====================================================

def trending_score(entity):
    views = entity.get("view_count", 0)
    last_viewed = entity.get("last_viewed_at")

    if not last_viewed:
        return 0

    hours = hours_since(last_viewed)
    return views / (1 + hours)

@router.get("/trending")
def trending_products(limit: int = 20):
    entities = table.query_entities(
        "PartitionKey eq 'PRODUCT'"
    )

    items = sorted(
        entities,
        key=trending_score,
        reverse=True
    )[:limit]

    return [
        {
            "product_id": e["RowKey"],
            "view_count": e["view_count"],
            "image_url": image_url_from_local_path(e["image_path"])
        }
        for e in items
    ]
