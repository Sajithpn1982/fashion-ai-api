from fastapi import APIRouter, HTTPException
from azure.data.tables import TableServiceClient
from datetime import datetime, timezone
import uuid
import os

from azure_blob import image_url_from_local_path
from azure_blob import image_url_from_local_path
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# =====================================================
# CONFIG
# =====================================================

ORDERS_TABLE = "Orders"
PRODUCT_SALES_TABLE = "ProductSales"

router = APIRouter(
    prefix="/purchase",
    tags=["Fashion Purchase"]
)

# =====================================================
# AZURE TABLE STORAGE
# =====================================================

service = TableServiceClient.from_connection_string(
    CONNECTION_STRING
)

service.create_table_if_not_exists(ORDERS_TABLE)
service.create_table_if_not_exists(PRODUCT_SALES_TABLE)

orders_table = service.get_table_client(ORDERS_TABLE)
product_sales_table = service.get_table_client(PRODUCT_SALES_TABLE)

# =====================================================
# UTILITIES
# =====================================================

def iso_now():
    return datetime.now(timezone.utc).isoformat()

# =====================================================
# CREATE PURCHASE (ORDER)
# =====================================================

@router.post("/{user_id}")
def create_purchase(
    user_id: str,
    product_id: str,
    image_path: str,
    quantity: int,
    price: float
):
    """
    Create a purchase order (payment can be added later)
    """

    if quantity <= 0 or price <= 0:
        raise HTTPException(status_code=400, detail="Invalid quantity or price")

    order_id = str(uuid.uuid4())
    total_amount = quantity * price

    # 1️⃣ Create order
    orders_table.create_entity({
        "PartitionKey": user_id,
        "RowKey": order_id,
        "product_id": product_id,
        "image_path": image_path,
        "quantity": quantity,
        "price": price,
        "total_amount": total_amount,
        "status": "PAID",  # simplify; integrate payment later
        "created_at": iso_now()
    })

    # 2️⃣ Update product sales
    try:
        p = product_sales_table.get_entity("PRODUCT", product_id)
        p["total_sales"] += quantity
        p["total_revenue"] += total_amount
        p["image_path"] = image_path
        product_sales_table.update_entity(p, mode="replace")
    except:
        product_sales_table.create_entity({
            "PartitionKey": "PRODUCT",
            "RowKey": product_id,
            "image_path": image_path,
            "total_sales": quantity,
            "total_revenue": total_amount
        })

    return {
        "order_id": order_id,
        "product_id": product_id,
        "image_path": image_path,
        "quantity": quantity,
        "total_amount": total_amount,
        "status": "PAID"
    }

# =====================================================
# GET MY ORDERS
# =====================================================

@router.get("/orders/{user_id}")
def get_my_orders(user_id: str, limit: int = 50):
    """
    Get all orders for a user
    """

    entities = orders_table.query_entities(
        f"PartitionKey eq '{user_id}'"
    )

    items = sorted(
        entities,
        key=lambda e: e["created_at"],
        reverse=True
    )[:limit]

    return [
        {
            "order_id": e["RowKey"],
            "product_id": e["product_id"],
            "image_path": e["image_path"],
            "image_url": image_url_from_local_path(e["image_path"]),
            "quantity": e["quantity"],
            "total_amount": e["total_amount"],
            "status": e["status"],
            "created_at": e["created_at"]
        }
        for e in items
    ]

# =====================================================
# GET ORDER DETAILS
# =====================================================

@router.get("/order/{user_id}/{order_id}")
def get_order(user_id: str, order_id: str):
    """
    Get single order details
    """
    try:
        e = orders_table.get_entity(user_id, order_id)
        return {
            "order_id": order_id,
            "product_id": e["product_id"],
            "image_path": e["image_path"],
            "image_url": image_url_from_local_path(e["image_path"]),
            "quantity": e["quantity"],
            "price": e["price"],
            "total_amount": e["total_amount"],
            "status": e["status"],
            "created_at": e["created_at"]
        }
    except:
        raise HTTPException(status_code=404, detail="Order not found")

# =====================================================
# TOP SELLING PRODUCTS
# =====================================================

@router.get("/top-selling")
def top_selling(limit: int = 20):
    """
    Get top selling products
    """
    entities = product_sales_table.query_entities(
        "PartitionKey eq 'PRODUCT'"
    )

    items = sorted(
        entities,
        key=lambda e: e.get("total_sales", 0),
        reverse=True
    )[:limit]

    return [
        {
            "product_id": e["RowKey"],
            "image_path": e["image_path"],
            "image_url": image_url_from_local_path(e["image_path"]),
            "total_sales": e["total_sales"]
        }
        for e in items
    ]
