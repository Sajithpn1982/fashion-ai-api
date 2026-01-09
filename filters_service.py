import json
from datetime import datetime
from table_client import filters_table

PK = "FILTER"

def save_filters(email: str, filters: dict):
    filters_table.upsert_entity({
        "PartitionKey": PK,
        "RowKey": email,
        "filters": json.dumps(filters),
        "updatedAt": datetime.utcnow().isoformat()
    })

def get_filters(email: str):
    try:
        entity = filters_table.get_entity(PK, email)
        return json.loads(entity["filters"])
    except Exception:
        return {}
