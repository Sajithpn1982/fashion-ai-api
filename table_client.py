import os
from azure.data.tables import TableServiceClient

conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not conn:
    raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")

service = TableServiceClient.from_connection_string(conn)

users_table = service.get_table_client("users")
filters_table = service.get_table_client("userfilters")

for t in (users_table, filters_table):
    try:
        t.create_table()
    except Exception:
        pass
