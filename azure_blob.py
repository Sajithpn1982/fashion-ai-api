import os
from azure.storage.blob import BlobServiceClient
#from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
TMP_DIR = "/tmp"

def download_blob(blob_name: str) -> str:
    """
    Downloads a blob to /tmp only once per container.
    Returns the local file path.
    """
    local_path = os.path.join(TMP_DIR, blob_name)

    # Reuse if already downloaded
    if os.path.exists(local_path):
        print(f"♻️ Using cached blob: {blob_name}")
        return local_path

    print(f"⬇️ Downloading blob: {blob_name}")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    blob_service = BlobServiceClient.from_connection_string(
        CONNECTION_STRING
    )
    container = blob_service.get_container_client(CONTAINER_NAME)
    blob = container.get_blob_client(blob_name)

    # Atomic write (safe)
    tmp_path = local_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(blob.download_blob().readall())

    os.replace(tmp_path, local_path)
    return local_path

