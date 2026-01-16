import os
from pathlib import Path
from datetime import datetime, timedelta
from azure.storage.blob import (
    BlobServiceClient,
    generate_container_sas,
    ContainerSasPermissions
)

# =====================================================
# CONFIG
# =====================================================

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

TMP_DIR = "/tmp"

# Images live under this virtual directory
AZURE_BLOB_PREFIX = "images"
ALLOWED_IMAGE_PREFIX = "images/img/"

# SAS validity
SAS_EXPIRY_HOURS = 12

if not CONNECTION_STRING:
    raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")

if not CONTAINER_NAME:
    raise RuntimeError("AZURE_CONTAINER_NAME not set")

# =====================================================
# SINGLETONS
# =====================================================

_blob_service_client = None
_container_client = None
_cached_sas_token = None
_cached_sas_expiry = None


def _get_container_client():
    global _blob_service_client, _container_client

    if _container_client:
        return _container_client

    _blob_service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING
    )
    _container_client = _blob_service_client.get_container_client(
        CONTAINER_NAME
    )

    return _container_client


def _get_account_name_and_key():
    """
    Extract account name & key from connection string
    """
    parts = dict(
        item.split("=", 1)
        for item in CONNECTION_STRING.split(";")
        if "=" in item
    )

    return parts["AccountName"], parts["AccountKey"]


# =====================================================
# SAS GENERATION (CACHED)
# =====================================================

def _get_container_sas() -> str:
    """
    Generate (and cache) a read-only container SAS token.
    """
    global _cached_sas_token, _cached_sas_expiry

    now = datetime.utcnow()

    # Reuse if still valid
    if _cached_sas_token and _cached_sas_expiry and now < _cached_sas_expiry:
        return _cached_sas_token

    account_name, account_key = _get_account_name_and_key()

    expiry = now + timedelta(hours=SAS_EXPIRY_HOURS)

    sas = generate_container_sas(
        account_name=account_name,
        container_name=CONTAINER_NAME,
        account_key=account_key,
        permission=ContainerSasPermissions(read=True),
        expiry=expiry,
        start=now - timedelta(minutes=5),
        protocol="https"
    )

    _cached_sas_token = f"?{sas}"
    _cached_sas_expiry = expiry

    return _cached_sas_token


# =====================================================
# SERVER-SIDE DOWNLOAD (FAISS / METADATA)
# =====================================================

def download_blob(blob_name: str) -> str:
    """
    Download a blob to /tmp only once.
    Used for FAISS index & metadata.
    """
    local_path = os.path.join(TMP_DIR, blob_name)

    if os.path.exists(local_path):
        print(f"♻️ Using cached blob: {blob_name}")
        return local_path

    print(f"⬇️ Downloading blob: {blob_name}")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    container = _get_container_client()
    blob = container.get_blob_client(blob_name)

    tmp_path = local_path + ".tmp"
    with open(tmp_path, "wb") as f:
        stream = blob.download_blob()
        stream.readinto(f)

    os.replace(tmp_path, local_path)

    return local_path


# =====================================================
# CLIENT-SIDE IMAGE URL GENERATION
# =====================================================

def image_url_from_local_path(local_path: str) -> str:
    """
    Convert DeepFashion local path to Azure Blob URL
    with dynamically generated SAS token.
    """

    p = Path(local_path)
    parts = p.parts

    if "img" not in parts:
        raise ValueError(f"Invalid DeepFashion path: {local_path}")

    img_index = parts.index("img")
    relative_path = "/".join(parts[img_index:])

    blob_path = f"{AZURE_BLOB_PREFIX}/{relative_path}"

    # 🔒 Enforce directory boundary
    if not blob_path.startswith(ALLOWED_IMAGE_PREFIX):
        raise ValueError("Blocked access outside images/img")

    sas = _get_container_sas()
    account_name, _ = _get_account_name_and_key()

    return (
        f"https://{account_name}.blob.core.windows.net/"
        f"{CONTAINER_NAME}/{blob_path}"
        f"{sas}"
    )
def upload_blob(local_path: str, blob_name: str):
    container = _get_container_client()
    blob = container.get_blob_client(blob_name)

    with open(local_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)
