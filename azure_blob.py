import os
from azure.storage.blob import BlobServiceClient
from io import BytesIO

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

blob_service = BlobServiceClient.from_connection_string(
    CONNECTION_STRING
)
container = blob_service.get_container_client(CONTAINER_NAME)


def download_blob(path: str) -> BytesIO:
    blob = container.get_blob_client(path)
    return BytesIO(blob.download_blob().readall())


def upload_blob(path: str, data: bytes):
    blob = container.get_blob_client(path)
    blob.upload_blob(data, overwrite=True)
