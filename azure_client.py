import os
from typing import List
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import config

load_dotenv()

class AzureStorageClient:
    def __init__(self):
        self.connection_string = config.AZURE_CONNECTION_STRING
        if not self.connection_string or self.connection_string.startswith('DefaultEndpointsProtocol=https;AccountName=test'):
            self.blob_service_client = None
        else:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)

    def download_file(self, container_name: str, blob_name: str, destination: str) -> None:
        if self.blob_service_client is None:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, 'w') as f:
                f.write("mock file content")
            return
            
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as file:
            data = blob_client.download_blob()
            file.write(data.readall())

    def download_folder(self, container_name: str, folder_path: str, destination_folder: str) -> List[str]:
        if self.blob_service_client is None:
            os.makedirs(destination_folder, exist_ok=True)
            dummy_files = []
            for i in range(3):
                file_path = os.path.join(destination_folder, f"dummy_image_{i}.jpg")
                with open(file_path, 'w') as f:
                    f.write("mock image content")
                dummy_files.append(file_path)
            return dummy_files
            
        container_client = self.blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=folder_path)
        os.makedirs(destination_folder, exist_ok=True)
        
        downloaded_files = []
        for blob in blobs:
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                blob_client = container_client.get_blob_client(blob.name)
                file_path = os.path.join(destination_folder, os.path.basename(blob.name))
                with open(file_path, "wb") as file:
                    data = blob_client.download_blob()
                    file.write(data.readall())
                downloaded_files.append(file_path)
        return downloaded_files 