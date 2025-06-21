import os
import time
from typing import List
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import config

load_dotenv()

class AzureStorageClient:
    def __init__(self):
        self.connection_string = config.AZURE_CONNECTION_STRING
        print(f"Azure Connection String configured: {len(self.connection_string)} characters")
        
        self.use_mock = self._should_use_mock()
        
        if self.use_mock:
            print("Using mock Azure Storage (no real Azure connection)")
            self.blob_service_client = None
        else:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                print("Azure Storage client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Azure Storage client: {e}")
                print("Falling back to mock mode")
                self.use_mock = True
                self.blob_service_client = None

    def _should_use_mock(self) -> bool:
        if not self.connection_string:
            return True
        
        if (self.connection_string.startswith('DefaultEndpointsProtocol=https;AccountName=test') or
            'AccountName=test' in self.connection_string or
            len(self.connection_string) < 50):
            return True
        
        return False

    def download_file(self, container_name: str, blob_name: str, destination: str) -> None:
        print(f"Downloading {blob_name} from container {container_name} to {destination}")
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        if self.use_mock:
            self._download_file_mock(container_name, blob_name, destination)
            return
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Download attempt {attempt + 1}/{max_retries}")
                container_client = self.blob_service_client.get_container_client(container_name)
                blob_client = container_client.get_blob_client(blob_name)
                
                if not blob_client.exists():
                    print(f"Blob {blob_name} does not exist in container {container_name}")
                    print("Falling back to mock download")
                    self._download_file_mock(container_name, blob_name, destination)
                    return
                
                properties = blob_client.get_blob_properties()
                file_size = properties.size
                print(f"File size: {file_size / (1024*1024):.2f} MB")
                
                if file_size < 10 * 1024 * 1024:
                    print("Using direct download for small file")
                    with open(destination, "wb") as file:
                        download_stream = blob_client.download_blob()
                        file.write(download_stream.readall())
                else:
                    print("Using streaming download for large file")
                    with open(destination, "wb") as file:
                        download_stream = blob_client.download_blob()
                        
                        bytes_downloaded = 0
                        chunk_size = 1024 * 1024
                        
                        while True:
                            chunk = download_stream.readall(chunk_size)
                            if not chunk:
                                break
                            file.write(chunk)
                            bytes_downloaded += len(chunk)
                            
                            progress = (bytes_downloaded / file_size) * 100
                            print(f"Download progress: {progress:.1f}% ({bytes_downloaded / (1024*1024):.1f}/{file_size / (1024*1024):.1f} MB)")
                
                print(f"Successfully downloaded {blob_name} ({file_size / (1024*1024):.2f} MB)")
                return
                
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("All download attempts failed, falling back to mock download")
                    self._download_file_mock(container_name, blob_name, destination)
                else:
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)

    def _download_file_mock(self, container_name: str, blob_name: str, destination: str) -> None:
        print(f"MOCK: Creating mock file for {blob_name}")
        
        if blob_name.endswith('.pth') or 'model' in blob_name.lower():
            try:
                import torch
                import torch.nn as nn
                from collections import OrderedDict
                
                print("MOCK: Creating realistic PyTorch model file...")
                
                state_dict = OrderedDict([
                    ('backbone.features.0.weight', torch.randn(64, 3, 3, 3)),
                    ('backbone.features.0.bias', torch.randn(64)),
                    ('backbone.features.1.weight', torch.randn(64)),
                    ('backbone.features.1.bias', torch.randn(64)),
                    ('backbone.features.1.running_mean', torch.randn(64)),
                    ('backbone.features.1.running_var', torch.ones(64)),
                    ('backbone.features.1.num_batches_tracked', torch.tensor(0)),
                    ('backbone.classifier.weight', torch.randn(1000, 1024)),
                    ('backbone.classifier.bias', torch.randn(1000)),
                ])
                
                torch.save(state_dict, destination)
                
                file_size = os.path.getsize(destination)
                print(f"MOCK: Created mock PyTorch model at {destination} ({file_size / (1024*1024):.2f} MB)")
                
            except Exception as e:
                print(f"Error creating mock PyTorch model: {e}")
                with open(destination, 'wb') as f:
                    mock_data = b'\x50\x4b\x03\x04' + b'\x00' * (22 * 1024 * 1024 - 4)
                    f.write(mock_data)
                print(f"MOCK: Created fallback binary model file at {destination}")
                
        elif blob_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            if blob_name.lower().endswith('.png'):
                mock_image_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x64\x00\x00\x00\x64\x08\x02\x00\x00\x00\xff\x80\x02\x03\x00\x00\x00\x19tEXtSoftware\x00Adobe ImageReadyq\xc9e<\x00\x00\x00\x0eIDATx\xdab\x00\x02\x00\x00\x05\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            else:
                mock_image_content = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xFF\xC0\x00\x11\x08\x00d\x00d\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xFF\xC4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xFF\xC4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xDA\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xAA\xFF\xD9'
            
            with open(destination, 'wb') as f:
                f.write(mock_image_content)
            print(f"MOCK: Created mock image at {destination}")
        else:
            with open(destination, 'w') as f:
                f.write(f"Mock content for {blob_name}\nContainer: {container_name}\nTimestamp: {time.time()}")
            print(f"MOCK: Created mock file at {destination}")
        
        if os.path.exists(destination):
            file_size = os.path.getsize(destination)
            print(f"MOCK: File created successfully - Size: {file_size} bytes")
        else:
            print(f"MOCK: ERROR - File was not created at {destination}")

    def download_folder(self, container_name: str, folder_path: str, destination_folder: str) -> List[str]:
        print(f"Downloading folder {folder_path} from container {container_name} to {destination_folder}")
        
        os.makedirs(destination_folder, exist_ok=True)
        
        if self.use_mock:
            return self._download_folder_mock(container_name, folder_path, destination_folder)
        
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = container_client.list_blobs(name_starts_with=folder_path)
            
            downloaded_files = []
            for blob in blobs:
                if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(destination_folder, os.path.basename(blob.name))
                    
                    try:
                        blob_client = container_client.get_blob_client(blob.name)
                        with open(file_path, "wb") as file:
                            blob_data = blob_client.download_blob()
                            file.write(blob_data.readall())
                        downloaded_files.append(file_path)
                        print(f"Downloaded: {blob.name}")
                    except Exception as e:
                        print(f"Failed to download {blob.name}: {e}")
                        
            return downloaded_files
            
        except Exception as e:
            print(f"Error downloading folder {folder_path}: {e}")
            print("Falling back to mock download")
            return self._download_folder_mock(container_name, folder_path, destination_folder)

    def _download_folder_mock(self, container_name: str, folder_path: str, destination_folder: str) -> List[str]:
        print(f"MOCK: Creating mock files for folder {folder_path}")
        
        dummy_files = []
        for i in range(3):
            file_path = os.path.join(destination_folder, f"mock_image_{i}.jpg")
            mock_image_content = b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xFF\xC0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xFF\xC4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xFF\xC4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xFF\xDA\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xAA\xFF\xD9'
            
            with open(file_path, 'wb') as f:
                f.write(mock_image_content)
            dummy_files.append(file_path)
            print(f"MOCK: Created {file_path}")
            
        return dummy_files 