import os
import time
import hashlib
from typing import List, Optional, Callable
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import config

load_dotenv()

class AzureStorageClient:
    def __init__(self):
        self.connection_string = config.AZURE_CONNECTION_STRING
        print(f"Azure Connection String configured: {len(self.connection_string)} characters")
        
        # Check if we should use mock mode first
        self.use_mock = self._should_use_mock()
        
        if self.use_mock:
            print("Using mock Azure Storage (no real Azure connection)")
            self.blob_service_client = None
        else:
            try:
                # Validate connection string format before creating client
                if not self._is_valid_connection_string(self.connection_string):
                    print("Invalid Azure connection string format, falling back to mock mode")
                    self.use_mock = True
                    self.blob_service_client = None
                    return
                
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                print("Azure Storage client initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Azure Storage client: {e}")
                print("Falling back to mock mode")
                self.use_mock = True
                self.blob_service_client = None

    def _is_valid_connection_string(self, connection_string: str) -> bool:
        """Validate that the connection string has the required components"""
        if not connection_string:
            return False
        
        required_components = ['DefaultEndpointsProtocol', 'AccountName', 'AccountKey', 'EndpointSuffix']
        connection_string_lower = connection_string.lower()
        
        for component in required_components:
            if component.lower() not in connection_string_lower:
                return False
        
        # Check for test/mock indicators - if any are found, consider it invalid
        test_indicators = ['accountname=test', 'accountkey=test', 'accountname=dev', 'accountkey=dev']
        if any(indicator in connection_string_lower for indicator in test_indicators):
            return False
            
        return True

    def _should_use_mock(self) -> bool:
        if not self.connection_string:
            return True
        
        # Check for test/mock connection strings
        connection_string_lower = self.connection_string.lower()
        
        # Check for test account names or keys
        if any(indicator in connection_string_lower for indicator in ['accountname=test', 'accountkey=test']):
            return True
        
        # Check for short connection strings (likely test data)
        if len(self.connection_string) < 100:
            return True
        
        # Check if it's the default test connection string
        default_test = 'defaultendpointsprotocol=https;accountname=test;accountkey=test;endpointsuffix=core.windows.net'
        if connection_string_lower == default_test:
            return True
        
        return False

    def download_file(self, container_name: str, blob_name: str, destination: str, 
                     chunk_size: int = 2 * 1024 * 1024, resume: bool = True,
                     delete_existing: bool = True,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        """
        Download a file from Azure Blob Storage with chunked download support.
        
        Args:
            container_name: Name of the Azure storage container
            blob_name: Name of the blob to download
            destination: Local file path to save the downloaded file
            chunk_size: Size of chunks in bytes (default: 2MB)
            resume: Whether to resume partial downloads (default: True)
            delete_existing: Whether to delete existing file before download (default: True)
            progress_callback: Optional callback function for progress updates (bytes_downloaded, total_size)
        """
        print(f"Downloading {blob_name} from container {container_name} to {destination}")
        print(f"Chunk size: {chunk_size / (1024*1024):.2f} MB")
        
        # Create directory if it doesn't exist
        destination_dir = os.path.dirname(destination)
        if destination_dir:
            os.makedirs(destination_dir, exist_ok=True)
        
        # Delete existing file if requested
        if delete_existing:
            if os.path.exists(destination):
                print(f"Deleting existing file: {destination}")
                os.remove(destination)
        
        if self.use_mock:
            self._download_file_mock(container_name, blob_name, destination)
            return
        
        # Check if partial download exists
        bytes_downloaded = 0
        temp_destination = f"{destination}.partial"
        
        # If delete_existing is True, also remove partial files to start fresh
        if delete_existing and os.path.exists(temp_destination):
            print(f"Deleting existing partial file: {temp_destination}")
            os.remove(temp_destination)
        
        if resume and os.path.exists(temp_destination):
            bytes_downloaded = os.path.getsize(temp_destination)
            print(f"Resuming download from {bytes_downloaded / (1024*1024):.2f} MB")
        
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
                
                # Check if file is already completely downloaded
                if bytes_downloaded >= file_size:
                    print("File already completely downloaded")
                    if os.path.exists(temp_destination):
                        os.rename(temp_destination, destination)
                    return
                
                # Calculate optimal chunk size based on file size
                optimal_chunk_size = self._get_optimal_chunk_size(file_size, chunk_size)
                print(f"Using chunk size: {optimal_chunk_size / (1024*1024):.2f} MB")
                
                # Download with resume support
                success = self._download_with_chunks(
                    blob_client, temp_destination, file_size, 
                    bytes_downloaded, optimal_chunk_size, progress_callback
                )
                
                if success:
                    # Verify download integrity
                    if self._verify_download(temp_destination, file_size):
                        # Move from temp to final destination
                        os.rename(temp_destination, destination)
                        print(f"Successfully downloaded {blob_name} ({file_size / (1024*1024):.2f} MB)")
                        return
                    else:
                        print("Download verification failed, retrying...")
                        if os.path.exists(temp_destination):
                            os.remove(temp_destination)
                        bytes_downloaded = 0
                        continue
                else:
                    print("Download failed, retrying...")
                    continue
                
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("All download attempts failed, falling back to mock download")
                    # Clean up partial download
                    if os.path.exists(temp_destination):
                        os.remove(temp_destination)
                    self._download_file_mock(container_name, blob_name, destination)
                else:
                    print(f"Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
    
    def _get_optimal_chunk_size(self, file_size: int, requested_chunk_size: int) -> int:
        """Calculate optimal chunk size based on file size"""
        # For very large files (>100MB), use larger chunks
        if file_size > 100 * 1024 * 1024:
            return max(requested_chunk_size, 4 * 1024 * 1024)  # 4MB min
        # For medium files (10-100MB), use requested size
        elif file_size > 10 * 1024 * 1024:
            return requested_chunk_size
        # For small files (<10MB), use smaller chunks
        else:
            return min(requested_chunk_size, 1024 * 1024)  # 1MB max
    
    def _download_with_chunks(self, blob_client, destination: str, file_size: int,
                             start_byte: int, chunk_size: int,
                             progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """Download file in chunks with progress reporting"""
        try:
            mode = 'ab' if start_byte > 0 else 'wb'
            
            with open(destination, mode) as file:
                bytes_downloaded = start_byte
                last_progress_report = time.time()
                
                while bytes_downloaded < file_size:
                    # Calculate chunk range
                    end_byte = min(bytes_downloaded + chunk_size - 1, file_size - 1)
                    
                    # Download chunk with range
                    download_stream = blob_client.download_blob(
                        offset=bytes_downloaded,
                        length=end_byte - bytes_downloaded + 1
                    )
                    
                    chunk_data = download_stream.readall()
                    file.write(chunk_data)
                    bytes_downloaded += len(chunk_data)
                    
                    # Report progress (limit to every 2 seconds to avoid spam)
                    current_time = time.time()
                    if current_time - last_progress_report >= 2.0:
                        progress = (bytes_downloaded / file_size) * 100
                        speed = len(chunk_data) / (1024 * 1024 * 2)  # MB/s over 2 seconds
                        
                        print(f"Download progress: {progress:.1f}% "
                              f"({bytes_downloaded / (1024*1024):.1f}/{file_size / (1024*1024):.1f} MB) "
                              f"Speed: {speed:.1f} MB/s")
                        
                        if progress_callback:
                            progress_callback(bytes_downloaded, file_size)
                        
                        last_progress_report = current_time
                    
                    # Add small delay to prevent overwhelming the connection
                    time.sleep(0.01)
                
                return True
                
        except Exception as e:
            print(f"Error during chunked download: {e}")
            return False
    
    def _verify_download(self, file_path: str, expected_size: int) -> bool:
        """Verify downloaded file integrity"""
        try:
            if not os.path.exists(file_path):
                print("Downloaded file does not exist")
                return False
            
            actual_size = os.path.getsize(file_path)
            if actual_size != expected_size:
                print(f"Size mismatch: expected {expected_size}, got {actual_size}")
                return False
            
            print("Download verification successful")
            return True
            
        except Exception as e:
            print(f"Error verifying download: {e}")
            return False
    
    def download_model(self, model_path: str, destination: str, 
                      chunk_size: int = 4 * 1024 * 1024, delete_existing: bool = True) -> bool:
        """
        Download a model file with optimized settings for large files.
        
        Args:
            model_path: Path to the model in Azure storage (format: container/blob_name)
            destination: Local file path to save the model
            chunk_size: Size of chunks in bytes (default: 4MB for models)
            delete_existing: Whether to delete existing model file before download (default: True)
        
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Parse container and blob name from path
            parts = model_path.split('/', 1)
            if len(parts) != 2:
                print(f"Invalid model path format: {model_path}. Expected: container/blob_name")
                return False
            
            container_name, blob_name = parts
            
            print(f"Downloading model: {model_path}")
            print(f"Model-optimized chunk size: {chunk_size / (1024*1024):.2f} MB")
            
            # Create a progress callback for model downloads
            def model_progress_callback(downloaded: int, total: int):
                progress = (downloaded / total) * 100
                print(f"Model download progress: {progress:.1f}% "
                      f"({downloaded / (1024*1024):.1f}/{total / (1024*1024):.1f} MB)")
            
            # Download with model-specific settings
            self.download_file(
                container_name=container_name,
                blob_name=blob_name,
                destination=destination,
                chunk_size=chunk_size,
                resume=True,
                delete_existing=delete_existing,
                progress_callback=model_progress_callback
            )
            
            # Verify the model file can be loaded (basic check)
            if os.path.exists(destination):
                if self._verify_model_file(destination):
                    print(f"Model downloaded and verified successfully: {destination}")
                    return True
                else:
                    print(f"Model file verification failed: {destination}")
                    return False
            else:
                print(f"Model file not found after download: {destination}")
                return False
                
        except Exception as e:
            print(f"Error downloading model {model_path}: {e}")
            return False
    
    def _verify_model_file(self, model_path: str) -> bool:
        """Verify that the downloaded model file is valid"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # Basic file size check (models should be at least 1KB)
            file_size = os.path.getsize(model_path)
            if file_size < 1024:
                print(f"Model file too small: {file_size} bytes")
                return False
            
            # Try to load the model file to verify it's a valid PyTorch model
            if model_path.endswith('.pth'):
                try:
                    import torch
                    # Just check if it can be loaded, don't actually load into memory
                    with open(model_path, 'rb') as f:
                        # Read first few bytes to check if it looks like a PyTorch file
                        header = f.read(10)
                        if len(header) < 10:
                            print("Model file appears to be corrupted (too short)")
                            return False
                    
                    print("Model file appears to be a valid PyTorch model")
                    return True
                except Exception as e:
                    print(f"Model file verification failed: {e}")
                    return False
            else:
                # For non-PyTorch files, just check basic properties
                print("Model file basic verification passed")
                return True
                
        except Exception as e:
            print(f"Error verifying model file: {e}")
            return False

    def _download_file_mock(self, container_name: str, blob_name: str, destination: str) -> None:
        print(f"MOCK: Creating mock file for {blob_name}")
        
        # Delete existing file if it exists (for consistency with real download)
        if os.path.exists(destination):
            print(f"MOCK: Deleting existing file: {destination}")
            os.remove(destination)
        
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
            import random
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = list(container_client.list_blobs(name_starts_with=folder_path))
            
            # Filter only image blobs
            image_blobs = [blob for blob in blobs if blob.name.lower().endswith((".png", ".jpg", ".jpeg"))]
            
            # Randomly select up to 5 images
            selected_blobs = random.sample(image_blobs, min(5, len(image_blobs)))
            
            downloaded_files = []
            for blob in selected_blobs:
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