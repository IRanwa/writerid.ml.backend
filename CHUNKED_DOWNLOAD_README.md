# Enhanced Chunked Download Functionality

This document describes the enhanced chunked download functionality implemented in the Azure Storage client for the WriterID ML Backend.

## Features

### 🚀 **Chunked Downloads**
- Downloads large files in configurable chunks (default: 2MB)
- Automatic chunk size optimization based on file size
- Prevents memory overflow for large model files

### 🔄 **Resume Capability**
- Automatically resumes interrupted downloads
- Uses `.partial` files to track download progress
- Saves bandwidth and time for large files

### 📊 **Progress Tracking**
- Real-time progress reporting with percentage and speed
- Custom progress callbacks for integration with UI
- Detailed logging for debugging

### ✅ **Download Verification**
- File size verification after download
- Basic integrity checks for PyTorch model files
- Automatic retry on verification failure

### 🛡️ **Error Handling**
- Exponential backoff retry strategy
- Graceful fallback to mock downloads in development
- Comprehensive error logging

## Usage Examples

### Basic Model Download

```python
from azure_client import AzureStorageClient

client = AzureStorageClient()

# Download a model with default settings
success = client.download_model(
    model_path="models/my_model.pth",  # container/blob_name format
    destination="./local_model.pth"
)

if success:
    print("Model downloaded successfully!")
```

### Custom Chunk Size and Progress Tracking

```python
def my_progress_callback(downloaded: int, total: int):
    progress = (downloaded / total) * 100
    print(f"Progress: {progress:.1f}%")

# Download with custom settings
client.download_file(
    container_name="data",
    blob_name="large_dataset.zip",
    destination="./dataset.zip",
    chunk_size=8 * 1024 * 1024,  # 8MB chunks
    resume=True,
    delete_existing=True,  # Delete existing file before download
    progress_callback=my_progress_callback
)
```

### Resume Interrupted Downloads

```python
# This will automatically resume if a partial download exists
client.download_file(
    container_name="models",
    blob_name="large_model.pth",
    destination="./model.pth",
    resume=True  # This is the default
)
```

### Delete Existing Files Before Download

```python
# Always delete existing files before downloading (default behavior)
client.download_file(
    container_name="models",
    blob_name="updated_model.pth",
    destination="./model.pth",
    delete_existing=True  # This is the default
)

# Keep existing files if they exist (useful for caching)
client.download_file(
    container_name="data",
    blob_name="dataset.zip",
    destination="./dataset.zip",
    delete_existing=False  # Don't delete existing files
)

# For models, you can also control deletion
success = client.download_model(
    model_path="models/latest_model.pth",
    destination="./model.pth",
    delete_existing=True  # Ensure fresh download
)
```

## Configuration Options

### Chunk Sizes

The system automatically optimizes chunk sizes based on file size:

| File Size | Default Chunk Size | Optimization |
|-----------|-------------------|--------------|
| < 10MB    | 1MB max          | Smaller chunks for efficiency |
| 10-100MB  | 2MB (default)    | Standard chunk size |
| > 100MB   | 4MB min          | Larger chunks for performance |

### Custom Chunk Sizes

You can override the default chunk size for specific use cases:

```python
# For images (small files)
client.download_file(..., chunk_size=512 * 1024)  # 512KB

# For documents (medium files)
client.download_file(..., chunk_size=2 * 1024 * 1024)  # 2MB

# For videos/large models (large files)
client.download_file(..., chunk_size=16 * 1024 * 1024)  # 16MB
```

## API Reference

### `download_file()`

```python
def download_file(
    self, 
    container_name: str, 
    blob_name: str, 
    destination: str,
    chunk_size: int = 2 * 1024 * 1024,
    resume: bool = True,
    delete_existing: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> None
```

**Parameters:**
- `container_name`: Azure storage container name
- `blob_name`: Name of the blob to download
- `destination`: Local file path to save the downloaded file
- `chunk_size`: Size of chunks in bytes (default: 2MB)
- `resume`: Whether to resume partial downloads (default: True)
- `delete_existing`: Whether to delete existing file before download (default: True)
- `progress_callback`: Optional callback function for progress updates

### `download_model()`

```python
def download_model(
    self, 
    model_path: str, 
    destination: str,
    chunk_size: int = 4 * 1024 * 1024,
    delete_existing: bool = True
) -> bool
```

**Parameters:**
- `model_path`: Path to the model in Azure storage (format: container/blob_name)
- `destination`: Local file path to save the model
- `chunk_size`: Size of chunks in bytes (default: 4MB for models)
- `delete_existing`: Whether to delete existing model file before download (default: True)

**Returns:**
- `bool`: True if download successful, False otherwise

## Performance Considerations

### Optimal Chunk Sizes

- **Small files (< 10MB)**: Use 512KB - 1MB chunks
- **Medium files (10-100MB)**: Use 2-4MB chunks
- **Large files (> 100MB)**: Use 4-16MB chunks

### Network Considerations

- Larger chunks = fewer network requests but more memory usage
- Smaller chunks = more network requests but less memory usage
- The system includes a small delay (0.01s) between chunks to prevent overwhelming the connection

### Resume Functionality

- Partial downloads are saved as `.partial` files
- Resume checks file size to determine where to continue
- Automatically cleans up partial files on successful completion

### File Deletion Strategy

- **Clean Downloads**: By default, existing files are deleted before downloading to ensure clean, uncorrupted downloads
- **Partial File Handling**: When `delete_existing=True`, both the target file and any `.partial` files are removed
- **Resume vs. Delete**: If you want to resume downloads, set `delete_existing=False` and `resume=True`
- **Model Updates**: For ML models, deleting existing files ensures you get the latest version without conflicts

## Error Handling

The system includes comprehensive error handling:

1. **Network Errors**: Automatic retry with exponential backoff
2. **File Errors**: Verification and cleanup of corrupted downloads
3. **Azure Errors**: Graceful fallback to mock downloads in development
4. **Partial Downloads**: Automatic cleanup and restart on failure

## Development Mode

In development mode (when using mock Azure credentials), the system:
- Creates realistic mock files for testing
- Generates appropriate file sizes for different file types
- Maintains the same API interface for seamless testing

## Running the Demo

Use the provided example script to test the functionality:

```bash
python download_example.py
```

This will demonstrate:
- Model downloads with default settings
- Custom chunk sizes and progress callbacks
- Resume functionality
- Different configurations for different file types

## Integration with Existing Code

The enhanced functionality is backward compatible. Existing code will continue to work with the new chunked download features automatically enabled.

For new integrations, consider using the `download_model()` method for model files as it includes additional verification and optimization specifically for machine learning models. 