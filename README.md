# WriterID ML Backend

A FastAPI-based machine learning service for writer identification using prototypical networks. This service provides REST API endpoints for identifying handwriting samples by comparing them against known writer samples.

## Features

- **Writer Identification**: Uses prototypical networks to identify handwriting samples
- **Azure Integration**: Downloads models and data from Azure Blob Storage
- **GPU Support**: Optimized for CUDA-enabled GPUs with automatic fallback to CPU
- **REST API**: FastAPI-based endpoints for easy integration
- **Grayscale Image Processing**: Specialized for handwriting analysis
- **Multi-Model Support**: Supports various backbone architectures (GoogLeNet, ResNet, etc.)

## Architecture

The system uses a prototypical network approach for few-shot learning:

1. **Support Set**: Multiple handwriting samples from known writers
2. **Query Image**: Unknown handwriting sample to identify
3. **Prototype Computation**: Creates prototypes for each writer class
4. **Distance Calculation**: Computes distances between query and prototypes
5. **Classification**: Predicts the most likely writer based on similarity

## API Endpoints

### POST /predict
Identifies the writer of a query handwriting sample.

**Request:**
```json
{
  "task_id": "string"
}
```

**Response:**
```json
{
  "task_id": "string",
  "query_image": "string",
  "prediction": {
    "writer_id": "string",
    "confidence": 0.95
  }
}
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Azure Storage Account (for model and data storage)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd writerid-ml-backend
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration:**
Create a `.env` file with the following variables:
```env
EXTERNAL_API_BASE_URL=https://your-api-server.com
EXTERNAL_API_KEY=your-api-key
AZURE_CONNECTION_STRING=your-azure-connection-string
```

4. **Model Setup:**
Place your trained model file as `default_model.pth` in the root directory, or configure Azure storage for model retrieval.

## Usage

### Running the Service

```bash
python app.py
```

The service will start on `http://localhost:5000` with the following features:
- Automatic GPU detection and setup
- Model loading and validation
- API endpoint availability

### API Documentation

Once running, visit `http://localhost:5000/docs` for interactive API documentation.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXTERNAL_API_BASE_URL` | External API server URL | `https://localhost:44302` |
| `EXTERNAL_API_KEY` | API authentication key | `WID-API-2024-SecureKey-XYZ789` |
| `AZURE_CONNECTION_STRING` | Azure Storage connection string | Test connection string |
| `DEBUG` | Enable debug mode | `True` |
| `LOG_LEVEL` | Logging level | `DEBUG` |

### Model Configuration

The system supports multiple backbone architectures:
- **GoogLeNet** (default): Best performance for handwriting analysis
- **ResNet18**: Good balance of speed and accuracy
- **SqueezeNet**: Lightweight model for resource-constrained environments
- **MobileNet V3**: Mobile-optimized architecture
- **EfficientNet**: High accuracy with efficient computation

## File Structure

```
writerid-ml-backend/
├── app.py                 # Main FastAPI application
├── models.py              # Prototypical network implementation
├── image_processor.py     # Image preprocessing utilities
├── azure_client.py        # Azure Blob Storage integration
├── network_architectures.py # Backbone network definitions
├── schemas.py             # Pydantic data models
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── default_model.pth     # Default trained model (not included)
```

## Development

### Adding New Backbone Networks

1. Import the model in `network_architectures.py`
2. Add to the `weights_map` and `model_fn_map` dictionaries
3. Implement the adaptation logic in `_adapt_to_single_channel`
4. Add the model configuration in `_create_backbone`

### Customizing Image Processing

Modify `image_processor.py` to adjust:
- Image size (default: 224x224)
- Normalization parameters
- Data augmentation transforms

## Performance

### GPU Optimization
- Automatic CUDA detection and setup
- Multi-GPU support with DataParallel
- Memory management with cache clearing
- Optimized tensor operations

### Azure Integration
- Chunked downloads for large files
- Resume capability for interrupted downloads
- Mock mode for development without Azure
- Progress tracking for long operations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or image resolution
   - Use CPU mode for testing
   - Clear GPU cache between predictions

2. **Model Loading Errors**
   - Verify model file exists and is not corrupted
   - Check model architecture compatibility
   - Ensure proper PyTorch version

3. **Azure Connection Issues**
   - Verify connection string format
   - Check network connectivity
   - Use mock mode for development

### Logging

The service provides detailed logging for:
- GPU setup and memory allocation
- Model loading and validation
- API request processing
- Error handling and debugging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/docs` 