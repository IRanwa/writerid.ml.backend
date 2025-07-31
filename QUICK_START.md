# Quick Start Guide

Get the WriterID ML Backend running in minutes!

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- CUDA-compatible GPU (optional, but recommended for performance)

## Installation

### Option 1: Automated Installation (Recommended)

**Linux/macOS:**
```bash
git clone <repository-url>
cd writerid-ml-backend
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
git clone <repository-url>
cd writerid-ml-backend
install.bat
```

### Option 2: Manual Installation

1. **Clone and navigate:**
```bash
git clone <repository-url>
cd writerid-ml-backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows
```

3. **Install PyTorch:**
```bash
# With CUDA (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Create environment file:**
```bash
cp .env.example .env  # if available
# or create .env manually with your configuration
```

## Running the Service

1. **Activate virtual environment:**
```bash
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows
```

2. **Start the service:**
```bash
python app.py
```

3. **Verify it's running:**
- Open http://localhost:5000/docs in your browser
- You should see the FastAPI interactive documentation

## Testing the API

### Using curl:
```bash
curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"task_id": "test-task-123"}'
```

### Using the web interface:
1. Go to http://localhost:5000/docs
2. Click on the `/predict` endpoint
3. Click "Try it out"
4. Enter your task_id
5. Click "Execute"

## Configuration

Edit the `.env` file to configure:

```env
# External API settings
EXTERNAL_API_BASE_URL=https://your-api-server.com
EXTERNAL_API_KEY=your-api-key

# Azure Storage (optional)
AZURE_CONNECTION_STRING=your-azure-connection-string

# Debug settings
DEBUG=True
LOG_LEVEL=DEBUG
```

## Model Setup

1. **Place your model file:**
   - Copy your trained model to `default_model.pth` in the root directory
   - Or configure Azure storage to download models automatically

2. **Model requirements:**
   - PyTorch format (.pth)
   - Compatible with GoogLeNet backbone
   - Trained for grayscale handwriting images

## Troubleshooting

### Common Issues:

1. **"Module not found" errors:**
   - Make sure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **CUDA errors:**
   - Install CPU version: `pip install torch torchvision`
   - Or install correct CUDA version for your GPU

3. **Port already in use:**
   - Change port in `app.py` line: `uvicorn.run(app, host="0.0.0.0", port=5001)`

4. **Model loading errors:**
   - Check if `default_model.pth` exists
   - Verify model architecture compatibility

### Getting Help:

- Check the full [README.md](README.md) for detailed documentation
- Review logs in the terminal for error messages
- Visit http://localhost:5000/docs for API documentation

## Next Steps

- [Read the full documentation](README.md)
- [Configure Azure integration](README.md#azure-integration)
- [Customize the model architecture](README.md#development)
- [Add your own handwriting samples](README.md#usage) 