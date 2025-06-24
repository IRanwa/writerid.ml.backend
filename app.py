import os
import requests
from fastapi import FastAPI, HTTPException
import torch
import tempfile
import shutil
from dotenv import load_dotenv
import urllib3
from urllib.parse import urlparse

from models import PrototypicalNetwork
from image_processor import ImageProcessor
from azure_client import AzureStorageClient
from schemas import TaskRequest, TaskInfo, Prediction, PredictionResponse, TaskExecutionInfo, TaskExecutionRequest, TaskExecutionResponse
import config
import uvicorn

# Load environment variables
load_dotenv()

# Configuration from config file
BACKEND_API_URL = config.BACKEND_API_URL
EXTERNAL_API_BASE_URL = config.EXTERNAL_API_BASE_URL
EXTERNAL_API_KEY = config.EXTERNAL_API_KEY

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def is_localhost_or_dev_env(url: str) -> bool:
    """Check if URL is localhost or development environment"""
    parsed = urlparse(url)
    return parsed.hostname in ['localhost', '127.0.0.1'] or parsed.hostname.endswith('.local')

def make_request(url: str, headers: dict = None, **kwargs) -> requests.Response:
    """Make HTTP request with appropriate SSL handling and headers"""
    # For localhost/development, disable SSL verification
    if is_localhost_or_dev_env(url):
        kwargs['verify'] = False
    
    # Add headers if provided
    if headers:
        kwargs['headers'] = headers
    
    return requests.get(url, **kwargs)

def setup_device():
    """Setup and configure device for PyTorch operations"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nGPU Setup:")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Set default device and clear cache
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("\nNo GPU available, using CPU")
        return torch.device('cpu')

class PredictionService:
    def __init__(self):
        self.model = None
        self.device = setup_device()
        print(f"PredictionService initialized with device: {self.device}")
        try:
            self.image_processor = ImageProcessor(device=self.device)
            self.azure_client = AzureStorageClient()
        except Exception as e:
            print(f"Error initializing services: {e}")
            raise

    def initialize_model(self, backbone_name: str = "googlenet", pretrained: bool = True) -> None:
        try:
            print(f"Initializing model with backbone: {backbone_name} for grayscale input")
            self.model = PrototypicalNetwork(backbone_name=backbone_name, pretrained=pretrained, device=self.device)
            self.model.to(self.device)
            if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
                print("Using DataParallel for multi-GPU setup")
                self.model = torch.nn.DataParallel(self.model)
            self.model.eval()
            if self.device.type == 'cuda':
                print(f"Model moved to GPU. Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def load_model(self, model_path: str) -> None:
        try:
            print(f"Loading model from: {model_path}")
            if self.device.type == 'cuda':
                state_dict = torch.load(model_path, map_location=self.device)
            else:
                state_dict = torch.load(model_path, map_location='cpu')
            print(f"Initializing model with backbone: googlenet for grayscale input")
            self.model = PrototypicalNetwork(backbone_name="googlenet", pretrained=True, device=self.device)
            self.model.to(self.device)
            if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
                print("Using DataParallel for multi-GPU setup")
                self.model = torch.nn.DataParallel(self.model)
            self.model.eval()
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict_flexible(state_dict, strict=False)
            else:
                self.model.load_state_dict_flexible(state_dict, strict=False)
            print(f"Model loaded successfully on {self.device}")
            if self.device.type == 'cuda':
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @torch.no_grad()
    def evaluate_task(self, support_images_paths: list[str], query_image_path: str, support_labels: list[int] = None, rejection_threshold: float = 15.0) -> tuple[int, float]:
        """Evaluate a single task using the prototypical network approach"""
        if self.model is None:
            raise ValueError("Model not initialized. Call load_model or initialize_model first.")
        
        try:
            print(f"Evaluating task with {len(support_images_paths)} support images")
            print(f"Using rejection threshold: {rejection_threshold}")
            
            # Process support images
            support_images = self.image_processor.preprocess_images(support_images_paths)
            
            # Process support labels
            if support_labels is None:
                # Default: each image is its own class
                support_labels = torch.arange(len(support_images_paths), device=self.device, dtype=torch.long)
            else:
                # Convert list to tensor with long dtype
                support_labels = torch.tensor(support_labels, dtype=torch.long, device=self.device)
            
            # Process query image
            query_tensor = self.image_processor.preprocess_image(query_image_path)
            
            print(f"Support images shape: {support_images.shape}")
            print(f"Support labels: {support_labels}")
            print(f"Support labels type: {type(support_labels)}")
            print(f"Support labels values: {support_labels.tolist() if hasattr(support_labels, 'tolist') else support_labels}")
            print(f"Query image shape: {query_tensor.shape}")
            print(f"Using device: {self.device}")
            
            # Get predictions using the new forward signature with rejection threshold
            scores = self.model.forward(support_images, support_labels, query_tensor, rejection_threshold=rejection_threshold)
            print(f"Scores shape: {scores.shape}, Scores: {scores}")
            
            # Convert scores to probabilities and get predictions
            probabilities = torch.softmax(scores, dim=1)
            print(f"Probabilities shape: {probabilities.shape}, Probabilities: {probabilities}")
            
            # Check if the prediction was rejected
            max_prob = torch.max(probabilities, dim=1).values.item()
            predicted_label = torch.argmax(probabilities, dim=1).item()
            
            # If the maximum probability is very low (due to rejection), mark as rejected
            if max_prob < 0.01:  # Very low probability indicates rejection
                print(f"PREDICTION REJECTED: Query image doesn't match any known writer well")
                print(f"Maximum probability: {max_prob:.6f} (below rejection threshold)")
                predicted_label = -1  # Special value for rejected predictions
                confidence = 0.0
            else:
                print(f"Predicted label: {predicted_label}")
                # Defensive: check if predicted_label is in range
                if probabilities.shape[1] <= predicted_label:
                    print(f"Warning: predicted_label {predicted_label} is out of range for probabilities shape {probabilities.shape}")
                    confidence = float('nan')
                else:
                    confidence = probabilities[0][predicted_label].item()
                
                # Check confidence threshold (75%)
                if confidence < 0.75:
                    print(f"CONFIDENCE TOO LOW: {confidence:.4f} < 0.75 threshold")
                    print(f"Rejecting prediction due to low confidence")
                    predicted_label = -1  # Mark as unknown writer
                    confidence = 0.0
                else:
                    print(f"Prediction: label={predicted_label}, confidence={confidence:.4f}")
            
            if self.device.type == 'cuda':
                # Clear GPU cache after prediction
                torch.cuda.empty_cache()
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise

prediction_service = PredictionService()
app = FastAPI(title="WriterID Prediction API")

@app.on_event("startup")
async def startup_event():
    """Print system information on startup"""
    print(f"\n=== WriterID Prediction API Startup ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)")
    
    print(f"Backend API URL: {BACKEND_API_URL}")
    print(f"External API Base URL: {EXTERNAL_API_BASE_URL}")
    
    # Mask API key for security (show only first 8 and last 4 characters)
    if len(EXTERNAL_API_KEY) > 12:
        masked_key = f"{EXTERNAL_API_KEY[:8]}***{EXTERNAL_API_KEY[-4:]}"
    else:
        masked_key = "***"
    print(f"External API Key: {masked_key}")
    
    # Check for default model file
    default_model_path = os.path.join(os.getcwd(), "default_model.pth")
    if os.path.exists(default_model_path):
        model_size = os.path.getsize(default_model_path)
        print(f"Default Model: Found at {default_model_path} ({model_size / (1024*1024):.2f} MB)")
    else:
        print(f"Default Model: NOT FOUND at {default_model_path}")
        print("  ⚠️  Place your default_model.pth file in the root directory for default model functionality")
    
    print("=== Startup Complete ===\n")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TaskRequest) -> PredictionResponse:
    """
    Execute writer identification task by calling external API for execution info
    and then performing the identification process
    """
    try:
        # Get task execution information from external API
        print(f"Task Id: {request.task_id}")
        external_api_url = f"{EXTERNAL_API_BASE_URL}/api/external/tasks/{request.task_id}/execution-info"
        
        # Set headers for external API request
        headers = {
            'X-API-Key': EXTERNAL_API_KEY,
            'Content-Type': 'application/json'
        }
        
        print(f"Calling external API: {external_api_url}")
        response = make_request(external_api_url, headers=headers)
        print(f"Response: {response.status_code}")
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to get task execution info: {response.text}")
        
        execution_info = TaskExecutionInfo(**response.json())
        print(f"Execution Info: {execution_info}")
        # Create task-specific working directory
        task_work_dir = os.path.join(tempfile.gettempdir(), f"task_{execution_info.taskId}")
        os.makedirs(task_work_dir, exist_ok=True)
        print(f"Task Work Dir: {task_work_dir}")
        try:
            # 1. Load model - check if default model should be used
            if execution_info.useDefaultModel:
                # Use local default model from root directory
                print("Using local default model from root directory")
                local_default_model = os.path.join(os.getcwd(), "default_model.pth")
                
                if not os.path.exists(local_default_model):
                    raise HTTPException(status_code=500, detail=f"Default model file not found at {local_default_model}")
                
                # Copy to task directory
                model_path = os.path.join(task_work_dir, "model.pth")
                try:
                    import shutil
                    shutil.copy2(local_default_model, model_path)
                    print(f"Default model copied to: {model_path}")
                except Exception as e:
                    print(f"Error copying default model: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to copy default model: {str(e)}")
            else:
                # Use specific model from modelContainerName
                if not execution_info.modelContainerName:
                    raise HTTPException(status_code=400, detail="Model container name is required when not using default model")
                
                print(f"Using specific model from container: {execution_info.modelContainerName}")
                model_path = os.path.join(task_work_dir, "model.pth")
                try:
                    # Assuming model file is named model.pth in the container
                    prediction_service.azure_client.download_file(
                        execution_info.modelContainerName,
                        "model.pth",
                        model_path
                    )
                    print(f"Specific model downloaded successfully to: {model_path}")
                except Exception as e:
                    print(f"Error downloading specific model: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to download model from {execution_info.modelContainerName}: {str(e)}")
            
            # Verify model file exists and has reasonable size
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Model file was not created at {model_path}")
            
            model_size = os.path.getsize(model_path)
            print(f"Model file size: {model_size / (1024*1024):.2f} MB")
            
            if model_size < 1024:  # Less than 1KB seems too small for a model
                print(f"Warning: Model file seems very small ({model_size} bytes)")
            
            # Load the model
            print("Loading model into PredictionService...")
            try:
                prediction_service.load_model(model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
            
            # 2. Download query image
            query_image_path = os.path.join(task_work_dir, execution_info.queryImageFileName)
            query_blob_path = f"{execution_info.queryImageFileName}"
            prediction_service.azure_client.download_file(
                execution_info.taskContainerName,
                query_blob_path,
                query_image_path
            )
            
            # 3. Download only selected writers from dataset container
            writer_samples = []
            writers_dir = os.path.join(task_work_dir, "writers")
            os.makedirs(writers_dir, exist_ok=True)
            
            for writer_id in execution_info.selectedWriters:
                writer_folder = os.path.join(writers_dir, f"writer_{writer_id}")
                writer_blob_path = f"{writer_id}"
                
                # Download writer's sample images
                sample_images = prediction_service.azure_client.download_folder(
                    execution_info.datasetContainerName,
                    writer_blob_path,
                    writer_folder
                )
                
                if sample_images:
                    writer_samples.append({
                        'writer_id': writer_id,
                        'sample_paths': sample_images
                    })
                    print(f"Downloaded {len(sample_images)} samples for writer {writer_id}")
            
            if not writer_samples:
                raise HTTPException(status_code=400, detail="No valid writer samples found for selected writers")
            
            # 4. Perform prediction using prototypical network
            # Use up to 5 samples from each writer for support set
            support_image_paths = []
            support_labels = []
            writer_ids_order = []
            
            for writer_idx, sample in enumerate(writer_samples):
                writer_id = sample['writer_id']
                sample_paths = sample['sample_paths']
                
                # Use up to 5 samples from each writer
                num_samples = min(5, len(sample_paths))
                selected_samples = sample_paths[:num_samples]
                
                for sample_path in selected_samples:
                    support_image_paths.append(sample_path)
                    support_labels.append(writer_idx)  # Use writer_idx as class label
                
                writer_ids_order.append(writer_id)
                print(f"Added {num_samples} support images for writer {writer_id} (class {writer_idx})")
            
            if not support_image_paths:
                raise HTTPException(status_code=400, detail="No support images available")
            
            print(f"Total support images: {len(support_image_paths)} from {len(writer_ids_order)} writers")
            print(f"Support labels: {support_labels}")
            
            # Evaluate the task
            predicted_idx, confidence = prediction_service.evaluate_task(support_image_paths, query_image_path, support_labels)
            
            # Handle rejected predictions
            if predicted_idx == -1:
                print(f"Prediction rejected - query image doesn't match any known writer")
                return PredictionResponse(
                    task_id=execution_info.taskId,
                    query_image=execution_info.queryImageFileName,
                    prediction=Prediction(
                        writer_id="unknown",  # Return "unknown" as writer ID
                        confidence=0.0  # With 0.0 confidence to indicate rejection
                    )
                )
            
            # Map prediction index back to writer ID
            predicted_writer = writer_ids_order[predicted_idx] if 0 <= predicted_idx < len(writer_ids_order) else None
            
            print(f"Prediction completed: Writer {predicted_writer} with confidence {confidence:.4f}")
            
            return PredictionResponse(
                task_id=execution_info.taskId,
                query_image=execution_info.queryImageFileName,
                prediction=Prediction(
                    writer_id=predicted_writer,
                    confidence=confidence
                )
            )
            
        finally:
            # 5. Cleanup - remove task working directory
            if os.path.exists(task_work_dir):
                shutil.rmtree(task_work_dir)
                print(f"Cleaned up task directory: {task_work_dir}")
            
    except Exception as e:
        print(f"Error in task execution: {str(e)}")
        # Try to clean up on error as well
        task_work_dir = os.path.join(tempfile.gettempdir(), f"task_{request.task_id}")
        if os.path.exists(task_work_dir):
            shutil.rmtree(task_work_dir, ignore_errors=True)
        
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 