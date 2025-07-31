import os
import requests
from fastapi import FastAPI, HTTPException
import torch
import tempfile
from dotenv import load_dotenv
import urllib3
from urllib.parse import urlparse

from models import PrototypicalNetwork
from image_processor import ImageProcessor
from azure_client import AzureStorageClient
from schemas import TaskRequest, Prediction, PredictionResponse, TaskExecutionInfo
import config
import uvicorn

load_dotenv()

EXTERNAL_API_BASE_URL = config.EXTERNAL_API_BASE_URL
EXTERNAL_API_KEY = config.EXTERNAL_API_KEY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def is_localhost_or_dev_env(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in ['localhost', '127.0.0.1'] or parsed.hostname.endswith('.local')

def make_request(url: str, headers: dict = None, **kwargs) -> requests.Response:
    if is_localhost_or_dev_env(url):
        kwargs['verify'] = False
    
    if headers:
        kwargs['headers'] = headers
    
    return requests.get(url, **kwargs)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        return device
    else:
        return torch.device('cpu')

class PredictionService:
    def __init__(self):
        self.model = None
        self.device = setup_device()
        try:
            self.image_processor = ImageProcessor(device=self.device)
            self.azure_client = AzureStorageClient()
        except Exception as e:
            print(f"Error initializing services: {e}")
            raise



    def load_model(self, model_path: str) -> None:
        try:
            if self.device.type == 'cuda':
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            else:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model = PrototypicalNetwork(backbone_name="googlenet", pretrained=True, device=self.device)
            self.model.to(self.device)
            if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.eval()
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict_flexible(state_dict, strict=False)
            else:
                self.model.load_state_dict_flexible(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @torch.no_grad()
    def evaluate_task(self, support_images_paths: list[str], query_image_path: str, support_labels: list[int] = None) -> tuple[int, float]:
        if self.model is None:
            raise ValueError("Model not initialized. Call load_model first.")
        
        try:
            # Process support images with consistent transforms
            support_images = self.image_processor.preprocess_images(support_images_paths)
            
            if support_labels is None:
                support_labels = torch.arange(len(support_images_paths), device=self.device, dtype=torch.long)
            else:
                support_labels = torch.tensor(support_labels, dtype=torch.long, device=self.device)
            
            # Process query image with consistent transforms
            query_tensor = self.image_processor.preprocess_image(query_image_path)
            
            scores = self.model.forward(support_images, support_labels, query_tensor)
            probabilities = torch.softmax(scores, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            
            if probabilities.shape[1] <= predicted_label:
                confidence = float('nan')
            else:
                confidence = probabilities[0][predicted_label].item()
            
            if self.device.type == 'cuda':
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
    print(f"WriterID Prediction API started")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    default_model_path = os.path.join(os.getcwd(), "default_model.pth")
    if os.path.exists(default_model_path):
        model_size = os.path.getsize(default_model_path)
        print(f"Default Model: Found ({model_size / (1024*1024):.2f} MB)")
    else:
        print(f"Default Model: NOT FOUND - Place default_model.pth in root directory")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TaskRequest) -> PredictionResponse:
    import shutil
    
    try:
        # Get task execution information from external API
        external_api_url = f"{EXTERNAL_API_BASE_URL}/api/external/tasks/{request.task_id}/execution-info"
        
        # Set headers for external API request
        headers = {
            'X-API-Key': EXTERNAL_API_KEY,
            'Content-Type': 'application/json'
        }
        
        response = make_request(external_api_url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to get task execution info: {response.text}")
        
        execution_info = TaskExecutionInfo(**response.json())
        # Create task-specific working directory
        task_work_dir = os.path.join(tempfile.gettempdir(), f"task_{execution_info.taskId}")
        os.makedirs(task_work_dir, exist_ok=True)
        try:
            if execution_info.useDefaultModel:
                local_default_model = os.path.join(os.getcwd(), "default_model.pth")
                
                if not os.path.exists(local_default_model):
                    raise HTTPException(status_code=500, detail=f"Default model file not found at {local_default_model}")
                
                model_path = os.path.join(task_work_dir, "model.pth")
                try:
                    shutil.copy2(local_default_model, model_path)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to copy default model: {str(e)}")
            else:
                if not execution_info.modelContainerName:
                    raise HTTPException(status_code=400, detail="Model container name is required when not using default model")
                
                model_path = os.path.join(task_work_dir, "model.pth")
                try:
                    prediction_service.azure_client.download_file(
                        execution_info.modelContainerName,
                        "best_model.pth",
                        model_path
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to download model from {execution_info.modelContainerName}: {str(e)}")
            
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Model file was not created at {model_path}")
            
            try:
                prediction_service.load_model(model_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
            
            query_image_path = os.path.join(task_work_dir, execution_info.queryImageFileName)
            query_blob_path = f"{execution_info.queryImageFileName}"
            prediction_service.azure_client.download_file(
                execution_info.taskContainerName,
                query_blob_path,
                query_image_path
            )
            
            writer_samples = []
            writers_dir = os.path.join(task_work_dir, "writers")
            os.makedirs(writers_dir, exist_ok=True)
            
            for writer_id in execution_info.selectedWriters:
                writer_folder = os.path.join(writers_dir, f"writer_{writer_id}")
                writer_blob_path = f"{writer_id}"
                
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
            
            if not writer_samples:
                raise HTTPException(status_code=400, detail="No valid writer samples found for selected writers")
            
            support_image_paths = []
            support_labels = []
            writer_ids_order = []
            
            for writer_idx, sample in enumerate(writer_samples):
                writer_id = sample['writer_id']
                sample_paths = sample['sample_paths']
                
                num_samples = min(5, len(sample_paths))
                selected_samples = sample_paths[:num_samples]
                
                for sample_path in selected_samples:
                    support_image_paths.append(sample_path)
                    support_labels.append(writer_idx)
                
                writer_ids_order.append(writer_id)
            
            if not support_image_paths:
                raise HTTPException(status_code=400, detail="No support images available")
            
            predicted_idx, confidence = prediction_service.evaluate_task(support_image_paths, query_image_path, support_labels)
            
            predicted_writer = writer_ids_order[predicted_idx] if 0 <= predicted_idx < len(writer_ids_order) else None
            
            return PredictionResponse(
                task_id=execution_info.taskId,
                query_image=execution_info.queryImageFileName,
                prediction=Prediction(
                    writer_id=predicted_writer,
                    confidence=confidence
                )
            )
            
        finally:
            if os.path.exists(task_work_dir):
                shutil.rmtree(task_work_dir)
            
    except Exception as e:
        task_work_dir = os.path.join(tempfile.gettempdir(), f"task_{request.task_id}")
        if os.path.exists(task_work_dir):
            shutil.rmtree(task_work_dir, ignore_errors=True)
        
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 