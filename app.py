import os
import requests
from fastapi import FastAPI, HTTPException
import torch
import tempfile
import shutil
from dotenv import load_dotenv

from models import PrototypicalNetwork
from image_processor import ImageProcessor
from azure_client import AzureStorageClient
from schemas import TaskRequest, TaskInfo, Prediction, PredictionResponse
import config
import uvicorn

# Load environment variables
load_dotenv()

# Configuration from config file
BACKEND_API_URL = config.BACKEND_API_URL

class PredictionService:
    def __init__(self):
        self.model = None
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.image_processor = ImageProcessor()
        self.azure_client = AzureStorageClient()

    def initialize_model(self, backbone_name: str = "googlenet", pretrained: bool = True) -> None:
        """Initialize a new model with specified backbone"""
        self.model = PrototypicalNetwork(backbone_name=backbone_name, pretrained=pretrained)
        self.model.to(self.device)
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    def load_model(self, model_path: str) -> None:
        """Load a saved model from path"""
        # First initialize with default architecture
        self.initialize_model()
        
        # Load the saved state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Print model device info
        print(f"\nModel loaded and moved to {self.device}")
        if self.device.type == 'cuda':
            print(f"Model GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    @torch.no_grad()
    def evaluate_task(self, support_images_paths: list[str], query_image_path: str) -> tuple[int, float]:
        """Evaluate a single task using the prototypical network approach"""
        if self.model is None:
            raise ValueError("Model not initialized. Call load_model or initialize_model first.")
            
        # Process support images
        support_images = self.image_processor.preprocess_images(support_images_paths).to(self.device)
        support_labels = torch.arange(len(support_images_paths)).to(self.device)
        
        # Process query image
        query_tensor = self.image_processor.preprocess_image(query_image_path).to(self.device)
        
        # Get predictions
        scores = self.model(support_images, support_labels, query_tensor)
        
        # Convert scores to probabilities and get predictions
        probabilities = torch.softmax(scores, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_label].item()
        
        if self.device.type == 'cuda':
            # Clear GPU cache after prediction
            torch.cuda.empty_cache()
        
        return predicted_label, confidence

prediction_service = PredictionService()
app = FastAPI(title="WriterID Prediction API")

@app.on_event("startup")
async def startup_event():
    """Print GPU information on startup"""
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"Compute Capability: {props.major}.{props.minor}")
    else:
        print("\nNo GPU available, using CPU")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TaskRequest) -> PredictionResponse:
    try:
        # Get task information from backend
        response = requests.get(f"{BACKEND_API_URL}/task/{request.task_id}")
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to get task information")
        
        task_info = TaskInfo(**response.json())
        
        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp()
        try:
            # Download model file
            model_path = os.path.join(temp_dir, "model.pth")
            prediction_service.azure_client.download_file(
                task_info.model_container,
                task_info.model_blob_path,
                model_path
            )
            
            # Load model
            prediction_service.load_model(model_path)
            
            # Download query image
            query_image_path = os.path.join(temp_dir, "query_image.jpg")
            prediction_service.azure_client.download_file(
                task_info.query_container,
                task_info.query_image_path,
                query_image_path
            )
            
            # Process each writer's samples
            writer_samples = []
            for writer_info in task_info.writer_samples:
                # Download writer's sample images
                writer_folder = os.path.join(temp_dir, f"writer_{writer_info.writer_id}")
                sample_images = prediction_service.azure_client.download_folder(
                    task_info.samples_container,
                    writer_info.folder_path,
                    writer_folder
                )
                if sample_images:
                    writer_samples.append({
                        'writer_id': writer_info.writer_id,
                        'sample_paths': sample_images
                    })
            
            if not writer_samples:
                raise HTTPException(status_code=400, detail="No valid writer samples found")
            
            # Evaluate the task
            all_sample_paths = [sample['sample_paths'][0] for sample in writer_samples]  # Take first sample from each writer
            predicted_idx, confidence = prediction_service.evaluate_task(all_sample_paths, query_image_path)
            
            predicted_writer = writer_samples[predicted_idx]['writer_id'] if 0 <= predicted_idx < len(writer_samples) else None
            
            return PredictionResponse(
                task_id=request.task_id,
                query_image=task_info.query_image_path,
                prediction=Prediction(
                    writer_id=predicted_writer,
                    confidence=confidence
                )
            )
            
        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 