import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import List
import io

class ImageProcessor:
    def __init__(self, device: torch.device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.image_size = 224
        self.grayscale_mean = [0.5]
        self.grayscale_std = [0.5]
        
        self.transform = transforms.Compose([
            transforms.Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)]),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.grayscale_mean, std=self.grayscale_std)
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('L')
            
            tensor = self.transform(image)
            
            tensor = tensor.unsqueeze(0)
            
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise

    def preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        try:
            tensors = []
            
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                image = Image.open(image_path).convert('L')
                
                tensor = self.transform(image)
                
                tensors.append(tensor)
            
            batch_tensor = torch.stack(tensors)
            
            batch_tensor = batch_tensor.to(self.device)
            
            return batch_tensor
            
        except Exception as e:
            print(f"Error preprocessing images: {e}")
            raise

    def preprocess_image_from_bytes(self, image_bytes: bytes) -> torch.Tensor:
        try:
            # Load from bytes and convert to grayscale
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            
            # Apply consistent transform
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            print(f"Error preprocessing image from bytes: {e}")
            raise

    def preprocess_images_from_bytes(self, image_bytes_list: List[bytes]) -> torch.Tensor:
        try:
            tensors = []
            
            for image_bytes in image_bytes_list:
                # Load from bytes and convert to grayscale
                image = Image.open(io.BytesIO(image_bytes)).convert('L')
                
                # Apply consistent transform
                tensor = self.transform(image)
                
                tensors.append(tensor)
            
            # Stack all tensors
            batch_tensor = torch.stack(tensors)
            
            # Move to device
            batch_tensor = batch_tensor.to(self.device)
            
            return batch_tensor
            
        except Exception as e:
            print(f"Error preprocessing images from bytes: {e}")
            raise 