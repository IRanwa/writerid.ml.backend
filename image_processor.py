import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import List
import os

class ImageProcessor:
    def __init__(self, device: torch.device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"ImageProcessor initialized with device: {self.device}")
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        try:
            with Image.open(image_path).convert('RGB') as img:
                img_tensor = self.transform(img)
            
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            return img_tensor
            
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")

    def preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        if not image_paths:
            raise ValueError("No image paths provided")
            
        img_tensors = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found at {path}")
            
            try:
                with Image.open(path).convert('RGB') as img:
                    img_tensor = self.transform(img)
                    img_tensors.append(img_tensor)
            except Exception as e:
                raise ValueError(f"Error processing image {path}: {e}")
        
        batch_tensor = torch.stack(img_tensors).to(self.device)
        return batch_tensor 