import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import List, Union
import numpy as np
import io

class ImageProcessor:
    def __init__(self, device: torch.device = None, input_channels: int = 1):
        """
        Initialize the ImageProcessor with grayscale preprocessing transforms.
        
        Args:
            device: PyTorch device to use for tensor operations
            input_channels: Number of input channels expected by the model (1 for grayscale)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.input_channels = input_channels
        self.image_size = 224
        
        # Grayscale normalization values
        self.grayscale_mean = [0.5]
        self.grayscale_std = [0.5]
        
        # Transform for grayscale images (matching reference implementation)
        self.transform = transforms.Compose([
            transforms.Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)]),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.grayscale_mean, std=self.grayscale_std)
        ])
        
        print(f"ImageProcessor initialized on device: {self.device} with {input_channels} input channels (grayscale)")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image for model inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor of shape (1, 1, 224, 224)
        """
        try:
            # Load image and convert to grayscale
            image = Image.open(image_path).convert('L')
            
            # Apply transforms
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise

    def preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Preprocess multiple images for model inference.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Preprocessed image tensor of shape (N, 1, 224, 224) where N is number of images
        """
        try:
            tensors = []
            
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Load image and convert to grayscale
                image = Image.open(image_path).convert('L')
                
                # Apply transforms
                tensor = self.transform(image)
                tensors.append(tensor)
            
            # Stack all tensors
            batch_tensor = torch.stack(tensors)
            
            # Move to device
            batch_tensor = batch_tensor.to(self.device)
            
            return batch_tensor
            
        except Exception as e:
            print(f"Error preprocessing images: {e}")
            raise

    def preprocess_image_from_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess an image from bytes data.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Preprocessed image tensor of shape (1, 1, 224, 224)
        """
        try:
            # Load from bytes and convert to grayscale
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            
            # Apply transforms
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
        """
        Preprocess multiple images from bytes data.
        
        Args:
            image_bytes_list: List of image data as bytes
            
        Returns:
            Preprocessed image tensor of shape (N, 1, 224, 224) where N is number of images
        """
        try:
            tensors = []
            
            for image_bytes in image_bytes_list:
                # Load from bytes and convert to grayscale
                image = Image.open(io.BytesIO(image_bytes)).convert('L')
                
                # Apply transforms
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