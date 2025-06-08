import torch
import torch.nn as nn

def initialize_backbone(backbone_name: str = "googlenet", pretrained: bool = True) -> nn.Module:
    if backbone_name.lower() == "googlenet":
        try:
            import torchvision.models as models
            model = models.googlenet(pretrained=pretrained)
            model.fc = nn.Identity()
        except ImportError:
            print("Warning: TorchVision not available, using mock model")
            model = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 1024))
    else:
        raise ValueError(f"Unsupported backbone architecture: {backbone_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model 