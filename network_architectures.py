import torch
import torch.nn as nn

def initialize_backbone(backbone_name: str = "googlenet", pretrained: bool = True, device: torch.device = None, input_channels: int = 3) -> nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Initializing {backbone_name} backbone on device: {device} with {input_channels} input channels")
    
    if backbone_name.lower() == "googlenet":
        try:
            import torchvision.models as models
            
            if pretrained:
                model = models.googlenet(weights='DEFAULT')
            else:
                model = models.googlenet(weights=None)
                
            model.fc = nn.Identity()
            
            if input_channels != 3:
                print(f"Modifying first conv layer for {input_channels} input channels")
                original_conv1 = model.conv1.conv
                
                new_conv1 = nn.Conv2d(
                    input_channels, 
                    original_conv1.out_channels,
                    kernel_size=original_conv1.kernel_size,
                    stride=original_conv1.stride,
                    padding=original_conv1.padding,
                    bias=original_conv1.bias is not None
                )
                
                if input_channels == 1 and pretrained:
                    with torch.no_grad():
                        new_conv1.weight[:, 0, :, :] = original_conv1.weight.mean(dim=1)
                        if original_conv1.bias is not None:
                            new_conv1.bias.copy_(original_conv1.bias)
                elif input_channels > 3 and pretrained:
                    with torch.no_grad():
                        for i in range(input_channels):
                            new_conv1.weight[:, i, :, :] = original_conv1.weight[:, i % 3, :, :]
                        if original_conv1.bias is not None:
                            new_conv1.bias.copy_(original_conv1.bias)
                
                model.conv1.conv = new_conv1
            
            print(f"GoogleNet loaded successfully (pretrained={pretrained})")
            
        except ImportError as e:
            print(f"Warning: TorchVision not available ({e}), using mock model")
            model = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Flatten(), 
                nn.Linear(input_channels, 1024)
            )
        except Exception as e:
            print(f"Error loading GoogleNet: {e}, using mock model")
            model = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Flatten(), 
                nn.Linear(input_channels, 1024)
            )
    else:
        raise ValueError(f"Unsupported backbone architecture: {backbone_name}")
    
    try:
        model = model.to(device)
        model.eval()
        print(f"Backbone moved to {device} successfully")
        
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 224, 224, device=device)
            output = model(dummy_input)
            print(f"Backbone test successful. Output shape: {output.shape}")
            
    except Exception as e:
        print(f"Error setting up backbone on device {device}: {e}")
        if device.type == 'cuda':
            print("Falling back to CPU")
            device = torch.device('cpu')
            model = model.to(device)
            model.eval()
    
    return model 