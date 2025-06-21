import torch
import torch.nn as nn
from network_architectures import initialize_backbone

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone_name: str = "googlenet", pretrained: bool = True, device: torch.device = None, input_channels: int = 3):
        super(PrototypicalNetwork, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Initializing PrototypicalNetwork on device: {self.device}")
        
        self.input_channels = input_channels
        
        self.backbone = initialize_backbone(backbone_name, pretrained, self.device, input_channels)
        self.to(self.device)

    def load_state_dict_flexible(self, state_dict, strict=False):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if "size mismatch" in str(e) and "conv" in str(e):
                print(f"Shape mismatch detected, attempting to fix: {e}")
                
                fixed_state_dict = {}
                
                for key, value in state_dict.items():
                    if 'conv1' in key and 'weight' in key and value.dim() == 4:
                        if value.shape[1] == 1 and self.input_channels == 3:
                            print(f"Expanding {key} from {value.shape} to match 3-channel input")
                            fixed_value = value.repeat(1, 3, 1, 1)
                            fixed_state_dict[key] = fixed_value
                        elif value.shape[1] == 3 and self.input_channels == 1:
                            print(f"Converting {key} from {value.shape} to match 1-channel input")
                            fixed_value = value.mean(dim=1, keepdim=True)
                            fixed_state_dict[key] = fixed_value
                        else:
                            fixed_state_dict[key] = value
                    else:
                        fixed_state_dict[key] = value
                
                return super().load_state_dict(fixed_state_dict, strict=strict)
            else:
                raise e

    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_image: torch.Tensor) -> torch.Tensor:
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_image = query_image.to(self.device)

        support_features = self.backbone(support_images)
        query_features = self.backbone(query_image)
        
        n_classes = len(torch.unique(support_labels))
        class_prototypes = torch.zeros(n_classes, support_features.shape[1], device=self.device)
        
        for class_idx in range(n_classes):
            class_mask = (support_labels == class_idx)
            class_features = support_features[class_mask]
            if len(class_features) > 0:
                class_prototypes[class_idx] = class_features.mean(dim=0)
        
        dists = torch.cdist(query_features, class_prototypes)
        class_scores = -dists
        
        return class_scores

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        return self.backbone(images) 