import torch
from torch import nn
from torch import Tensor
from network_architectures import BackboneNetworkHandler
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone_name: str = "googlenet", pretrained: bool = True, device: torch.device = None, input_channels: int = 1):
        super(PrototypicalNetwork, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        backbone_handler = BackboneNetworkHandler(name=backbone_name, pretrained=pretrained)
        self.backbone = backbone_handler.get_model()
        

        
        self.to(self.device)

    def load_state_dict_flexible(self, state_dict, strict=False):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if "size mismatch" in str(e) and "conv" in str(e):
                fixed_state_dict = {}
                
                for key, value in state_dict.items():
                    if 'conv1' in key and 'weight' in key and value.dim() == 4:
                        if value.shape[1] == 1 and self.backbone.conv1.conv.in_channels == 3:
                            fixed_value = value.repeat(1, 3, 1, 1)
                            fixed_state_dict[key] = fixed_value
                        elif value.shape[1] == 3 and self.backbone.conv1.conv.in_channels == 1:
                            fixed_value = value.mean(dim=1, keepdim=True)
                            fixed_state_dict[key] = fixed_value
                        else:
                            fixed_state_dict[key] = value
                    else:
                        fixed_state_dict[key] = value
                
                return super().load_state_dict(fixed_state_dict, strict=strict)
            else:
                raise e

    def forward(self, support_images: Tensor, support_labels: Tensor, query_images: Tensor) -> Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)
        
        support_labels = support_labels.to(z_support.device)
        unique_labels = torch.unique(support_labels)
        n_way = len(unique_labels)
        
        all_prototypes = []
        for label_idx in range(n_way):
            current_label_val = unique_labels[label_idx]
            label_mask = torch.nonzero(support_labels == current_label_val, as_tuple=False)
            if label_mask.numel() > 0:
                label_mask = label_mask.squeeze(-1)
                proto = z_support[label_mask].mean(0)
                all_prototypes.append(proto)
        
        if not all_prototypes:
            raise ValueError("No prototypes computed - all classes have empty support sets")
        
        z_proto = torch.stack(all_prototypes)
        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        
        return scores

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        return self.backbone(images)
    
 