import torch
from torch import nn
from torch import Tensor
from network_architectures import BackboneNetworkHandler
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone_name: str = "googlenet", pretrained: bool = True, device: torch.device = None, input_channels: int = 3):
        super(PrototypicalNetwork, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Initializing PrototypicalNetwork on device: {self.device}")
        
        # Use the reference BackboneNetworkHandler
        backbone_handler = BackboneNetworkHandler(name=backbone_name, pretrained=pretrained)
        self.backbone = backbone_handler.get_model()
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
                        if value.shape[1] == 1 and self.backbone.conv1.conv.in_channels == 3:
                            print(f"Expanding {key} from {value.shape} to match 3-channel input")
                            fixed_value = value.repeat(1, 3, 1, 1)
                            fixed_state_dict[key] = fixed_value
                        elif value.shape[1] == 3 and self.backbone.conv1.conv.in_channels == 1:
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

    def forward(self, support_images: Tensor, support_labels: Tensor, query_images: Tensor, rejection_threshold: float = 75.0) -> Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        
        print(f"Forward pass - support_images: {support_images.shape}, support_labels: {support_labels}, query_images: {query_images.shape}")
        
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)
        
        print(f"Features - z_support: {z_support.shape}, z_query: {z_query.shape}")
        
        support_labels = support_labels.to(z_support.device)
        unique_labels = torch.unique(support_labels)
        n_way = len(unique_labels)
        
        print(f"unique_labels: {unique_labels}, dtype: {unique_labels.dtype}, Number of unique classes: {n_way}")
        
        all_prototypes = []
        for label_idx in range(n_way):
            current_label_val = unique_labels[label_idx]
            label_mask = torch.nonzero(support_labels == current_label_val, as_tuple=False)
            print(f"Class {current_label_val.item()}: label_mask shape={label_mask.shape}")
            if label_mask.numel() > 0:
                label_mask = label_mask.squeeze(-1)
                proto = z_support[label_mask].mean(0)
                all_prototypes.append(proto)
                print(f"Class {current_label_val.item()}: prototype shape={proto.shape}")
            else:
                print(f"Warning: No support images found for class {current_label_val.item()}")
        
        if not all_prototypes:
            raise ValueError("No prototypes computed - all classes have empty support sets")
        
        z_proto = torch.stack(all_prototypes)
        print(f"Prototypes shape: {z_proto.shape}")
        
        dists = torch.cdist(z_query, z_proto)
        print(f"Distances shape: {dists.shape}, distances: {dists}")
        
        # Add rejection logic based on minimum distance
        min_distances = torch.min(dists, dim=1, keepdim=True)[0]
        print(f"Minimum distances: {min_distances}")
        
        # Create rejection mask - if minimum distance is above threshold, reject
        rejection_mask = min_distances > rejection_threshold
        print(f"Rejection threshold: {rejection_threshold}")
        print(f"Rejection mask: {rejection_mask}")
        
        # Apply rejection by setting scores to very low values for rejected queries
        scores = -dists
        # Expand rejection mask to match scores shape for broadcasting
        rejection_mask_expanded = rejection_mask.expand_as(scores)
        scores[rejection_mask_expanded] = -1000.0  # Very low score for rejected queries
        
        print(f"Scores shape: {scores.shape}, scores: {scores}")
        
        return scores

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        return self.backbone(images) 