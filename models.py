import torch
import torch.nn as nn
from network_architectures import initialize_backbone

class PrototypicalNetwork(nn.Module):
    def __init__(self, backbone_name: str = "googlenet", pretrained: bool = True):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = initialize_backbone(backbone_name, pretrained)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

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
            class_prototypes[class_idx] = class_features.mean(dim=0)
        
        dists = torch.cdist(query_features, class_prototypes)
        class_scores = -dists
        
        return class_scores

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images) 