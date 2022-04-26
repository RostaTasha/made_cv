import torch.nn as nn
import torchvision.models as models
from utils import NUM_PTS



class MyModel(nn.Module):
    def __init__(self, freeze):
        super(MyModel, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=True)
        self.backbone.requires_grad_(not freeze)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2 * NUM_PTS, bias=True)
        self.backbone.fc.requires_grad_(True)
    
    def forward(self, d):
        return self.backbone.forward(d)