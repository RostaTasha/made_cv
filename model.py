import torch.nn as nn
import torchvision.models as models
from utils import NUM_PTS



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 2 * NUM_PTS, bias=True)
        self.model.fc.requires_grad_(True)
    
    def forward(self, d):
        return self.model.forward(d)