# Author: Arashi
# Time: 2021/6/29 上午8:55
# Desc:
from modelStructure.baseModel import *
from torch import nn
from torchvision import models
class ResNetReDefine(BaseModel):
    def __init__(self,outputShape = 19):
        super(ResNetReDefine,self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.layer = nn.Linear(in_features=self.model.fc.in_features,out_features=outputShape)
        self.model.fc = self.layer
    def forward(self, x):
        return self.model(x)