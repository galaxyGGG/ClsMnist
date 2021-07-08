# Author: Arashi
# Time: 2021/6/29 上午8:56
# Desc:
# from torch import nn
# from  torchvision import models
# class ResNetReDefine(nn.Module):
#     def __init__(self,outputShape = 6):
#         super(ResNetReDefine,self).__init__()
#         self.model = models.resnet50(pretrained=False)
#         self.layer = nn.Linear(in_features=self.model.fc.in_features,out_features=outputShape)
#         self.model.fc = self.layer
#     def forward(self, x):
#         return self.model(x)