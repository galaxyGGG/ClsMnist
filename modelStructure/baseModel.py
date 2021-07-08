# Author: Arashi
# Time: 2021/6/29 上午9:04
# Desc:
import torch.nn as nn
class BaseModel(nn.Module):
    def __init__(self,outputShape=10):
        super(BaseModel,self).__init__()
