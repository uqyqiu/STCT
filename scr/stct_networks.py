import torch.nn as nn
import torch
import torch.nn.functional as F
from scr.CNN13 import SpatialFeatureExtractor
from scr.ConvTransformer import transformer
from scr.util import PositionalEncoding


class STCT(nn.Module):
  def __init__(self,):
    super(STCT,self).__init__()
    self.cnn = SpatialFeatureExtractor()
    self.transform = transformer(256,256,39,13)
    self.pos = PositionalEncoding(39)
    self.linear = nn.Linear(256,5)
    self.bn = nn.BatchNorm1d(256)
    self.relu = nn.ReLU()

  def forward(self,x):
    out = self.cnn(x)
    out += self.pos(out)
    out = self.transform(out)
    out = self.bn(out)
    out=self.relu(out)
    out,_ = torch.max(out,dim=2)
    out = self.linear(out)
    out = F.log_softmax(out,dim=1)
    return out