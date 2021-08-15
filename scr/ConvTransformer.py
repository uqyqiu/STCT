import torch.nn as nn

class Convproj(nn.Module):
  def __init__(self,in_plane,out_plane,normalization_size):
    super(Convproj,self).__init__()
    self.conv = nn.Conv1d(in_plane,out_plane,3,1,1)
    self.bn = nn.BatchNorm1d(in_plane)
    self.relu = nn.ReLU()

  def forward(self,x):
      proj = self.relu(self.bn(self.conv(x)))
      return proj

class transformer(nn.Module):
   def __init__(self,in_plane,out_plane,multihead_size,head):
      super(transformer,self).__init__()
      self.query = Convproj(in_plane,out_plane,multihead_size)
      self.key = Convproj(in_plane,out_plane,multihead_size)
      self.value = Convproj(in_plane,out_plane,multihead_size)
      self.multihead_attn = nn.MultiheadAttention(multihead_size,head,dropout=0.1,
                                                  batch_first=True)
      self.norm1 = nn.LayerNorm(multihead_size)
      self.norm2 = nn.LayerNorm(multihead_size)

      self.share_mlp1 = nn.Conv1d(in_plane,in_plane*2,3,1,1)
      self.share_mlp2 = nn.Conv1d(in_plane*2,in_plane,3,1,1)
      self.drop = nn.Dropout(0.1)
      self.sigma = nn.Sigmoid()
      self.relu = nn.ReLU()
      self.dropout = nn.Dropout(0.1)
      self.dropout2 = nn.Dropout(0.1)
      self.bn1 = nn.BatchNorm1d(in_plane*2)
      self.bn2= nn.BatchNorm1d(in_plane)
   
   def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, _ = self.multihead_attn(q,k,v)
        # add & Norm
        out_med = self.norm1(attn_output + x)
        # MLP
        out = self.share_mlp1(out_med)
        out = self.relu(out)
        out = self.share_mlp2(out)
        out = self.relu(out)
        # add & norm
        out = self.norm2(out+out_med)
        return out
