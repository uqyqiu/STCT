import torch.nn as nn


class BasicConv1d_A(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicConv1d_A, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.conv2 = nn.Conv1d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool1d(2,2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpooling(x)
        return x

class BasicConv1d_B(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicConv1d_B, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.conv2 = nn.Conv1d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.conv3 = nn.Conv1d(out_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.bn3 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool1d(2,2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpooling(x)
        return x

class SpatialFeatureExtractor(nn.Module):
    def __init__(self,):
        super(SpatialFeatureExtractor, self).__init__()
        # Modules
        self.spatial_features = nn.Sequential(
            BasicConv1d_A(1,64),
            BasicConv1d_A(64,128),
            BasicConv1d_B(128,256),
            BasicConv1d_B(256,256),
            BasicConv1d_B(256,256)
        )
    def forward(self, input):
        return self.spatial_features(input)