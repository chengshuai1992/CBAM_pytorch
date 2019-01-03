import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BottlenNeck(nn.Module):
    def __init__(self,in_,planes,stride=1, downsample=None):
        super(BottlenNeck,self).__init__()
        self.layer =nn.Sequential(
            nn.Conv2d(in_,planes,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes,planes*4,kernel_size=1,bias=False),
            nn.BatchNorm2d(planes*4),
            nn.ReLU(inplace=True),
            
        )
        self.ca = ChannelAttention(planes*4) #ChannelAttention
        self.sa = SpatialAttention()  #SpatialAttention
        self.relu =nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self,x):
        residual =x
        out =self.layer(x)
        out =self.ca(out)*out
        out =self.sa(out)*out
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CBAM(nn.Module):
    def __init__(self,block,num_classes=1000):
        super(CBAM,self).__init__()
        self.inplanes = 64
        self.layer_0 =nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        def m_layer(block,planes,count,stride):
            downsample = None
            if stride !=1 or planes*4!=self.inplanes:
                downsample =nn.Sequential(
                    nn.Conv2d(self.inplanes,planes*4,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(planes*4)
                )
            layers =[]
            layers.append(block(self.inplanes,planes,stride,downsample))
            self.inplanes =planes*4
            for i in range(1,count):
                layers.append(block(self.inplanes,planes))
            return layers
        self.layer_1 =nn.Sequential(
            *m_layer(block,64,3,1),
            *m_layer(block,128,4,2),
            *m_layer(block,256,6,2),
            *m_layer(block,512,3,2),
        )
        self.avgpool =nn.AvgPool2d(7,stride=1)
        self.fc =nn.Linear(512*4,num_classes)
    def forward(self,x):
        x =self.layer_0(x)
        x =self.layer_1(x)
        x =self.avgpool(x)
        x =x.view(x.size(0),-1)
        x =self.fc(x)
        return x
def resnet50_cbam(**kwargs):
    model = CBAM(BottlenNeck, **kwargs)
    return model