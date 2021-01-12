import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelMask(nn.Module):
    def __init__(self, input_channels, ratio=8):
        super(ChannelMask, self).__init__()
        squeeze_size = int(input_channels / ratio)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(input_channels, squeeze_size, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(squeeze_size, input_channels, 1)
        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2)
    def forward(self, x):
        x1 = self.avg_pooling(x)
        x1 = self.mlp(x1)
        x2 = self.max_pooling(x)
        x2 = self.mlp(x2)
        return torch.sigmoid(x1 + x2)

class TemporalMask(nn.Module):
    def __init__(self, kernel_size = 7):
        super(TemporalMask, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size = kernel_size, padding = padding)
    
    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # out = torch.cat([avg_out, max_out], dim=1)
        # out = self.conv1(out)
        out = torch.sigmoid(max_out)
        return out

class BlockV2(nn.Module):
    expansion = 1
    def __init__(self, input_channels, output_channels, stride = 1, downsample = None):
        super(BlockV2, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, 7, stride, padding=3, bias = False)
        self.bn1 = nn.BatchNorm1d(output_channels)

        self.conv2 = nn.Conv1d(output_channels, output_channels, 7, stride = 1, padding=3, bias = False)
        self.bn2 = nn.BatchNorm1d(output_channels)

        self.chn = ChannelMask(output_channels)
        self.tmp = TemporalMask()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        cm = self.chn(out)
        out = cm * out
        sm = self.tmp(out)
        out = sm * out
        
        out += residual 

        return out

class Backbone(nn.Module):
    def __init__(self, input_channels, layers=[1, 1, 1, 1], num_classes=6):
        self.inplanes = 64
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BlockV2, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(BlockV2, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BlockV2, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BlockV2, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm1d(256)
        self.avgpool = nn.AvgPool1d(6, stride=1, padding=0)

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 113, 30)
        self.fc2 = nn.Linear(30, num_classes)

    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x


class Net_Seq_E2E(nn.Module):
    def __init__(self, input_channels, seq_len = 5, frame_len = 3750):
        super(Net_Seq_E2E, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.frame_len = frame_len

        self.net = Backbone(input_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 6)
        self.fc2_1 = nn.Conv1d(256, 256, kernel_size = 5)
        self.fc2_2 = nn.Linear(256, 6)
    

    def forward(self, x, target):
        bs, seq_len = x.shape[0], x.shape[1]
        mid_mask = [i for i in range(bs * seq_len) if i % seq_len == int(seq_len/2)]
        x_seq = x.view([-1, self.frame_len, self.input_channels]) # [bs * 5, 3750, 5]

        # CNN
        pred_seq = self.net(x_seq)              # [bs * 5, 256, 113]
        pred_seq = self.bn1(pred_seq)
        pred_seq = torch.relu(pred_seq)         # [bs, 512]
        pred_seq = self.gap(pred_seq)

        x1 = pred_seq[mid_mask]                     # [bs, 512, 1]
        x2 = F.relu(pred_seq.detach())              # [bs*5, 512, 1]#######
        x2 = x2.view([bs, seq_len, 256, -1])        # [bs, 5, 512, 1]
        x2 = x2.permute([0,2,1,3]).contiguous().view([bs, 256, -1]) # [bs, 512, 5]

        # Loss 1
        x1 = x1.view(bs, -1)
        x1 = self.drop1(x1)
        x1 = self.fc1(x1)
        loss1 = self.criterion1(x1, target)

        # Loss 2
        x2 = self.fc2_1(x2)
        x2 = x2.view(bs, -1)
        x2 = self.bn2(x2)
        x2 = torch.relu(x2)
        x2 = self.drop2(x2)
        x2 = self.fc2_2(x2)
        loss2 = self.criterion2(x2, target)
        # loss = alpha * loss1 + (1 - alpha) * loss2
        # x = alpha * x1 + (1 - alpha) * x2
        return x1, x2, loss1, loss2

         