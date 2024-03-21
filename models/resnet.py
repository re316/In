'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if x.dim() == out.dim() - 1:
            x = x.unsqueeze(1)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def forward_bottle_MD(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out_list.append(out)
        return out, out_list

    def intermediate_bottle_MD(self, x, layer_index, c):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        c += 1
        if layer_index == c:
            return out, c
        return out, c


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, output_dim=10, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = output_dim

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

    def mc_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = nn.functional.dropout(out, p=0.5)
        out = self.linear(out)
        return out

    def deep_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return out

    def feature_list_MD(self, x):
        out_list=[]
        out = x
        out = F.relu(self.bn1(self.conv1(out)))
        out_list.append(out)

        for m in self.layer1:
            out, layer_features = m.forward_bottle_MD(out)
            out_list = out_list + layer_features
        for m in self.layer2:
            out, layer_features = m.forward_bottle_MD(out)
            out_list = out_list + layer_features
        for m in self.layer3:
            out, layer_features = m.forward_bottle_MD(out)
            out_list = out_list + layer_features
        for m in self.layer4:
            out, layer_features = m.forward_bottle_MD(out)
            out_list = out_list + layer_features
        out = F.avg_pool2d(out, 4)
        out_list.append(out)
        out = out.view(out.size(0), -1)
        out_list.append(out)
        out = self.dropout(out)
        out = self.linear(out)
        out_list.append(out)
        return out, out_list

    def intermediate_forward_MD(self, x, layer_index):
        out = x
        c = -1
        out = F.relu(self.bn1(self.conv1(out)))
        c += 1
        if layer_index == c:
            return out
        for m in self.layer1:
            out, c = m.intermediate_bottle_MD(out, layer_index, c)
            if layer_index == c:
                return out
        for m in self.layer2:
            out, c = m.intermediate_bottle_MD(out, layer_index, c)
            if layer_index == c:
                return out
        for m in self.layer3:
            out, c = m.intermediate_bottle_MD(out, layer_index, c)
            if layer_index == c:
                return out
        for m in self.layer4:
            out, c = m.intermediate_bottle_MD(out, layer_index, c)
            if layer_index == c:
                return out
        out = F.avg_pool2d(out, 4)
        c += 1
        if layer_index == c:
            return out
        out = out.view(out.size(0), -1)
        c += 1
        if layer_index == c:
            return out
        out = self.dropout(out)
        out = self.linear(out)
        c += 1
        if layer_index == c:
            return out
        return out

    def layer_wise(self, x):
        out_list = []
        out_list.append(x) #1
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)#2
        out = self.layer1(out)
        out_list.append(out)#3
        out = self.layer2(out)
        out_list.append(out)#4
        out = self.layer3(out)
        out_list.append(out)#5
        out = self.layer4(out)
        out_list.append(out)#6
        out = F.avg_pool2d(out, 4)
        out_list.append(out)#7
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        out_list.append(out)#8
        return out_list

    def layer_wise_odds_are_odd(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out_list.append(out)
        out = self.linear(out)
        out_list.append(out)
        return out_list

    def libre_forward(self, x, return_features):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out_1 = self.layer4(out)
        if out_1.dim() == 5:
            out = F.avg_pool2d(out_1.flatten(0,1), 4).view(*out_1.shape[:3])
            out = out.view(*out.shape[:2], -1)
            out = self.dropout(out)
            out = self.linear(out.flatten(0,1)).view(*out.shape[:2], -1)
        else:
            out = F.avg_pool2d(out_1, 4)
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.linear(out)
        if return_features:
            return out_1, out
        else:
            return out

    def feature_func(self, image):
        fea_list = [image]
        out = F.relu(self.bn1(self.conv1(image)))
        fea_list.append(out)
        out = self.layer1(out)
        out = self.layer2(out)
        fea_list.append(out)
        out = self.layer3(out)
        out = self.layer4(out)
        fea_list.append(out)
        return fea_list



def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(output_dim=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], output_dim=output_dim)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())





