'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, output_dim=10, dropout_rate=0.5):
        super(VGG, self).__init__()
        self.output_dim = output_dim
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(512, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    # 保留dropout 输出最后一层
    def mc_features(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = nn.functional.dropout(out, p=0.5)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out
    
    # 输出倒数第二层
    def deep_features(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        return out
    
    def feature_list_MD(self, x):
        out_list = []
        out = x
        for i, m in enumerate(self.features):
            out = m(out)
            out_list.append(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        out_list.append(out)
        return out, out_list

    def intermediate_forward_MD(self, x, layer_index):
        c = -1
        out = x
        for i, m in enumerate(self.features):
            c += 1
            out = m(out)
            if layer_index == c:
                return out

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        c += 1
        if layer_index == c:
            return out
        return out

    def layer_wise(self, x):
        out_list = []
        out = x
        # out_list.append(out)
        for i, m in enumerate(self.features):
            if isinstance(m, nn.MaxPool2d):
                out = m(out)
                out_list.append(out)
            else:
                out = m(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        out_list.append(out)
        return out_list
        
    def layer_wise_odds_are_odd(self, x):
        out_list = []
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)

        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out_list.append(out)
        out = self.fc3(out)
        out_list.append(out)
        return out_list

    def libre_forward(self, x, return_features):
        out_1 = self.features[:-2](x)
        if out_1.dim() == 5:
            out = self.features[-2](out_1.flatten(0,1)).view((*out_1.shape[:3], 1,1))
            out = self.features[-1](out.flatten(0, 1)).view(*out.shape[:3])
            out = out.view(*out.shape[:2], -1)
            out = self.dropout(out)
            out = self.relu1(self.fc1(out.flatten(0, 1))).view(*out.shape[:2], -1)
            out = self.relu2(self.fc2(out.flatten(0, 1))).view(*out.shape[:2], -1)
            out = self.fc3(out.flatten(0, 1)).view(*out.shape[:2], -1)
        else:
            out = self.features[-2:](out_1)
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.relu1(self.fc1(out))
            out = self.relu2(self.fc2(out))
            out = self.fc3(out)
        if return_features:
            return out_1, out
        else:
            return out


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
