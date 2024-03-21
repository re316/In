import os

import torch
import torch.nn as nn
import torch.nn.functional as F



class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.output_dim = 10

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def feature_func(self, image):
        results = [image]
        x1 =  F.relu(F.max_pool2d(self.conv1(image), 2))
        results.append(x1)
        x2 = self.conv2_drop(self.conv2(x1))
        results.append(x2)
        x3 = F.relu(F.max_pool2d(x2, 2))
        results.append(x3)
    
        return results

    def mc_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def deep_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        return x

    def feature_list_MD(self, x):
        out_list = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        out_list.append(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        out_list.append(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        out_list.append(x)
        x = self.fc2(x)
        out_list.append(x)
        return x, out_list

    def intermediate_forward_MD(self, x, layer_index):
        c = -1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        c += 1
        if layer_index == c:
            return x
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        c += 1
        if layer_index == c:
            return x
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        c += 1
        if layer_index == c:
            return x
        x = self.fc2(x)
        c += 1
        if layer_index == c:
            return x
        return x

    def layer_wise(self, x):
        out_list = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        out_list.append(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        out_list.append(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        out_list.append(x)
        x = self.fc2(x)
        out_list.append(x)
        return out_list

    def layer_wise_odds_are_odd(self, x):
        out_list = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        out_list.append(x)
        x = self.fc2(x)
        out_list.append(x)
        return out_list

    def libre_forward(self, x, return_features):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        out_1 = self.conv2(x)
        if out_1.dim() == 5:
            x = self.conv2_drop(out_1)
            x = F.relu(F.max_pool2d(x.flatten(0,1), 2).view((*x.shape[:3],4,4)))
            x = x.view(*x.shape[:2], -1)
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc1(x.flatten(0,1)).view(*x.shape[:2], -1))
            x = self.fc2(x.flatten(0,1)).view(*x.shape[:2], -1)
        else:
            x = self.conv2_drop(out_1)
            x = F.relu(F.max_pool2d(x, 2))
            x = x.view(x.size(0), -1)
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        if return_features:
            return out_1, x
        else:
            return x




















