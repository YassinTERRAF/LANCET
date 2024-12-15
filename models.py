#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from area_attention import AreaAttention
from g_mlp_pytorch import gMLP
from g_mlp_pytorch import SpatialGatingUnit
from asteroid.masknn import norms

class MultiConv(nn.Module):
    '''
    Multi-scale block without short-cut connections
    '''
    def __init__(self, channels = 16, **kwargs):
        super(MultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x3,x5),1)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ResMultiConv(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3,x5),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class ResConv3(nn.Module):
    '''
    Resnet with 3x3 kernels
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResConv3, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=2*channels, padding=1)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):        
        x = self.conv3(x) + torch.cat((x,x),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class ResConv5(nn.Module):
    '''
    Resnet with 5x5 kernels
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResConv5, self).__init__()
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=2*channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):        
        x = self.conv5(x) + torch.cat((x,x),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class CNN_Area(nn.Module):
    '''Area attention, Mingke Xu, 2020
    '''
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(CNN_Area, self).__init__()
        self.height=height
        self.width=width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        i = 80 * ((shape[0]-1)//2) * ((shape[1]-1)//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

class CNN_AttnPooling(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(CNN_AttnPooling, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.top_down = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=4)
        self.bottom_up = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=1)
        # i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4) 
        # self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)

        x1 = self.top_down(x)
        x1 = F.softmax(x1,1)
        x2 = self.bottom_up(x)

        x = x1 * x2

        # x = x.sum((2,3))        
        x = x.mean((2,3))        

        return x

class CNN_GAP(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(CNN_GAP, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MHCNN(nn.Module):
    '''
    Multi-Head Attention
    '''
    def __init__(self, head=4, attn_hidden=64,**kwargs):
        super(MHCNN, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=self.attn_hidden, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHCNN_AreaConcat, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden * ((shape[0]-1)//2) * ((shape[1]-1)//4) * self.head
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        # x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat_gap(nn.Module):
    def __init__(self, head=4, attn_hidden=64,**kwargs):
        super(MHCNN_AreaConcat_gap, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden 
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat_gap1(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHCNN_AreaConcat_gap1, self).__init__()
        self.head = head
        self.attn_hidden = 32
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden 
        self.fc = nn.Linear(in_features=(shape[0]-1)//2, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 3)
        x = attn
        x = x.contiguous().permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

class AACNN(nn.Module):
    '''
    Area Attention, ICASSP 2020
    '''
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(AACNN, self).__init__()
        self.height=height
        self.width=width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        i = 80 * ((shape[0] - 1)//2) * ((shape[1] - 1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=height,
            max_area_width=width,
            dropout_rate=0.5,
        )


    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x,x,x)
        x = F.relu(x)
        x = x.reshape(*shape)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class AACNN_HeadConcat(nn.Module):
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(AACNN_HeadConcat, self).__init__()
        self.height=height
        self.width=width
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        i = 80 * ((shape[0] - 1)//4) * ((shape[1] - 1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=height,
            max_area_width= width,
            dropout_rate=0.5,
            # top_k_areas=0
        )


    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        
        x = self.area_attention(x,x,x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class GLAM5(nn.Module):
    '''
    GLobal-Aware Multiscale block with 5x5 convolutional kernels in CNN architecture
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(GLAM5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=16, padding=(2, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=16, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class GLAM(nn.Module):
    '''
    GLobal-Aware Multiscale block with 3x3 convolutional kernels in CNN architecture
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(GLAM, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class gMLPResConv5(nn.Module):
    '''
    GLAM5 - Multiscale
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResConv5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResConv5(16)
        self.conv3 = ResConv5(32)
        self.conv4 = ResConv5(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPMultiConv(nn.Module):
    '''
    GLAM - Resnet
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPMultiConv, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = MultiConv(16)
        self.conv3 = MultiConv(32)
        self.conv4 = MultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MHResMultiConv3(nn.Module):
    '''
    Multi-Head-Attention with Multiscale blocks
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        i = self.attn_hidden * self.head * (shape[0]//2) * (shape[1]//4)
        self.fc = nn.Linear(in_features=i, out_features=4)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class AAResMultiConv3(nn.Module):
    '''
    Area Attention with Multiscale blocks
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(AAResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)

        i = 128 * (shape[0]//2) * (shape[1]//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=3,
            max_area_width=3,
            dropout_rate=0.5,
        )

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x,x,x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class ResMultiConv3(nn.Module):
    '''
    GLAM - gMLP
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(ResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class ResMultiConv5(nn.Module):
    '''
    GLAM5 - gMLP
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(ResMultiConv5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=16, padding=(2, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=16, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPgResMultiConv3(nn.Module):
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPgResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//4) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())
        self.sgu = SpatialGatingUnit(dim = shape[0] * shape[1] * 2, dim_seq = 16, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 26, 63)
        xa = self.bn1a(xa) # (32, 16, 26, 63)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        shape = x.shape
        x = x.view(*x.shape[:-2],-1)
        x = self.sgu(x)
        x = x.view(shape[0], shape[1], shape[2]//2, shape[3])

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class aMLPResMultiConv3(nn.Module):
    def __init__(self, shape=(26,63), **kwargs):
        super(aMLPResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, attn_dim = 64, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPResMultiConv35(nn.Module):
    '''
    Temporal and Spatial convolution with multiscales
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResMultiConv35, self).__init__()
        self.conv1a1 = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=8, padding=(1, 0))
        self.conv1a2 = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=8, padding=(2, 0))
        self.conv1b1 = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=8, padding=(0, 1))
        self.conv1b2 = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=8, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa1 = self.conv1a1(input[0]) # (32, 8, 26, 63)
        xa2 = self.conv1a2(input[0]) # (32, 8, 26, 63)
        xa = torch.cat((xa1,xa2),1)
        xa = self.bn1a(xa) # (32, 16, 26, 63)
        xa = F.relu(xa)

        xb1 = self.conv1b1(input[0])
        xb2 = self.conv1b2(input[0])
        xb = torch.cat((xb1,xb2),1)
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return 



class ResMultiConv2(nn.Module):
    '''
    Multi-scale block with short-cut connections, modified to include a 7x7 convolution.
    '''
    def __init__(self, channels=16, **kwargs):
        super(ResMultiConv2, self).__init__()

        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.conv7 = nn.Conv2d(kernel_size=(7, 7), in_channels=channels, out_channels=channels, padding=3)

        self.bn = nn.BatchNorm2d(channels * 3)  # Output channels are tripled

    def forward(self, x):

        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        x = torch.cat((x3, x5, x7), 1)  # Concatenate along the channel dimension
        
        x = self.bn(x)

        x = F.relu(x)

        return x


class GLAMv2(nn.Module):
    '''
    Adjusted GLobal-Aware Multiscale block with multiple convolutional kernels and parallel branches.
    '''
    def __init__(self, shape=(26, 63), **kwargs):
        super(GLAMv2, self).__init__()


        # Multi-dimensional emotion feature extraction
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv1c = nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=16, padding=(1, 1))

        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn1c = nn.BatchNorm2d(16)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Parallel multi-scale feature extraction networks
        # First branch
        self.conv2a = ResMultiConv(48)  # Input channels: 16 (conv1a) + 16 (conv1b) + 16 (conv1c) = 48
        self.conv3a = ResMultiConv(96)  # Output of conv2a is 48*3=144 after concatenation in ResMultiConv
        self.conv4a = ResMultiConv(192)  # Output of conv3a is 144*3=432 after concatenation in ResMultiConv
        
        # Second branch
        self.conv2b = ResMultiConv(48)
        self.conv3b = ResMultiConv(96)
        self.conv4b = ResMultiConv(192)
        
        # Final convolutional layer after concatenation
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=768, out_channels=128, padding=2)  # Input channels are doubled
        self.bn5 = nn.BatchNorm2d(128)

        # Global feature fusion
        dim = (shape[0] // 2 // 2) * (shape[1] // 2 // 2)  # After two pooling layers
        self.gmlp = gMLP(dim=dim, depth=1, seq_len=128, act=nn.Tanh())

        # Fully connected layer for classification
        i = 128 * dim  # Corrected dimension
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # Multi-dimensional emotion feature extraction
        xa = self.conv1a(x)  # (32, 16, 26, 63)
        xa = self.bn1a(xa)
        xa = F.relu(xa)
        
        xb = self.conv1b(x)  # (32, 16, 26, 63)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        
        xc = self.conv1c(x)  # (32, 16, 26, 63)
        xc = self.bn1c(xc)
        xc = F.relu(xc)

        # Concatenate spatial, temporal, and spatial-temporal features
        x = torch.cat((xa, xb, xc), 1)  # (32, 48, 26, 63)

        # First branch
        x1 = self.conv2a(x)  # (32, 144, 26, 63)
        x1 = self.maxp(x1)  # (32, 144, 13, 31)

        x1 = self.conv3a(x1)  # (32, 432, 13, 31)
        x1 = self.maxp(x1)  # (32, 432, 6, 15)

        x1 = self.conv4a(x1)  # (32, 432, 6, 15)

        # Second branch
        x2 = self.conv2b(x)  # (32, 144, 26, 63)
        x2 = self.maxp(x2)  # (32, 144, 13, 31)

        x2 = self.conv3b(x2)  # (32, 432, 13, 31)
        x2 = self.maxp(x2)  # (32, 432, 6, 15)

        x2 = self.conv4b(x2)  # (32, 432, 6, 15)

        # Concatenate features from both branches
        x = torch.cat((x1, x2), 1)  # (32, 432 + 432, 6, 15) -> (32, 864, 6, 15)

        # Apply final convolutional layer
        x = self.conv5(x)  # (32, 864, 6, 15) -> (32, 128, 6, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # Flatten and apply gMLP
        x = x.view(x.shape[0], x.shape[1], -1)  # (32, 128, 90)
        
        x = self.gmlp(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)  # Flatten for the fully connected layer

        # Classification
        x = self.fc(x)
        return x




class SoftAttentionPooling(nn.Module):
    def __init__(self, num_frames):
        super(SoftAttentionPooling, self).__init__()
        self.W = nn.Parameter(torch.randn(1, num_frames))  # Trainable parameter W of size 1xnum_frames

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, feature_dim, num_frames]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, feature_dim]
        """
        # Transpose x to [batch_size, num_frames, feature_dim] for multiplication
        x_transposed = x.permute(0, 2, 1)  # Shape: [batch_size, num_frames, feature_dim]


        # Compute attention scores (unnormalized)
        attention_scores = torch.tanh(torch.matmul(x_transposed, self.W.T))  # Shape: [batch_size, num_frames, 1]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, num_frames, 1]
    
        # Apply attention weights to the input tensor by multiplying weights with features
        weighted_sum = torch.sum(attention_weights * x_transposed, dim=1)  # Shape: [batch_size, feature_dim]



        return weighted_sum
   


class ChannelTimeSenseSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10], subband_num=1):
        super(ChannelTimeSenseSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio

        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        small_feature = self.smallConv1d(input_tensor)

        middle_feature = self.middleConv1d(input_tensor)

        large_feature = self.largeConv1d(input_tensor)

        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)

        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))

        return output_tensor


class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, kernel_size, padding, dilation, norm_type="bN", delta=False):
        super(Conv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)
        self.delta = delta
        if delta:
            self.linear = nn.Linear(in_chan, in_chan)
            self.linear_norm = norms.get(norm_type)(in_chan*2)

        in_bottle = in_chan if not delta else in_chan*2
        in_conv1d = nn.Conv1d(in_bottle, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        if self.delta:
            delta = self.linear(x.transpose(1, -1)).transpose(1, -1)
            x = torch.cat((x, delta), 1)
            x = self.linear_norm(x)

        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out











class TCN(nn.Module):
    def __init__(self, in_chan=128, shape=(64, 81), out_classes=4, n_blocks=5, bn_chan=64, n_repeats=3, hid_chan=128, kernel_size=3, norm_type="gLN", **kwargs):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        # self.attention = ChannelTimeSenseSELayer(in_chan)
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2**x, norm_type=norm_type))

        out_conv = nn.Conv1d(bn_chan, out_classes, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Final linear layer for classification after pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_final = nn.Linear(bn_chan, out_classes)
    def forward(self, x):
        x = x.squeeze(1)  # Remove the channel dimension; shape: [batch_size, 128, 81]
        # x = self.attention(x)  # Apply attention layer
        x = self.bottleneck(x)  # Bottleneck to reduce dimensions; shape: [batch_size, bn_chan, 81]
        
        for block in self.TCN:
            residual = block(x)
            x = x + residual

        # Perform global average pooling to get representative features
        x = self.global_avg_pool(x).squeeze(-1)  # shape: [batch_size, bn_chan]
        
        # Final classification
        output = self.fc_final(x)  # shape: [batch_size, out_classes]

        return output




class TCN_attention(nn.Module):
    def __init__(self, in_chan=128, shape=(64, 81), out_classes=4, n_blocks=5, bn_chan=64, n_repeats=3, hid_chan=128, kernel_size=3, norm_type="gLN", **kwargs):
        super(TCN_attention, self).__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.attention = ChannelTimeSenseSELayer(in_chan)
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2**x, norm_type=norm_type))

        out_conv = nn.Conv1d(bn_chan, out_classes, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Final linear layer for classification after pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_final = nn.Linear(bn_chan, out_classes)

    def forward(self, x):

        print(x.shape)
        x = x.squeeze(1)  # Remove the channel dimension; shape: [batch_size, 128, 81]
        x = self.attention(x)  # Apply attention layer
        x = self.bottleneck(x)  # Bottleneck to reduce dimensions; shape: [batch_size, bn_chan, 81]
        
        for block in self.TCN:
            residual = block(x)
            x = x + residual

        # Perform global average pooling to get representative features
        x = self.global_avg_pool(x).squeeze(-1)  # shape: [batch_size, bn_chan]
        
        # Final classification
        output = self.fc_final(x)  # shape: [batch_size, out_classes]

        return output






class TCN_soft(nn.Module):
    def __init__(self, in_chan=26, shape=(64, 81), out_classes=4, n_blocks=5, bn_chan=64, n_repeats=3, hid_chan=128, kernel_size=3, norm_type="gLN", **kwargs):
        super(TCN_soft, self).__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2**x, norm_type=norm_type))

        out_conv = nn.Conv1d(bn_chan, out_classes, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Final linear layer for classification after pooling
        self.soft_attention = SoftAttentionPooling(64)
        self.fc_final = nn.Linear(bn_chan, out_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Remove the channel dimension; shape: [batch_size, 128, 81]
        x = self.bottleneck(x)  # Bottleneck to reduce dimensions; shape: [batch_size, bn_chan, 81]
        
        for block in self.TCN:
            residual = block(x)
            x = x + residual

        # Perform attention pooling to get representative features
        x = self.soft_attention(x)
        
        # Final classification
        output = self.fc_final(x)  # shape: [batch_size, out_classes]

        return output




class SoftAttentionPooling_Test(nn.Module):
    def __init__(self, input_dim):
        super(SoftAttentionPooling_Test, self).__init__()
        
        # Learnable context vector (attention query)
        self.context_vector = nn.Parameter(torch.randn(input_dim))

        # Attention mechanism uses a projection to compute scores
        self.attention_projection = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        """
        Input:
        - x: [batch_size, input_dim, num_frames], where input_dim is the feature dimension 
              and num_frames is the number of time steps (frames).
        
        Output:
        - out: [batch_size, input_dim], weighted sum of the features over the time frames.
        """
        # Transpose to [batch_size, num_frames, input_dim]
        x = x.transpose(1, 2)  # Now x is [batch_size, num_frames, input_dim]
        
        # Project each time step's features and compute compatibility with the context vector
        projected_x = torch.tanh(self.attention_projection(x))  # Shape: [batch_size, num_frames, input_dim]
        
        # Compute attention scores as dot products with the context vector
        # Shape: [batch_size, num_frames]
        attention_scores = torch.matmul(projected_x, self.context_vector)
        
        # Apply softmax to get normalized attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [batch_size, num_frames]
        # Compute weighted sum of features (weighted by attention weights)
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), x)  # Shape: [batch_size, 1, input_dim]
        
        # Remove the extra dimension and return
        return weighted_sum.squeeze(1)  # Shape: [batch_size, input_dim]


class TCN_soft_test(nn.Module):
    def __init__(self, in_chan=128, shape=(64, 81), out_classes=4, n_blocks=5, bn_chan=64, n_repeats=3, hid_chan=128, kernel_size=3, norm_type="gLN", **kwargs):
        super(TCN_soft_test, self).__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2**x, norm_type=norm_type))

        out_conv = nn.Conv1d(bn_chan, out_classes, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Final linear layer for classification after pooling
        self.soft_attention = SoftAttentionPooling_Test(64)
        self.fc_final = nn.Linear(bn_chan, out_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Remove the channel dimension; shape: [batch_size, 128, 81]
        x = self.bottleneck(x)  # Bottleneck to reduce dimensions; shape: [batch_size, bn_chan, 81]
        
        for block in self.TCN:
            residual = block(x)
            x = x + residual

        # Perform attention pooling to get representative features
        x = self.soft_attention(x)
        
        # Final classification
        output = self.fc_final(x)  # shape: [batch_size, out_classes]

        return output







class TCN_attention_soft_test(nn.Module):
    def __init__(self, in_chan=128, shape=(64, 81), out_classes=4, n_blocks=5, bn_chan=64, n_repeats=3, hid_chan=128, kernel_size=3, norm_type="gLN", **kwargs):
        super(TCN_attention_soft_test, self).__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.attention = ChannelTimeSenseSELayer(in_chan)
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2**x, norm_type=norm_type))

        out_conv = nn.Conv1d(bn_chan, out_classes, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Final linear layer for classification after pooling
        self.soft_attention = SoftAttentionPooling(64)
        self.fc_final = nn.Linear(bn_chan, out_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Remove the channel dimension; shape: [batch_size, 128, 81]
        x = self.attention(x)  # Apply attention layer
        x = self.bottleneck(x)  # Bottleneck to reduce dimensions; shape: [batch_size, bn_chan, 81]
        
        for block in self.TCN:
            residual = block(x)
            x = x + residual

        # Perform global average pooling to get representative features
        x = self.soft_attention(x)

        # Final classification
        output = self.fc_final(x)  # shape: [batch_size, out_classes]

        return output







# Helper norms dictionary
norms = {
    "bN": nn.BatchNorm1d,
    "gLN": lambda x: nn.GroupNorm(1, x),  # Global Layer Normalization
}

class Conv1DBlock(nn.Module):
    def __init__(self, in_chan, hid_chan, kernel_size, padding, dilation, norm_type="bN"):
        super(Conv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)
        self.block = nn.Sequential(
            nn.Conv1d(in_chan, hid_chan, kernel_size, padding=padding, dilation=dilation),
            nn.PReLU(),
            conv_norm(hid_chan)
        )
        self.res_conv = nn.Conv1d(hid_chan, hid_chan, 1)  # Residual connection back to hidden channels

    def forward(self, x):
        shared_out = self.block(x)
        res_out = self.res_conv(shared_out)
        return res_out + x  # Residual connection


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, kernel_size, norm_type="bN"):
        super(TemporalConvBlock, self).__init__()

        # Three TCN cores with dilation rates 1, 2, and 4
        self.core1 = Conv1DBlock(in_channels, hid_channels, kernel_size, padding=(kernel_size - 1) // 2, dilation=1, norm_type=norm_type)
        self.core2 = Conv1DBlock(hid_channels, hid_channels, kernel_size, padding=(kernel_size - 1) * 2 // 2, dilation=2, norm_type=norm_type)
        self.core3 = Conv1DBlock(hid_channels, hid_channels, kernel_size, padding=(kernel_size - 1) * 4 // 2, dilation=4, norm_type=norm_type)

    def forward(self, x):
        x = self.core1(x)
        x = self.core2(x)
        x = self.core3(x)
        return x  # Final output of the block

class TCNet(nn.Module):
    def __init__(self, shape=(26,63) , in_features=26, num_frames=63, out_classes=4, hid_channels=26, norm_type="bN",**kwargs):
        super(TCNet, self).__init__()

        # Block 1: Kernel size 3
        self.block1 = TemporalConvBlock(in_channels=in_features, hid_channels=hid_channels, kernel_size=3, norm_type=norm_type)
        # Block 2: Kernel size 5
        self.block2 = TemporalConvBlock(in_channels=in_features, hid_channels=hid_channels, kernel_size=5, norm_type=norm_type)
        # Block 3: Kernel size 9
        self.block3 = TemporalConvBlock(in_channels=in_features, hid_channels=hid_channels, kernel_size=9, norm_type=norm_type)

        # Global average pooling for each block
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(hid_channels * 3, 128)  # Fuse features from all three blocks
        self.fc2 = nn.Linear(128, out_classes)

        # Activation and dropout layers
        self.activation = nn.ReLU()

    def forward(self, x):
        # Input shape: [batch_size, 1, in_features, num_frames]
        x = x.squeeze(1)  # Remove channel dimension; shape: [batch_size, in_features, num_frames]

        # Pass through the three blocks
        out1 = self.block1(x)  # Features from Block 1
        out2 = self.block2(x)  # Features from Block 2
        out3 = self.block3(x)  # Features from Block 3

        # Global average pooling for each block
        out1 = self.global_pool(out1).squeeze(-1)  # Shape: [batch_size, hid_channels]
        out2 = self.global_pool(out2).squeeze(-1)  # Shape: [batch_size, hid_channels]
        out3 = self.global_pool(out3).squeeze(-1)  # Shape: [batch_size, hid_channels]

        # Concatenate pooled outputs
        fused = torch.cat([out1, out2, out3], dim=-1)  # Shape: [batch_size, hid_channels * 3]

        # Fully connected layers
        x = self.fc1(fused)
        x = self.activation(x)
        x = self.fc2(x)  # Shape: [batch_size, out_classes]

        return x



class LANCET(nn.Module):
    def __init__(self, in_chan=128, shape=(64, 81), out_classes=4, n_blocks=5, bn_chan=64, n_repeats=3, hid_chan=128, kernel_size=3, norm_type="gLN", **kwargs):
        super(LANCET, self).__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.attention = ChannelTimeSenseSELayer(in_chan)
        
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, kernel_size, padding=padding, dilation=2**x, norm_type=norm_type))

        out_conv = nn.Conv1d(bn_chan, out_classes, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        
        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Final linear layer for classification after pooling
        self.soft_attention = SoftAttentionPooling(64)
        self.fc_final = nn.Linear(bn_chan, out_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Remove the channel dimension; shape: [batch_size, 128, 81]
        x = self.attention(x)  # Apply attention layer
        x = self.bottleneck(x)  # Bottleneck to reduce dimensions; shape: [batch_size, bn_chan, 81]
        
        for block in self.TCN:
            residual = block(x)
            x = x + residual

        # Perform global average pooling to get representative features
        x = self.soft_attention(x)

        # Final classification
        output = self.fc_final(x)  # shape: [batch_size, out_classes]

        return output







