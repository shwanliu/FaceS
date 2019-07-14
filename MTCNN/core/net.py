# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)

# 定义好模型然后定义loss的使用s
class LossFn:
    # meiy
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function 
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.landmark_factor = landmark_factor

        self.loss_cls = nn.BCELoss() 
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

    def cls_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)   
        gt_label = torch.squeeze(gt_label) 

        vaild_gt_label = gt_label[gt_label >= 0]
        vaild_pred_label = pred_label[gt_label >= 0]
        return self.loss_cls(vaild_pred_label,vaild_gt_label)*self.cls_factor

    def box_loss(self,gt_label, gt_offset, pred_offset):
        pred_offset =torch.squeeze(pred_offset)
        gt_label = torch.squeeze(gt_label)
        gt_offset = torch.squeeze(gt_offset)

        #如果label是0的话， 则我们不需要进行box的回归
        valid_gt_offset = gt_offset[ (gt_label == 1) | (gt_label == -1)]
        valid_pred_offset  = pred_offset[ (gt_label == 1) | (gt_label == -1)]

        return self.loss_box(valid_pred_offset,valid_gt_offset)*self.box_factor

    def landmark_loss(self,gt_label, gt_landmark, pred_landmark):
        gt_label = torch.squeeze(gt_label)
        gt_landmark = torch.squeeze(gt_landmark)
        pred_landmark = torch.squeeze(pred_landmark)
        #对于gt label为-2的样本进行landmark的回归
        # mask =torch.eq(gt_label,-2)

        # chose_index = torch.nonzero(mask.data)
        # chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[ gt_label == -2]
        valid_pred_landmark= pred_landmark[ gt_label == -2]
     
        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)*self.landmark_factor

# PNet
class PNet(nn.Module):
    def __init__(self, is_train=False, use_cuda=True):
        super(PNet,self).__init__()

        self.is_train = is_train
        self.use_cuda  = use_cuda
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,10,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(10,16,kernel_size=3,stride=1),
            nn.PReLU(),
            
            nn.Conv2d(16,32,kernel_size=3,stride=1),
            nn.PReLU()

        # ('conv_1', nn.Conv2d(3,10,3,1)),
        # ('prelu_1', nn.PReLU(10)),
        # ('maxpool', nn.MaxPool2d(2,2,ceil_mode=True)),
        # ('conv_2', nn.Conv2d(10,16,3,1)),
        # ('prelu_2', nn.PReLU(16)),
        # ('conv_2', nn.Conv2d(16,32,3,1)),
        # ('prelu_2', nn.PReLU(32))
        )

        #人脸分类
        self.cls = nn.Conv2d(32,2,1,1)
        #人脸框
        self.bbox = nn.Conv2d(32,4,1,1)
        # 关键点
        self.landMark = nn.Conv2d(32,10,1,1)
        self.apply(weight_init)

    def forward(self, x):
        x = self.pre_layer(x)
        bbox_ = self.bbox(x)
        cls_ = self.cls(x)
        landMark_  = self.landMark(x)
        cls_ = torch.sigmoid(cls_)
        
        return cls_ , bbox_ , landMark_

class RNet(nn.Module):
    def __init__(self,is_train=False, use_cuda=True):
        super(RNet,self).__init__()
        
        self.is_train = is_train
        self.use_cuda  = use_cuda
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,28,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            
            nn.Conv2d(28,48,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(48,64,kernel_size=2,stride=1),
            nn.PReLU()
        )
        #     ('conv_1', nn.Conv2d(3,28,3,1)),
        #     ('prelu_1', nn.PReLU(28)),
        #     ('maxpol', nn.MaxPool2d(3,2,ceil_mode=True)),
        #     ('conv_2', nn.Conv2d(28,48,3,1))
        #     ('prelu_2', nn.PReLU(48)),
        #     ('maxpool', nn.MaxPool2d(3,2,ceil_mode=True))
        #     ('conv_3', nn.Conv2d(48,64,2,1)),
        #     ('prelu_3', nn.PReLU(64)),

        self.fc_1 = nn.Linear(64*2*2,128)
        self.prelu_1 = nn.PReLU()

         #人脸分类
        self.cls = nn.Linear(128, 1)
        #人脸框
        self.bbox = nn.Linear(128, 4)
        # 关键点
        self.landMark = nn.Linear(128, 10)
        self.apply(weight_init)

    def forward(self,x):
        x = self.pre_layer(x)
        # 将x改为只有一维的向量，便于全连接
        x = x.view(x.size(0),-1)
        x = self.fc_1(x)
        x = self.prelu_1(x)
        # 人脸分类,需要使用到sigmoid规范化他的输出结果
        cls_ = torch.sigmoid(self.cls(x))
        # bbokx
        bbox_ = self.bbox(x)
        # landmark
        landMark_ = self.landMark(x)
        return cls_, bbox_,landMark_
    
class ONet(nn.Module):
    def __init__(self,is_train=False, use_cuda=True):
        super(ONet,self).__init__()
        
        self.is_train = is_train
        self.use_cuda  = use_cuda
        self.pre_layer =nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=2,stride=1),
            nn.PReLU()
        )

        self.fc_1 = nn.Linear(2*2*128,256)
        self.prelu_1 = nn.PReLU()
        # 人脸类别
        self.cls = nn.Linear(256,1)
        # 框回归
        self.bbox = nn.Linear(256,4)
        # 关键点回归
        self.landmark = nn.Linear(256,10)
        self.apply(weight_init)
    
    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)
        x = self.fc_1(x)
        x = self.prelu_1(x)

        # det
        cls_ = torch.sigmoid(self.cls(x))
        # bbox
        bbox_ = self.bbox(x)
        # landmark
        landmark_ = self.landmark(x)

        return cls_ , bbox_ , landmark_



    