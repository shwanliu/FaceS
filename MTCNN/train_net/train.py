# -*- coding: UTF-8 -*-
import datetime
import os
import sys
sys.path.append('..')
sys.path.append('../..')
from core.net import PNet,RNet,ONet,LossFn
from core.image_reader import TrainImageReader
import torch
from core.image_tools import convert_image_to_tensor
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from log.logger import Logger

def computer_accuracy(prob_cls, gt_cls,th=0.6):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    mask = torch.ge(gt_cls, 0)

    # vaild_gt_cls = gt_cls[ gt_cls >= 0 ]
    # vaild_prob_cls = prob_cls[ gt_cls >= 0 ]

    vaild_gt_cls = torch.masked_select(gt_cls,mask)
    vaild_prob_cls  = torch.masked_select(prob_cls,mask)

    size = min(vaild_gt_cls.size()[0],vaild_prob_cls.size()[0])
    

    # prob_ones = torch.ge(vaild_prob_cls, th).float()
    prob_ones = torch.ge(vaild_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,vaild_gt_cls).float()

    # right_ones = torch.eq(vaild_gt_cls,prob_ones).float()

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))

def train_pnet(modelPath, end_epoch, imdb ,batch_size,frequent=10, base_lr=0.01,use_cuda=True):

    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    lossfn = LossFn()
    net = PNet(is_train=True, use_cuda=use_cuda)
    net.train()

    if use_cuda:
        print("use GPU for running net")
        net.cuda()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    print("Loading imageDB.....")
    print("batch_size: %d"%batch_size)
    train_data = DataLoader(imdb,batch_size=batch_size,shuffle=True,num_workers=16,pin_memory=True)
    print("Loaded over")
    for cur_epoch in range(1,end_epoch+1):
        # # shuffle

        for batch_idx, data in enumerate(train_data):
            # 将图片转为tensor
            im_tensor =  data['image'].float()
            # print(im_tensor.dtype)
            # print(im_tensor.shape)
            gt_label =   data['label'].float()
            gt_bbox  =   data['bbox'].float()
            gt_landmark =data['landmark'].float()

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()
            
            cls_pred, box_pred, _ = net(im_tensor)
            # print("cls_pred:",cls_pred)
            # print("box_pred:",box_pred)
            cls_loss = lossfn.cls_loss(gt_label,cls_pred)

            bbox_loss = lossfn.box_loss(gt_label, gt_bbox, box_pred)

            # landmark_loss = lossfn.landmark_loss(gt_label, gt_landmark ,)

            all_loss = cls_loss*1.0 + bbox_loss*0.5 

            if batch_idx % frequent == 0:

                accuracy = (cls_pred >=0.6 ).float().mean()
    
                cls_loss_show = float(cls_loss.item())
                bbox_loss_show = float(bbox_loss.item())
                all_loss_show = float(all_loss.item())

                # (1) Log the scalar values
                info = {
                    'loss': all_loss_show,
                    'accuracy': accuracy
                }

                # for tag, value in info.items():
                #     # print(tag,value)
                #     Logger.scalar_summary()

                print("%s : Epoch: %d, step:%d,accuracy:%0.5f, det_loss:%0.5f, bbox_loss:%0.5f, all_loss:%0.5f, lr:%s"%(
                    datetime.datetime.now(),cur_epoch,batch_idx,accuracy,cls_loss_show,bbox_loss_show,all_loss_show, base_lr
                ))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(modelPath,'Pnet_epoch_%d.pt')%cur_epoch)
        torch.save(net,os.path.join(modelPath,'Pnet_epoch_%d.pkl')%cur_epoch,)

def train_rnet(modelPath, end_epoch, batch_size, frequent=10, base_lr=0.01, use_cuda=False):

    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    lossfn=LossFn()
    net = RNet(is_train=True, use_cuda=False)
    net.train()

    if use_cuda:
        net.cuda

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    for cur_epoch in range(1,end_epoch):
        start =  datetime.datetime.now()
        print(start)
        # 不使用图片数据训练
        im_tensor = torch.randn(5,3,24,24)
        gt_label = torch.randint(2,(5,1)).float()
        gt_bbox = torch.randn(5,4,1,1)
        gt_landmark = torch.randn(5,10,1,1)

        # 网络的输出由三部分组成
        cls_pred, box_pred, _ = net(im_tensor)
        cls_loss = lossfn.cls_loss(gt_label, cls_pred)
        bbox_loss = lossfn.box_loss(gt_label, gt_bbox, box_pred)
        
        all_loss = cls_loss*1.0 + bbox_loss*0.5

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        print(datetime.datetime.now() - start)
        torch.save(net.state_dict(), os.path.join(modelPath,'Rnet_epoch_%d.pt'%cur_epoch))

def train_onet(modelPath, end_epoch, batch_size, frequent=10, base_lr=0.01, use_cuda=False):

    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    lossfn = LossFn()
    net = ONet()
    net.train()

    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    for cur_epoch in range(1,end_epoch):
        start = datetime.datetime.now()
        print(start)
        # 不使用图片数据进行训练
        im_tensor = torch.randn(5,3,48,48)
        gt_label = torch.randint(1,(5,1)).float()
        gt_bbox = torch.randn(5,4,1,1)
        gt_landmark = torch.randn(5,10,1,1)
        
        # 网络的输出
        cls_ , box_pred ,pred_landmark = net(im_tensor)
        cls_loss = lossfn.cls_loss(gt_label, cls_)
        bbox_loss = lossfn.box_loss(gt_label,gt_bbox,box_pred)
        landmark_loss = lossfn.landmark_loss(gt_label, gt_landmark,pred_landmark)
       
        all_loss = cls_loss*0.5 + bbox_loss*0.5 + landmark_loss*1

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        print(datetime.datetime.now() - start)
        torch.save(net.state_dict(), os.path.join(modelPath,'Onet_epoch_%d.pt'%cur_epoch))

if __name__=="__main__":

    # modelPath, end_epoch,imdb, batch_size,frequent=10, base_lr=0.01,use_cuda=False
    # train_pnet('pnet',20,16,10,0.01,False)
    # train_rnet('rnet',20,16,10,0.01,False)
    train_onet('onet',20,16,10,0.01,False)