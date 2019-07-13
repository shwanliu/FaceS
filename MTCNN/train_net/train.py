# -*- coding: UTF-8 -*-
import datetime
import os
import sys
sys.path.append('..')
from core.net import PNet,RNet,ONet,LossFn
from core.image_reader import TrainImageReader
import torch
from core.image_tools import convert_image_to_tensor
from torch.autograd import Variable
import numpy as np

def computer_accuracy(prob_cls, gt_cls,th=0.6):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # mask = torch.ge(gt_cls, 0)

    vaild_gt_cls = gt_cls[ gt_cls >= 0 ]
    vaild_prob_cls = prob_cls[ gt_cls >= 0 ]

    # vaild_gt_cls = torch.masked_select(gt_cls,mask)
    # vaild_prob_cls  = torch.masked_select(prob_cls,mask)

    size = min(vaild_gt_cls.size()[0],vaild_prob_cls.size()[0])
    

    # prob_ones = torch.ge(vaild_prob_cls, th).float()
    prob_ones = vaild_prob_cls[ vaild_prob_cls >= 0].float()
    right_ones = prob_ones[vaild_gt_cls == prob_ones]

    # right_ones = torch.eq(vaild_gt_cls,prob_ones).float()

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))

def train_pnet(modelPath, end_epoch, batch_size,frequent=10, base_lr=0.01,use_cuda=False):

    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    lossfn = LossFn()
    net = PNet(is_train=True, use_cuda=True)
    net.train()

    if use_cuda:
        net.cuda()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    # for cur_epoch in range(1,end_epoch+1):

    #     # start =  datetime.datetime.now()
    #     # print(start)
    #     # 不使用图片数据训练
    #     im_tensor =torch.randn(5,3,12,12)  
    #     gt_label = torch.randint(2,(5,1)).float()
    #     gt_bbox = torch.randn(5,4,1,1)
    #     gt_landmark = torch.randn(5,10,1,1)


    #     # 网络的输出有三个部分，类别预测，bbox预测，以及人脸关键点预测
    #     cls_pred, box_pred, _ = net(im_tensor)
    #     cls_loss = lossfn.cls_loss(gt_label,cls_pred[:, 0])
    #     bbox_loss = lossfn.box_loss(gt_label, gt_bbox, box_pred)
    #     # landmark_loss = lossfn.landmark_loss(gt_label, gt_landmark ,)
    #     all_loss = cls_loss*1.0 + bbox_loss*0.5 

    #     # 
    #     optimizer.zero_grad()
    #     all_loss.backward()
    #     optimizer.step()
    #     # print(datetime.datetime.now() - start)
    #     torch.save(net.state_dict(), os.path.join(modelPath,'Pnet_epoch_%d.pt'%cur_epoch))


    train_data = TrainImageReader(imdb,12,batch_size,shuffle=True)

    for cur_epoch in range(1,end_epoch+1):
        # shuffle
        train_data.reset()

        for batch_idx,(image,(gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):
            
            # 将图片转为tensor
            im_tensor = [ convert_image_to_tensor(image[i,:,:,:]) for i in range(image.size[0])]
            
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())
            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float)

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()
            
            cls_pred, box_pred, _ = net(im_tensor)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)

            bbox_loss = lossfn.box_loss(gt_label, gt_bbox, box_pred)

            # landmark_loss = lossfn.landmark_loss(gt_label, gt_landmark ,)

            all_loss = cls_loss*1.0 + bbox_loss*0.5 

            if batch_idx % frequent == 0:
                acc = computer_acc(cls_pred,gt_label)

                acc_show = acc.data.cpu().numpy()
                cls_loss_show = cls_loss.data.cpu().numpy()
                bbox_loss_show = bbox_loss.data.cpu.numpy()
                all_loss_show = all_loss.data.cpu.numpy()
            
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            torch.save(net.state_dict(), os.path.join(modelPath,'Pnet_epoch_%d.pt'),)


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
        cls_loss = lossfn.cls_loss(gt_label, cls_pred[:, 0])
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
        cls_loss = lossfn.cls_loss(gt_label, cls_[:,0])
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