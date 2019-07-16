# -*- coding: UTF-8 -*-
import cv2
import time
import numpy as np 
import torch
import sys
sys.path.append('..')
from core.net import PNet,RNet,ONet
from core.image_tools import imgResize
import torchvision.transforms as transforms
from tools.utils import nms
transforms = transforms.ToTensor()

def create_mtcnn_net(p_model=None, r_model=None, o_model=None, use_cuda=True):
    
    pnet,rnet,onet = None, None, None

    if p_model is not None:
        pnet = PNet(use_cuda=use_cuda)
        if use_cuda:
            print('p_model:{0}'.format(p_model))
            pnet.load_state_dict(torch.load(p_model))
            pnet.cuda()
        else:
            pnet.load_state_dict(torch.load(p_model, map_location=lambda storage, loc:storage))

        pnet.eval()

    return pnet


class MtcnnDetector(object):
    def __init__(self, pnet=None, rnet=None, onet=None,
            min_face_size=12,
            threshold=[0.6, 0.7, 0.7],
            scale_factor=0.709):
        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.threshold = threshold
        self.scale_factor = scale_factor
    
    def generate_bbox(self, cls_map, reg, scale, threshold):
        stride = 2
        cellsize = 12 

        cls_map = cls_map.squeeze(0)
        reg = reg.squeeze(0)
        cls_map = cls_map.permute(1,2,0)
        reg = reg.permute(1,2,0)

        mask = torch.nonzero(cls_map >= threshold)

        if mask.size(0) == 0:
            return torch.tensor([])

        reg = reg[mask[:,0], mask[:,1], :]

        score = cls_map[cls_map >= threshold].view(-1,1)
        mask = mask.float()
        x1 = torch.round((stride*mask[:,1])/scale)
        x1 = x1.unsqueeze(-1)
        y1 = torch.round((stride*mask[:,0]) /scale)
        y1 = y1.unsqueeze(-1)
        x2 = torch.round((stride*mask[:,1] + cellsize) /scale) -1
        x2 = x2.unsqueeze(-1)
        y2 = torch.round((stride*mask[:,0] + cellsize) /scale) -1
        y2 = y2.unsqueeze(-1)
        boxes = torch.cat((x1, y1, x2, y2, score, reg),1)
        return boxes

    def resize_image(self, img, scale):
        # print("resize_image: %d",scale)
        height, width, channels = img.shape
        new_height = int(height*scale)
        new_width = int(width*scale)
        img_resized = cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_LINEAR)
        return img_resized
    def detect_pnet(self,im,use_cuda=False):
        
        h,w,c = im.shape
        net_size = 12
        current_scale = float(net_size) / self.min_face_size
        im_resized = self.resize_image(im,current_scale)
        current_height, current_width,_  = im_resized.shape

        all_boxes = list()
        i = 0
        while min(current_height, current_width) > net_size:
            feed_imgs = []
            image_tensor = transforms(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs)

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            cls_map,reg,_ = self.pnet_detector(feed_imgs)
            # import pdb
            # pdb.set_trace()
            # print("cls_map type is %s"%cls_map.type)
            # print("reg type is %s"%reg.type)


            boxes = self.generate_bbox(cls_map, reg, current_scale, self.threshold[0])

            # 图像金字塔部分
            current_scale *= self.scale_factor
            im_resized = self.resize_image(im , current_scale)
            current_height,current_width,_  = im_resized.shape

            if boxes.size(0) == 0:
                continue

            # 需要消化一下
            keep = nms(boxes[:, 0:5], 0.5, 'Union')
            if len(keep)==0:
                continue
            # print(keep)
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes)==0:
            return None,None
        all_boxes = torch.cat([ boxes for boxes in all_boxes])
        keep = nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes=all_boxes[keep]

        bw = all_boxes[:,2] - all_boxes[:,0] + 1
        bh = all_boxes[:,3] - all_boxes[:,1] + 1

        boxes = torch.stack([all_boxes[:,0], all_boxes[:,1], all_boxes[:,2], all_boxes[:,3], all_boxes[:,4]])

        align_topx = all_boxes[:,0] + all_boxes[:,5] * bw
        align_topy = all_boxes[:,1] + all_boxes[:,6] * bh
        align_bottomx = all_boxes[:,2] + all_boxes[:,7] * bw
        align_bottomy = all_boxes[:,3] + all_boxes[:,8] * bh

        boxes_align = torch.stack([ align_topx,
                            align_topy,
                            align_bottomx,
                            align_bottomy,
                            all_boxes[:,4],
                            # align_topx + all_boxes[:,9] * bw,
                            # align_topy + all_boxes[:,10] * bh,
                            # align_topx + all_boxes[:,11] * bw,
                            # align_topy + all_boxes[:,12] * bh,
                            # align_topx + all_boxes[:,13] * bw,
                            # align_topy + all_boxes[:,14] * bh,
                            # align_topx + all_boxes[:,15] * bw,
                            # align_topy + all_boxes[:,16] * bh,
                            # align_topx + all_boxes[:,17] * bw,
                            # align_topy + all_boxes[:,18] * bh,
                            ])
        # boxes_align = boxes_align.T

        return boxes.permute(1,0), boxes_align

    def detect_face(self,img,use_cuda=False):
        start = time.time()

        # pnet_detect
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])
        
        usedTime=time.time()-start
        print("imgSize:%s   DetectTime:%0.4fs "%(str(img.shape),usedTime))

        return boxes, boxes_align