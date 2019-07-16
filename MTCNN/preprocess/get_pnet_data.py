# -*- coding: UTF-8 -*-
import sys
import numpy as np
import cv2
import os
sys.path.append("..")
sys.path.append(os.getcwd())
from tools.utils import IoU

#图片存放位置 
prefix = '../../datasets/WIDER_train/images'
anno_file = 'anno/wider_train_gt.txt'

im_dir = '../datasets/'

pos_save_dir = '../datasets/train/12/positive'
neg_save_dir = '../datasets/train/12/negative'
part_save_dir = '../datasets/train/12/part'

if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)

if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)

if not os.path.exists(part_save_dir):
    os.makedirs(part_save_dir)

# 保存相应的标注信息
pos_anno = open(os.path.join('../datasets/train/12','pos_12.txt'), 'w')
part_anno = open(os.path.join('../datasets/train/12','part_12.txt'),'w')
neg_anno = open(os.path.join('../datasets/train/12','neg_12.txt'),'w')

# 读取原始标注信息进行处理

with open(anno_file,"r") as f:
    annotations = f.readlines()

num = len(annotations)
# print("total %d picture will process"%num)

pos_idx = 0
neg_idx = 0
part_idx = 0

idx = 0
box_idx = 0

# 已经wider_face的标注由x1,y1,w,h改为x1,y1,x2,y2
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = os.path.join(prefix, annotation[0])
    gt_bbox = list(map(float,annotation[1:]))
    gt_bboxes = np.array(gt_bbox, dtype=np.int32).reshape(-1,4)
    # print(im_path,gt_bbox)
    # print(gt_bboxes)
    img =  cv2.imread(im_path)

    # 图片的熟练
    idx += 1

    if idx %100 ==0:
        print("%d images processed"%idx)
    height, width, channel = img.shape

    # 直接在整个大图片里面进行随机crop操作
    neg_num = 0
    while neg_num <50:
        size = np.random.randint(12, min(height,width)/2)
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height -size)
        crop_box = np.array([nx, ny ,nx+size, ny+size])
        # print(crop_box)
        neg_num+=1
        # IoU的计算
        iou = IoU(crop_box,gt_bboxes)
        # crop在原来的图片上
        cropped_im = img[ny:ny+size , nx:nx+size, :]
        # print("11",img.shape)
        # print(cropped_im.shape)
        resized_im = cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)

        if np.max(iou) < 0.3:
            save_file_name = os.path.join(neg_save_dir,"%s.jpg"%neg_idx)
            neg_anno.write(save_file_name + ' 0\n')
            cv2.imwrite(save_file_name, resized_im)
            neg_idx+=1

    # 根绝gt_bbox的数据进行操作，保证了准确性，不像neg
    for bbox in gt_bboxes:
        x1,y1,x2,y2 = bbox
        w = x2 -x1 + 1
        h = y2 -y1 + 1

        # 忽略掉小的人脸
        if max(w,h) < 40 or x1 < 0 or y1<0:
            continue
        
        for i in range(5):
            size = np.random.randint(12,min(width,height)/2)
            offset_x = np.random.randint(max(-size, -x1), w)
            offset_y = np.random.randint(max(-size, -y1), h)

            nx1 = max(0, x1+offset_x)
            ny1 = max(0, y1+offset_y)

            crop_box = np.array([nx1,ny1,nx1+size,ny1+size])
            iou = IoU(crop_box,gt_bboxes)

            cropped_im = img[ny1:ny1+size, nx1:nx1+size, :]
            resized_im = cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)

            if np.max(iou) < 0.3:
                save_file_name = os.path.join(neg_save_dir,"%s.jpg"%neg_idx)
                neg_anno.write(save_file_name + ' 0\n')
                cv2.imwrite(save_file_name, resized_im)
                neg_idx+=1

        # 生成困难样本和part样本
        for i in range(20):
            size = np.random.randint(int(min(w,h)*0.8), np.ceil(1.25*max(w,h)))
            offset_x = np.random.randint(-0.2*w, w*0.2)
            offset_y = np.random.randint(-0.2*h, h*0.2)

            #nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            # ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx1 = max(0, x1+offset_x)
            ny1 = max(0, y1+offset_y)
            nx2 = nx1+size
            ny2 = ny1+size

            if nx2 > width or ny2 > height:
                continue
            
            crop_box = np.array([nx1, ny1, nx2, ny2])
            
            offset_x1 = (x1 - nx1)/float(size)
            offset_x2 = (x2 - nx2)/float(size)
            offset_y1 = (y1 - ny1)/float(size)
            offset_y2 = (y2 - ny2)/float(size)

            cropped_im = img[ny1:ny2, nx1:nx2, :]
            resized_im = cv2.resize(cropped_im,(12,12),interpolation=cv2.INTER_LINEAR)

            # 将所有
            box_ = bbox.reshape(1,-1)
            if IoU(crop_box, box_)>0.65:
                save_file_name = os.path.join(pos_save_dir,"%s.jpg"%pos_idx)
                pos_anno.write(save_file_name + ' 1 %0.2f %0.2f %0.2f %0.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file_name, resized_im)
                pos_idx+=1

            elif IoU(crop_box, box_)>=0.4:
                save_file_name = os.path.join(part_save_dir,"%s.jpg"%part_idx)
                part_anno.write(save_file_name + ' -1 %0.2f %0.2f %0.2f %0.2f\n'%(offset_x1,offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file_name, resized_im)
                part_idx+=1
        box_idx+=1
        print("%s images processed done, pos: %s part: %s neg: %s"%(idx,pos_idx,part_idx,neg_idx))

neg_anno.close()
part_anno.close()
pos_anno.close()