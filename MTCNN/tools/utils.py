# -*- coding: UTF-8 -*-
import numpy as np 
import torch
def IoU(box, boxes):

    # x1,y1,x2,y2
    x1 = box[0]
    y1 = box[1] 

    x2 = box[2]
    y2 = box[3]


    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    area = (boxes[:,2] - boxes[:,0] + 1) * (boxes[:,3] - boxes[:,1] + 1)

    # 得到相交的左上右下标签
    xx1 = np.maximum(x1, boxes[:,0])
    yy1 = np.maximum(y1, boxes[:,1])
    xx2 = np.minimum(x2, boxes[:,2])
    yy2 = np.minimum(y2, boxes[:,3])

    # 计算w，h对于相交的区域
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # 内部的区域面积大小
    inter = w*h
    overlap = inter/(box_area+area - inter)

    return overlap

def nms(dets, thresh, mode='Union'):

    dets = dets.cpu().detach().numpy() 
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    scores = dets[:,4]

    areas = (x2 - x1 + 1)*(y2 - y1 + 1)
    rank = scores.argsort()[::-1]
    keep=[]
    while rank.size > 0:
        i = rank[0]
        keep.append(i)
        xx1 = np.maximum(x1[i],x1[rank[1:]])
        yy1 = np.maximum(y1[i],y1[rank[1:]])
        xx2 = np.minimum(x2[i],x2[rank[1:]])
        yy2 = np.minimum(y2[i],y2[rank[1:]]) 
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w*h

        if mode == "Union":
            ovr = inter/(areas[i] + areas[rank[1:]]- inter)
        elif mode == "Minimum":
            ovr = inter/ np.minimum(areas[i], areas[rank[1:]])

        indx = np.where(ovr <= thresh)[0]
        rank = rank[indx+1]

    return keep
