# -*- coding: UTF-8 -*-
import numpy as np 

def IoU(box, boxes):

    # x1,y1,x2,y2
    x1 = box[0]
    y1 = box[1] 

    x2 = box[2]
    y2 = box[3]


    box_area = (x2 - x1 + 0.1) * (y2 - y1 + 0.1)
    area = (boxes[:,2] - boxes[:,0] + 0.1) * (boxes[:,3] - boxes[:,1] + 0.1)

    # 得到相交的左上右下标签
    xx1 = np.maximum(x1, boxes[:,0])
    yy1 = np.maximum(y1, boxes[:,1])
    xx2 = np.minimum(x2, boxes[:,2])
    yy2 = np.minimum(y2, boxes[:,3])

    # 计算w，h对于相交的区域
    w = np.maximum(0, xx2 - xx1 + 0.1)
    h = np.maximum(0, yy2 - yy1 + 0.1)

    # 内部的区域面积大小
    inter = w*h
    overlap = inter/(box_area+area - inter)

    return overlap