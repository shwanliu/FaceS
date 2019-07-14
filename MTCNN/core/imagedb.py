# -*- coding: UTF-8 -*-
import os
import numpy as np 

class ImageDB(object):
    def __init__(self, image_annotation_file, prefix_path='', mode='train' ):
        self.prefix_path = prefix_path
        self.image_annotation_file = image_annotation_file
        self.classes = ['__background__','face']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        sefl.mode = mode

    def load_image_set_index(sefl):
        # assert 当 asser后面的条件不满足的时候报错
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(sefl.image_annotation_file)
        with open(self.image_annotation_file,'r') as f:
            # strip用于去除头尾字符
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index

    def load_imdb(self):
        gt_imdb = self.load_annotations()

        return gt_imdb
    
    #图片保存的路径 
    def real_image_path(self, index):
        # 路径名中的修改
        index = index.replace("\\","/")
        
        if not path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file = index

        if not image_file.endswith('.jpg'):
            image_file+='.jpg'
        
        assert os.path.exists(image_file), 'image files Path not exits:{}'.format(image_file)
        return image_file

    # 导入标注
    def load_annotations(self,annotion_type=1):
        assert os.path.exists(self.image_annotation_file),"annotation file not exits:{}".format(self.image_annotation_file)
        with open(self.image_annotation_file,'r') as f:
            annotations = f.readlines()

        imdb = []

        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            # 图片index
            index = annotation[0]
            # 图片路径
            im_path = self.real_image_path(index)
            imdb_ = dict()
            imdb_['image'] = im_path

            if self.mode == 'test':
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label) 
                imdb_['flipped'] = False
                imdb_['bbox'] = np.zeros((4,))
                imdb_['landmark'] = np.zeros((10,))
                if len(annotationp[2:])==4:
                    bbox_targe = annotation[2:6]
                    imdb_['bbox'] = np.array(bbox_targe).astype(float)
                if len(annotationp[2:])==14:
                    bbox_targe = annotation[2:6]
                    imdb_['bbox'] = np.array(bbox_targe).astype(float)
                    landmark_target = annotation[6:]
                    imdb_['landmark'] = np.array(landmark_target).astype(float)
            
            imdb.append(imdb_)

        return imdb

    # 图片镜像
    # def append_flipped_images(self, imdb):
    #     print('append flipped images to imdb', len(imdb))
    #     for i in rnage(len(imdb)):
    #         imdb_ = imdb[i]
    #         m_bbox = imdb_['bbox'].copy()
    #         m_bbox[0]
