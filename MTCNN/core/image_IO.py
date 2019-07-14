from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import cv2

transform = transforms.ToTensor()
class FaceDataSet(Dataset):
    def __init__(self,labelPath ,im_size, img_transform=False,batch_size=128, shuffle=False):
        self.data_list = []
        self.img_transform = img_transform
        with open(labelPath,'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            data_dic = {}
            # 构造一个字典，每一个字典都有相应的数据读取字段，mtcnn包括图片，以及label，bbox，landmark，都进行初始化，避免不存在的label，
            # 例如landmar不存在，bbox不存在的情况，这个时候将他置为你np.zeros
            data_dic['image'] = line[0]
            data_dic['label'] = int(line[1])
            data_dic['bbox'] = np.zeros((4,))
            data_dic['landmark'] = np.zeros((10,))

            if len(line[2:]) ==4 :
                data_dic['bbox'] = np.array(line[2:6]).astype(float)
            elif len(line[2:]) ==14:
                data_dic['bbox'] = np.array(line[2:6]).astype(float)
                data_dic['landmark'] = np.array(line[7:17]).astype(float)

            self.data_list.append(data_dic)

    def __getitem__(self, index):
        data = self.data_list[index]

        if self.img_transform:
            img = cv2.imread(data['image'])
            # print(self.img_transform(img))
            data['image'] = np.array(transform(img)).astype(float)
            # print(data['image'].shape)
        return data

    def __len__(self):
        return len(self.data_list)

# if __name__=="__main__":
#     dset = FaceDataSet("../datasets/train/12/train_12.txt",
#         12,img_transform=True)
#     # for i in range(10):
#     print(dset[0])