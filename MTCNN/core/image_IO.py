from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os

class FaceDataSet(Dataset):
    def __init__(self, imgPath,labelPath ,im_size, img_transform=None ,batch_size=128, shuffle=False):
        self.image_list=[]
        self.gt_bbox_list=[]
        self.landmark_bbox_list=[]
        self.data_list = []
        with open(labelPath,'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if ".jpg" in line:
                self.image_list.append(os.path.join(imgPath,line))
                imagePath = os.path.join(imgPath,line)
            if len(line.split())==1:
                len_ = int(line.split[0])

            if len(line.split())==10:
                gt_bbox = list(map(int,line.split()[0:4]))
                # self.gt_bbox_list.append(list(map(int,line.split()[0:4])))
            i = 0 
            while i<len_:
                self.data_list.append(imagePath,)

        self.img_transform = img_transform

    def __getitem__(self, index):
        img_path = self.image_list[index]
        bbox_label = self.gt_bbox_list[index]
        img = img_path
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, bbox_label

    def __len__(self):
        return len(self.image_list)

if __name__=="__main__":
    dset = FaceDataSet("../../datasets",
        "../../datasets/wider_face_val_bbx_gt.txt",
        12)
    print(dset[0])
    print(dset[1])