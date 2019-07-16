import argparse
import sys
import os
sys.path.append(os.getcwd())
sys.path.append("..")
from core.image_IO import FaceDataSet
from train import train_pnet
import config as config

annotation_file = '../datasets/train/12/train_12.txt'
model_store_path = '../model_path'
end_epoch = 10
frequent = 20
lr = 0.01
batch_size = 512
use_cuda = True

def train_net(annotation_file, model_store_path, end_epoch=16, frequent=200 , lr=1, batch_size=256, use_cuda=True):
    print("create imageDB ........")
    imagedb = FaceDataSet(annotation_file,12,img_transform=True)
    print("begin Training ........")
    train_pnet(modelPath=model_store_path, end_epoch=end_epoch, imdb=imagedb, batch_size=batch_size, frequent=frequent, base_lr=lr)

# def parse_args():
#     parser = argparse('--')

if __name__=="__main__":

    train_net(annotation_file, model_store_path, end_epoch ,frequent, lr, batch_size, use_cuda=True)