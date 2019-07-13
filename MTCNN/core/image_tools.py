import torchvision.transforms as transforms
import torch
from torch.autograd.variable import Variable
import numpy as np

transforms = transforms.ToTensor()

def convert_image_to_tensor(image):
    return transforms(image)