import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torch
from pylab import *
import constants as ct

TRANSFORM = True
VISUALIZE = 0

def gen_data(path,val_percent,test_percent):

    if 0:
        delList = os.listdir(ct.TXT_PATH )
        for f in delList:
            filePath = os.path.join( ct.TXT_PATH , f )
            if os.path.isfile(filePath):
                print(filePath)
                os.remove(filePath)
                print (filePath + " was removed!")
               
    dirs = os.listdir(path)
    c = 0
    for dir in dirs:
        c += 1
        print(c, dir,' is generating')
        files = os.listdir(os.path.join(path,dir))
        img1_path = os.path.join(path,dir,'t0.jpg')
        img2_path = os.path.join(path,dir,'t1.jpg')
        gt_path = os.path.join(path,dir,'label.jpg')        
        if not os.path.exists(ct.TXT_PATH):
            os.makedirs(ct.TXT_PATH)  
        chance = np.random.randint(100)
        if chance<val_percent:
            with open(os.path.join(ct.TXT_PATH,'validation.txt'),'a') as f:
                f.write(img1_path+','+img2_path+','+gt_path)
                f.write('\n')
        elif chance<val_percent+test_percent:
            with open(os.path.join(ct.TXT_PATH,'test.txt'),'a') as f:
                f.write(img1_path+','+img2_path+','+gt_path)
                f.write('\n')
        else:
            with open(os.path.join(ct.TXT_PATH,'train.txt'),'a') as f:
                f.write(img1_path+','+img2_path+','+gt_path)
                f.write('\n')
        if VISUALIZE:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            gt = Image.open(gt_path)
            gt1 = Image.open(gt1_path)
            gt2 = Image.open(gt2_path)
            plt.figure(figsize=(200,300))
            plt.subplot(1,5,1)
            plt.imshow(img1)
            plt.subplot(1,5,2)
            plt.imshow(img2)
            plt.subplot(1,5,3)
            plt.imshow(gt)
            plt.subplot(1,5,4)
            plt.imshow(gt1)
            plt.subplot(1,5,5)
            plt.imshow(gt2)
            plt.show()

class OSCD_TRAIN(Dataset):
    def __init__(self, dir_nm):
        super(OSCD_TRAIN, self).__init__()
        self.dir_nm = dir_nm
        with open(os.path.join(self.dir_nm),'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)

    def __getitem__(self, idx):
        x1 = Image.open(self.list[idx].split(',')[0])
        x2 = Image.open(self.list[idx].split(',')[1])
        gt = Image.open(self.list[idx].split(',')[2].strip())
        
        t = [            
            transforms.RandomRotation((360,360), resample=False, expand=False, center=None),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((180,180), resample=False, expand=False, center=None),
            transforms.Resize((ct.ISIZE,ct.ISIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5 ), (0.5, 0.5, 0.5)),
                   ]
        if TRANSFORM:
            k = np.random.randint(4)        
            x1 = t[k](x1);x2 = t[k](x2);gt = t[k](gt);
            x1 = t[4](x1);x2 = t[4](x2);gt = t[4](gt);
            x1 = t[5](x1);x2 = t[5](x2);gt = t[5](gt);
            x1 = t[6](x1);x2 = t[6](x2);

        return x1, x2, gt

    def __len__(self):
        return self.file_size

class OSCD_TEST(Dataset):
    def __init__(self, dir_nm):
        super(OSCD_TEST, self).__init__()
        self.dir_nm = dir_nm
        with open(os.path.join(self.dir_nm),'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)

    def __getitem__(self, idx):
        x1 = Image.open(self.list[idx].split(',')[0])
        x2 = Image.open(self.list[idx].split(',')[1])
        gt = Image.open(self.list[idx].split(',')[2].strip())
        
        t = [            
            transforms.Resize((ct.ISIZE,ct.ISIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5 )),
                   ]
        if TRANSFORM:       
            x1 = t[0](x1);x2 = t[0](x2);gt = t[0](gt);
            x1 = t[1](x1);x2 = t[1](x2);gt = t[1](gt);
            x1 = t[2](x1);x2 = t[2](x2);
        return x1, x2, gt

    def __len__(self):
        return self.file_size

if __name__=='__main__':
    
    if ct.DATASET == 'CDD':
        gen_data(os.path.join(ct.TRAIN_DATA, 'train'),0,0)
        gen_data(os.path.join(ct.TRAIN_DATA, 'val'),100,0)
        gen_data(os.path.join(ct.TRAIN_DATA, 'test'),0,100)
    elif ct.DATASET == 'WHU-CD':
        gen_data(ct.TRAIN_DATA,10,10)
    else:
        gen_data(os.path.join(ct.TRAIN_DATA, 'train'),0,0)
        gen_data(os.path.join(ct.TRAIN_DATA, 'val'),100,0)
        gen_data(os.path.join(ct.TRAIN_DATA, 'test'),0,100)
