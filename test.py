from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import cv2

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from torch.nn import functional as F
from model import NetG
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import constants as ct
from make_dataset import OSCD_TEST
from torch.utils.data import Dataset, DataLoader
import evaluate as eva

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_network():
    threshold = ct.THRESHOLD
    test_dir = ct.TEST_TXT
    path = os.path.join(ct.BEST_WEIGHT_SAVE_DIR,'netG.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_dict = torch.load(path,map_location=torch.device(device))['model_state_dict']
    test_data = OSCD_TEST(test_dir)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    net = NetG(ct.ISIZE, ct.NC*2, ct.NZ, ct.NDF, ct.EXTRALAYERS).to(device)
#     net = nn.DataParallel(net) 
    net.load_state_dict(pretrained_dict,False)
    torch.no_grad()
    net.eval()
    i = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i, data in enumerate(test_dataloader):
        INPUT_SIZE = [ct.ISIZE,ct.ISIZE]
        x1, x2, gt = data
        x1 = x1.to(device, dtype=torch.float)
        x2 = x2.to(device, dtype=torch.float)
        gt = gt.to(device, dtype=torch.float)
        gt = gt[:,0,:,:].unsqueeze(1)

        x = torch.cat((x1,x2),1)
        fake = net(x)

        save_path = os.path.join(ct.IM_SAVE_DIR,'test_output_images')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
         
        if ct.SAVE_TEST_IAMGES:
            vutils.save_image(x1.data, os.path.join(save_path,'%d_x1.png'%i), normalize=True)
            vutils.save_image(x2.data, os.path.join(save_path,'%d_x2.png'%i), normalize=True)
            vutils.save_image(fake.data, os.path.join(save_path,'%d_gt_fake.png'%i) , normalize=True)
            vutils.save_image(gt, os.path.join(save_path,'%d_gt.png'%i), normalize=True)

        tp, fp, tn, fn = eva.f1(fake, gt)    
        TP += tp
        FN += fn
        TN += tn
        FP += fp
        i += 1
        print('testing {}th images'.format(i))
    iou = TP/(FN+TP+FP+1e-8)
    precision = TP/(TP+FP+1e-8)
    oa = (TP+TN)/(TP+FN+TN+FP+1e-8)
    recall = TP/(TP+FN+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    P = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/((TP+TN+FP+FN)**2+1e-8)
    Kappa = (oa-P)/(1-P+1e-8)
    results = {'iou':iou,'precision':precision,'oa':oa,'recall':recall,'f1':f1,'kappa':Kappa}

    
    with open(os.path.join(ct.OUTPUTS_DIR, 'test_score.txt'), 'a') as f:
        f.write('-----test results on the best model {}-----'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('\n')
        for key, value in results.items():
            print(key, value)
            f.write('{}: {}'.format(key, value))
            f.write('\n') 

if __name__ =='__main__':
    test_network()    