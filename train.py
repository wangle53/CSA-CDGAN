#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from make_dataset import gen_data
from torch.nn import functional as F
from make_dataset import gen_data,OSCD_TRAIN,OSCD_TEST
import constants as ct
import evaluate as eva
from loss import l1_loss,l2_loss,cos_loss,DiceLoss
from model import NetG, NetD, weights_init
import utils 
import visdom
from torch.autograd import Variable
from tqdm import tqdm
from collections import OrderedDict
import shutil


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
vis = visdom.Visdom(server="http://localhost", port=8097)

def train_network():
    
    init_epoch = 0
    best_f1 = 0
    total_steps = 0
    train_dir = ct.TRAIN_TXT
    val_dir = ct.VAL_TXT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    train_data = OSCD_TRAIN(train_dir)
    train_dataloader = DataLoader(train_data, batch_size=ct.BATCH_SIZE, shuffle=True)
    val_data = OSCD_TEST(val_dir)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    netg = NetG(ct.ISIZE, ct.NC*2, ct.NZ, ct.NDF, ct.EXTRALAYERS).to(device=device)
    netd = NetD(ct.ISIZE, ct.GT_C, 1, ct.NGF, ct.EXTRALAYERS).to(device=device)
    netg.apply(weights_init)
    netd.apply(weights_init)
    if ct.RESUME:
        assert os.path.exists(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_netG.pth')) \
                and os.path.exists(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_netG.pth')), \
                'There is not found any saved weights'
        print("\nLoading pre-trained networks.")
        init_epoch = torch.load(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_netG.pth'))['epoch']
        netg.load_state_dict(torch.load(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_netG.pth'))['model_state_dict'])
        netd.load_state_dict(torch.load(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_netD.pth'))['model_state_dict'])
        with open(os.path.join(ct.OUTPUTS_DIR, 'f1_score.txt')) as f:
            lines = f.readlines()
            best_f1 = float(lines[-2].strip().split(':')[-1])
        print("\tDone.\n")
        
    l_adv = l2_loss
    l_con = nn.L1Loss()
    l_enc = l2_loss
    l_bce = nn.BCELoss()
    l_cos = cos_loss
    dice = DiceLoss()
    optimizer_d = optim.Adam(netd.parameters(), lr=ct.LR, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(netg.parameters(), lr=ct.LR, betas=(0.5, 0.999))
    
    start_time = time.time()
    for epoch in range(init_epoch+1, ct.EPOCH):
        loss_g = []
        loss_d = []
        netg.train()
        netd.train()
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):
            INPUT_SIZE = [ct.ISIZE,ct.ISIZE] 
            x1, x2, gt = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            gt = gt.to(device, dtype=torch.float)
            gt = gt[:,0,:,:].unsqueeze(1)
            x = torch.cat((x1,x2),1)
             
            epoch_iter += ct.BATCH_SIZE
            total_steps += ct.BATCH_SIZE
            real_label = torch.ones (size=(x1.shape[0],), dtype=torch.float32, device=device)
            fake_label = torch.zeros(size=(x1.shape[0],), dtype=torch.float32, device=device)
             
            #forward

            fake = netg(x)
            pred_real = netd(gt)
            pred_fake = netd(fake).detach()
            err_d_fake = l_bce(pred_fake, fake_label)
            err_g = l_con(fake, gt)
            err_g_total = ct.G_WEIGHT*err_g + ct.D_WEIGHT*err_d_fake
            
            pred_fake_ = netd(fake.detach())
            err_d_real = l_bce(pred_real, real_label)
            err_d_fake_ = l_bce(pred_fake_, fake_label)
            err_d_total = (err_d_real + err_d_fake_) * 0.5
            
            #backward
            optimizer_g.zero_grad()
            err_g_total.backward(retain_graph = True)
            optimizer_g.step()
            optimizer_d.zero_grad()
            err_d_total.backward()
            optimizer_d.step()
             
            errors = utils.get_errors(err_d_total, err_g_total)            
            loss_g.append(err_g_total.item())
            loss_d.append(err_d_total.item())
             
            counter_ratio = float(epoch_iter) / len(train_dataloader.dataset)
            if(i%ct.DISPOLAY_STEP==0 and i>0):
                print('epoch:',epoch,'iteration:',i,' G|D loss is {}|{}'.format(np.mean(loss_g[-51:]),np.mean(loss_d[-51:])))
                if ct.DISPLAY:
                    utils.plot_current_errors(epoch, counter_ratio, errors,vis)
                    utils.display_current_images(gt.data, fake.data, vis)
        utils.save_current_images(epoch, gt.data, fake.data, ct.IM_SAVE_DIR, 'training_output_images')
         
        with open(os.path.join(ct.OUTPUTS_DIR,'train_loss.txt'),'a') as f:
            f.write('after %s epoch, loss is %g,loss1 is %g,loss2 is %g,loss3 is %g'%(epoch,np.mean(loss_g),np.mean(loss_d),np.mean(loss_g),np.mean(loss_d)))
            f.write('\n')
        if not os.path.exists(ct.WEIGHTS_SAVE_DIR):
            os.makedirs(ct.WEIGHTS_SAVE_DIR)
        utils.save_weights(epoch,netg,optimizer_g,ct.WEIGHTS_SAVE_DIR, 'netG')
        utils.save_weights(epoch,netd,optimizer_d,ct.WEIGHTS_SAVE_DIR, 'netD')
        duration = time.time()-start_time
        print('training duration is %g'%duration)



        #val phase
        print('Validating.................')
        pretrained_dict = torch.load(os.path.join(ct.WEIGHTS_SAVE_DIR,'current_netG.pth'))['model_state_dict']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = NetG(ct.ISIZE, ct.NC*2, ct.NZ, ct.NDF, ct.EXTRALAYERS).to(device=device)
        net.load_state_dict(pretrained_dict,False)
        with net.eval() and torch.no_grad(): 
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            for k, data in enumerate(val_dataloader):
                x1, x2, label = data
                x1 = x1.to(device, dtype=torch.float)
                x2 = x2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                label = label[:,0,:,:].unsqueeze(1)
                x = torch.cat((x1,x2),1)
                time_i = time.time()
                v_fake = net(x)
                
                tp, fp, tn, fn = eva.f1(v_fake, label)    
                TP += tp
                FN += fn
                TN += tn
                FP += fp
            
            precision = TP/(TP+FP+1e-8)
            oa = (TP+TN)/(TP+FN+TN+FP+1e-8)
            recall = TP/(TP+FN+1e-8)
            f1 = 2*precision*recall/(precision+recall+1e-8)
            if not os.path.exists(ct.BEST_WEIGHT_SAVE_DIR):
                os.makedirs(ct.BEST_WEIGHT_SAVE_DIR)
            if f1 > best_f1: 
                best_f1 = f1
                shutil.copy(os.path.join(ct.WEIGHTS_SAVE_DIR,'current_netG.pth'),os.path.join(ct.BEST_WEIGHT_SAVE_DIR,'netG.pth'))           
            print('current F1: {}'.format(f1))
            print('best f1: {}'.format(best_f1))
            with open(os.path.join(ct.OUTPUTS_DIR,'f1_score.txt'),'a') as f:
                f.write('current epoch:{},current f1:{},best f1:{}'.format(epoch,f1,best_f1))
                f.write('\n')  
                 
if __name__ == '__main__':
    train_network()
    
    