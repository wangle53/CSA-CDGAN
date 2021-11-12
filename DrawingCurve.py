import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


def loss_curve():
    loss_path = './outputs/train_loss.txt'
    test_path = './outputs/testModel.txt'
    epoch = []
    loss1=[]
    loss2=[]
    loss3=[]
    loss=[]
    epoch1 = []
    kappa = []
    if os.path.exists(loss_path):
        with open(loss_path,'r') as f:
            lines = f.readlines()
            for i in range(0,len(lines)):
                epoch.append(float(lines[i].split(',')[0].split(' ')[1])+1)
                loss.append(float(lines[i].split(',')[1].split(' ')[3]))
                loss1.append(float(lines[i].split(',')[2].split(' ')[2]))
                loss2.append(float(lines[i].split(',')[3].split(' ')[2]))
                loss3.append(float(lines[i].split(',')[4].split(' ')[2]))
    if os.path.exists(test_path):
        with open(test_path,'r') as f:
            lines = f.readlines()
            for i in range(0,len(lines)):
                epoch1.append((float(lines[i].split(',')[0].split(' ')[1])+1))
                kappa.append((float(lines[i].split(',')[-1].split(' ')[4])))
                
    fig = plt.figure(figsize = (18,10))
    ax1 = fig.add_subplot(1, 1, 1)
#     p1 = pl.plot(epoch,loss1,'g--',label=u'Loss1')
#     pl.legend()
#     p2 = pl.plot(epoch, loss2,'r--', label = u'Loss2')
#     pl.legend()
#     p3 = pl.plot(epoch,loss3, 'b--', label = u'Loss3')
#     pl.legend()
    p4 = pl.plot(epoch,loss, 'k-', label = u'Loss')
    pl.legend()
    p5 = pl.plot(epoch1,kappa, 'm-o', label = u'kappa')
    pl.legend()
    pl.xlabel(u'Epoch')
    pl.ylabel(u'kappa')
    plt.title('Compare loss for different layers in training')
    if False:
        tx0 = 20
        tx1 = 80
        ty0 = 0.000
        ty1 = 0.5
        sx = [tx0,tx1]
        sy = [ty0,ty0]
        pl.plot(sx,sy,"purple")
        axins = inset_axes(ax1, width=5, height=5, loc='center right')
        axins.plot(epoch,loss , color='red', ls='-')
        axins.plot(epoch1,kappa , color='blue', ls='-')
        axins.axis([tx0,tx1,ty0,ty1])

    plt.savefig("loss.png")
    pl.show()
    
if __name__ == '__main__':
    loss_curve()