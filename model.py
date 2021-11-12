import torch
import torch.nn as nn
import torch.nn.parallel
import constants as ct
import attention as at


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

class NetG(nn.Module):


    def __init__(self, isize, nc, nz, ndf, n_extra_layers=0):
        super(NetG, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.e1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            )
        self.e_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(t, ndf),
                            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, ndf),
                            nn.BatchNorm2d(ndf))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(t, ndf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.e2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e4 = nn.Sequential(
            nn.Conv2d(ndf*4, nz, 3, 1, 1, bias=False),
            )
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(nz, ndf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
            )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(ndf*4*2, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),
            )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(ndf*4, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            )
        self.d_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(ndf*2, ndf),
                            nn.Conv2d(ndf*2, ndf, 3, 1, 1, bias=False))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(ndf, ndf),
                            nn.BatchNorm2d(ndf))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(ndf, ndf),
                            nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(ndf*2, ct.GT_C, 4, 2, 1, bias=False),
#             nn.LeakyReLU(),
            nn.Sigmoid(),
#             nn.ReLU(),
            )
        #attention module

        self.at1 = at.csa_layer(1)
        self.at2 = at.csa_layer(1)
        self.at3 = at.csa_layer(1)
        self.at4 = at.csa_layer(1)
        
    def forward(self,x):
        
        e1 = self.e1(x)
        e_el = self.e_extra_layers(e1)
        e2 = self.e2(e_el)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d4 = self.d4(e4)
        
        d4 = self.at4(d4)
        
        c34 = torch.cat((e3,d4),1)
        d3 = self.d3(c34)
        d3 = self.at3(d3)
        
        c23 = torch.cat((e2,d3),1)
        d2 = self.d2(c23)
        d2 = self.at2(d2)
        
        cel2 = torch.cat((e_el,d2),1)
        d_el = self.d_extra_layers(cel2)
        e_el = self.at1(d_el)
        
        c11 = torch.cat((e1,d_el),1)
        d1 = self.d1(c11)
        
        return d1


class NetD(nn.Module):

    def __init__(self, isize, nc, nz, ndf, n_extra_layers=0):
        super(NetD, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.e1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            )
        self.e_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(t, ndf),
                            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, ndf),
                            nn.BatchNorm2d(ndf))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(t, ndf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.e2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.toplayer = nn.Sequential(
            nn.Conv2d(ndf*4, nz, 3, 1, 1, bias=False),
            nn.Sigmoid(),
            ) 
        self.avgpool = nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            )
    def forward(self,x):
        x = self.e1(x)
        x = self.e_extra_layers(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.toplayer(x)
        x = self.avgpool(x)
        x = x.view(-1,1).squeeze(1)
        return x
        
    
if __name__ == '__main__':
    netg = NetG(ct.ISIZE, ct.NC*2, ct.NZ, ct.NDF, ct.EXTRALAYERS)
    y = netg(torch.randn(2,6,256,256))    
    print(y.shape)
        
       