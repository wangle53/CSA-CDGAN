import os
import time
import numpy as np
import torchvision.utils as vutils
from collections import OrderedDict
import visdom
import torch

def plot_current_errors(epoch, counter_ratio, errors,vis):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        
        
#         plot_data = None
#         plot_res = None
#         if not hasattr('plot_data') or plot_data is None:
        plot_data = {}
        plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([errors[k] for k in plot_data['legend']])
        
        vis.line(win='wire train loss', update='append',
            X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            Y=np.array(plot_data['Y']),
            opts={
                'title': 'WIRE' + ' loss over time',
                'legend': plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            })

        
def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)
            
def display_current_images(reals, fakes, vis):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = normalize(reals.cpu().numpy())
        fakes = normalize(fakes.cpu().numpy())
#         fixed = normalize(fixed.cpu().numpy())

        vis.images(reals, win=1, opts={'title': 'Reals'})
        vis.images(fakes, win=2, opts={'title': 'Fakes'})
#         vis.images(fixed, win=3, opts={'title': 'fixed'})
        
def get_errors(err_d, err_g):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err_d', err_d.item()),
            ('err_g', err_g.item())])

        return errors
    
def save_current_images(epoch, reals, fakes,save_dir,name):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        save_path = os.path.join(save_dir,name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        vutils.save_image(reals, '%s/reals.png' % save_path, normalize=True)
        vutils.save_image(fakes, '%s/fakes_%03d.png' % (save_path, epoch+1), normalize=True)

def save_weights(epoch,net,optimizer,save_path, model_name):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
        }
    torch.save(checkpoint,os.path.join(save_path,'current_%s.pth'%(model_name)))
    if epoch % 20 == 0:
        torch.save(checkpoint,os.path.join(save_path,'%d_%s.pth'%(epoch,model_name)))
  
  
def plot_performance( epoch, performance, vis):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        plot_res = []
        plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        plot_res['X'].append(epoch)
        plot_res['Y'].append([performance[k] for k in plot_res['legend']])
        vis.line(win='AUC', update='append',
            X=np.stack([np.array(plot_res['X'])] * len(plot_res['legend']), 1),
            Y=np.array(plot_res['Y']),
            opts={
                'title': 'Testing ' + 'Performance Metrics',
                'legend': plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
        )  
  
    