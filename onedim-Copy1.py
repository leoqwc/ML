import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy

GPU = False
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor
    
    
    

def pad_circular1d(x, pad):
    x = torch.cat([x, x[0:pad]])
    x = torch.cat([x[-2 * pad:-pad], x])
    return x

class Pad1d(torch.nn.Module):
    def __init__(self, pad):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Pad1d, self).__init__()
        self.pad = pad
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        shape = [1,x.shape[1],x.shape[2]+2*self.pad]
        xx = Variable(torch.zeros(shape)).type(dtype)
        for i in range(x.shape[1]):
            xx[0,i] = pad_circular1d(x[0,i],self.pad)
        return xx # pad_circular1d(x, self.pad)

    
def conv(in_f, out_f, kernel_size, stride=1,bias=False,pad=True):
    '''
    Circular convolution
    '''
    to_pad = int((kernel_size - 1) / 2)
    if pad:
        padder = Pad1d(to_pad)
    else:
        padder = None

    convolver = nn.Conv1d(in_f, out_f, kernel_size, stride, padding=0, bias=bias)
    
    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers).type(dtype)



    
def print_filters(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            print(m.weight.data.cpu().numpy())  
            
            


def plot_gradients(out_grads):
    for i,g in enumerate(out_grads):
        plt.semilogy(g,label=str(i)) 
    plt.legend()
    plt.show()



                
class ChannelNormalization(torch.nn.Module):
    '''
    Normalization Layer (need to complete)
    '''

 
   

def decnet(
        num_output_channels=1, ## Channel number for the last layer
        num_channels_up=[1]*5, ## Channel number for the input and hidden layers, you should set it as: [1]*number_of_layers
        filter_size_up=2, ## Kernel Size
        act_fun=nn.ReLU(), 
        mode = "BN",
        res = False, ## Whether use resnet module
        ):
    '''
    Define the network structure. Output is the network
    '''
    
    num_channels_up = num_channels_up + [num_channels_up[-1]]
    n_scales = len(num_channels_up) 
    
    model = nn.Sequential()

    for i in range(len(num_channels_up)-1):
        if res:
            '''
            Please ignore the ResidualBlock, not used for HW3
            '''
            model.add(ResidualBlock( num_channels_up[i], num_channels_up[i+1],  filter_size_up, 1,mode))
        else:
            model.add(conv( num_channels_up[i], num_channels_up[i+1],  filter_size_up, 1))
            if act_fun != None:
                model.add(nn.ReLU())
            model.add(ChannelNormalization(num_channels_up[i],mode=mode))
 
            
    model.add(conv(num_channels_up[-1], num_output_channels, 1, bias=True,pad=False))

    
    return model



def fit(net,
        y,
        num_channels,
        net_input = None,
        num_iter = 5000,
        LR = 0.01,
       ):
'''
Write a function to fit the network and print the training loss vs iteration steps
'''




