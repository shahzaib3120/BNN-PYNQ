import torch
from lenet import LeNet
import numpy as np

net = LeNet()
net = torch.load('results/lenet_parameters.pt', map_location = 'cpu')
net.eval()

# conv layer 
arr_0 = net.features[0].weight.detach().numpy()
arr_1 = net.features[0].bias.detach().numpy()

# BN layer
arr_2 = net.features[1].bias.detach().numpy()
arr_3 = net.features[1].weight.detach().numpy()
arr_4 = net.features[1].running_mean.detach().numpy()
arr_5 = net.features[1].running_var.detach().numpy()
arr_5 = 1./(np.sqrt(arr_5))

# hard tanh + maxpool

# conv layer
arr_6 = net.features[4].weight.detach().numpy()
arr_7 = net.features[4].bias.detach().numpy()

# BN layer
arr_8 = net.features[5].bias.detach().numpy()
arr_9 = net.features[5].weight.detach().numpy()
arr_10 = net.features[5].running_mean.detach().numpy()
arr_11 = net.features[5].running_var.detach().numpy()
arr_11 = 1./(np.sqrt(arr_11))

# hardtanh + maxpool

# linear layer
arr_12 = np.transpose(net.classifier[0].weight.detach().numpy())
arr_13 = net.classifier[0].bias.detach().numpy()

# BN layer
arr_14 = net.classifier[1].bias.detach().numpy()
arr_15 = net.classifier[1].weight.detach().numpy()
arr_16 = net.classifier[1].running_mean.detach().numpy()
arr_17 = net.classifier[1].running_var.detach().numpy()
arr_17 = 1./(np.sqrt(arr_17))

# hardtanh + dropout

# linear layer
arr_18 = np.transpose(net.classifier[4].weight.detach().numpy())
arr_19 = net.classifier[4].bias.detach().numpy()

# BN layer
arr_20 = net.classifier[5].bias.detach().numpy()
arr_21 = net.classifier[5].weight.detach().numpy()
arr_22 = net.classifier[5].running_mean.detach().numpy()
arr_23 = net.classifier[5].running_var.detach().numpy()
arr_23 = 1./(np.sqrt(arr_23))


np.savez('results/lenet_parameters.npz',arr_0,arr_1,arr_2,arr_3,arr_4,arr_5,arr_6,arr_7, arr_8,arr_9,arr_10,arr_11,	arr_12,	arr_13,	arr_14,	arr_15,	arr_16,	arr_17, arr_18,	arr_19,	arr_20,	arr_21,	arr_22,	arr_23)