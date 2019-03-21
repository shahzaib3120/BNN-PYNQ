import torch
from lfc import lfc
import numpy as np


net = lfc()
net = torch.load('results/lfc/params.pt', map_location = 'cpu')
net.eval()

# linear layer
arr_0 = np.transpose(net.classifier[0].weight.detach().numpy())
arr_1 = net.classifier[0].bias.detach().numpy()

# BN layer 
arr_2 = net.classifier[1].bias.detach().numpy()
arr_3 = net.classifier[1].weight.detach().numpy()
arr_4 = net.classifier[1].running_mean.detach().numpy()
arr_5 = net.classifier[1].running_var.detach().numpy()
arr_5 = 1./(np.sqrt(arr_5))

# hardtanh

# linear layer
arr_6 = np.transpose(net.classifier[3].weight.detach().numpy())
arr_7 = net.classifier[3].bias.detach().numpy()

# BN layer
arr_8 = net.classifier[4].bias.detach().numpy()
arr_9 = net.classifier[4].weight.detach().numpy()
arr_10 = net.classifier[4].running_mean.detach().numpy()
arr_11 = net.classifier[4].running_var.detach().numpy()
arr_11 = 1./(np.sqrt(arr_11))

# hardtanh

# linear layer 
arr_12 = np.transpose(net.classifier[6].weight.detach().numpy())
arr_13 = net.classifier[6].bias.detach().numpy()

# BN layer
arr_14 = net.classifier[7].bias.detach().numpy()
arr_15 = net.classifier[7].weight.detach().numpy()
arr_16 = net.classifier[7].running_mean.detach().numpy()
arr_17 = net.classifier[7].running_var.detach().numpy()
arr_17 = 1./(np.sqrt(arr_17))

# hardtanh

# linear layer
arr_18 = np.transpose(net.classifier[9].weight.detach().numpy())
arr_19 = net.classifier[9].bias.detach().numpy()

# BN layer
arr_20 = net.classifier[10].bias.detach().numpy()
arr_21 = net.classifier[10].weight.detach().numpy()
arr_22 = net.classifier[10].running_mean.detach().numpy()
arr_23 = net.classifier[10].running_var.detach().numpy()
arr_23 = 1./(np.sqrt(arr_23))


np.savez('results/lfc/lfc_parameters.npz',arr_0,arr_1,arr_2,arr_3,arr_4,arr_5,arr_6,arr_7, arr_8,arr_9,arr_10,arr_11, arr_12, arr_13,	arr_14,	arr_15,	arr_16,	arr_17, arr_18,	arr_19,	arr_20,	arr_21,	arr_22,	arr_23)