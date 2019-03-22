import torch
from cnv import cnv
import numpy as np


net = cnv()
net = torch.load('results/cifar10-w1a1.pt', map_location = 'cpu')
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

# hard tanh

# conv layer
arr_6 = net.features[3].weight.detach().numpy()
arr_7 = net.features[3].bias.detach().numpy()

# BN layer
arr_8 = net.features[4].bias.detach().numpy()
arr_9 = net.features[4].weight.detach().numpy()
arr_10 = net.features[4].running_mean.detach().numpy()
arr_11 = net.features[4].running_var.detach().numpy()
arr_11 = 1./(np.sqrt(arr_11))

# hard tanh + maxpool

# conv layer
arr_12 = net.features[7].weight.detach().numpy()
arr_13 = net.features[7].bias.detach().numpy()

# BN  layer
arr_14 = net.features[8].bias.detach().numpy()
arr_15 = net.features[8].weight.detach().numpy()
arr_16 = net.features[8].running_mean.detach().numpy()
arr_17 = net.features[8].running_var.detach().numpy()
arr_17 = 1./(np.sqrt(arr_17))

# hard tanh

# conv layer
arr_18 = net.features[10].weight.detach().numpy()
arr_19 = net.features[10].bias.detach().numpy()

# BN layer
arr_20 = net.features[11].bias.detach().numpy()
arr_21 = net.features[11].weight.detach().numpy()
arr_22 = net.features[11].running_mean.detach().numpy()
arr_23 = net.features[11].running_var.detach().numpy()
arr_23 = 1./(np.sqrt(arr_23))

# hard tanh + maxpool

# conv layer
arr_24 = net.features[14].weight.detach().numpy()
arr_25 = net.features[14].bias.detach().numpy()

# BN layer
arr_26 = net.features[15].bias.detach().numpy()
arr_27 = net.features[15].weight.detach().numpy()
arr_28 = net.features[15].running_mean.detach().numpy()
arr_29 = net.features[15].running_var.detach().numpy()
arr_29 = 1./(np.sqrt(arr_29))

# hard tanh

# conv layer
arr_30 = net.features[17].weight.detach().numpy()
arr_31 = net.features[17].bias.detach().numpy()

# BN layer
arr_32 = net.features[18].bias.detach().numpy()
arr_33 = net.features[18].weight.detach().numpy()
arr_34 = net.features[18].running_mean.detach().numpy()
arr_35 = net.features[18].running_var.detach().numpy()
arr_35 = 1./(np.sqrt(arr_35))

# hard tanh 

# linear layer
arr_36 = np.transpose(net.classifier[0].weight.detach().numpy())
arr_37 = net.classifier[0].bias.detach().numpy()

# BN layer
arr_38 = net.classifier[1].bias.detach().numpy()
arr_39 = net.classifier[1].weight.detach().numpy()
arr_40 = net.classifier[1].running_mean.detach().numpy()
arr_41 = net.classifier[1].running_var.detach().numpy()
arr_41 = 1./(np.sqrt(arr_41))

# hardtanh

# linear layer
arr_42 = np.transpose(net.classifier[3].weight.detach().numpy())
arr_43 = net.classifier[3].bias.detach().numpy()

# BN layer
arr_44 = net.classifier[4].bias.detach().numpy()
arr_45 = net.classifier[4].weight.detach().numpy()
arr_46 = net.classifier[4].running_mean.detach().numpy()
arr_47 = net.classifier[4].running_var.detach().numpy()
arr_47 = 1./(np.sqrt(arr_47))

# hardtanh

# linear layer
arr_48 = np.transpose(net.classifier[6].weight.detach().numpy())
arr_49 = net.classifier[6].bias.detach().numpy()

# BN layer
arr_50 = net.classifier[7].bias.detach().numpy()
arr_51 = net.classifier[7].weight.detach().numpy()
arr_52 = net.classifier[7].running_mean.detach().numpy()
arr_53 = net.classifier[7].running_var.detach().numpy()
arr_53 = 1./(np.sqrt(arr_53))

np.savez('results/cifar10-w1a1.npz',arr_0,arr_1,arr_2,arr_3,arr_4,arr_5,arr_6,arr_7, arr_8,arr_9,arr_10,arr_11,	arr_12,	arr_13,	arr_14,	arr_15,	arr_16,	arr_17, arr_18,	arr_19,	arr_20,	arr_21,	arr_22,	arr_23,	arr_24,	arr_25,	arr_26,	arr_27,	arr_28,	arr_29,	arr_30,	arr_31,	arr_32,	arr_33,	arr_34,	arr_35,	arr_36,	arr_37,	arr_38,	arr_39,	arr_40,	arr_41,	arr_42,	arr_43,	arr_44,	arr_45,	arr_46,	arr_47,	arr_48,	arr_49,	arr_50,	arr_51,	arr_52,	arr_53)


