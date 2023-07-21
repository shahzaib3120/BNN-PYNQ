import torch
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d
from torch.nn import functional as F
from binarized_modules import *


class cnv(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, input_size=32, wb=1, ab=1, use_bias=False):
        super(cnv, self).__init__()
        out_dim = ((input_size-12)/4-4)
        self.features_out_dim = 256 * out_dim * out_dim
        # assert that output dimension is integer
        assert self.features_out_dim.is_integer(
        ), "Input dimension must be divisible by 4 and >= 32"
        self.features_out_dim = int(self.features_out_dim)
        self.features = nn.Sequential(
            BinarizeConv2d(wb, input_channels, 64, kernel_size=3,
                           stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),

            BinarizeConv2d(wb, 64, 64, kernel_size=3,
                           stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(wb, 64, 128, kernel_size=3,
                           stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),

            BinarizeConv2d(wb, 128, 128, kernel_size=3,
                           stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinarizeConv2d(wb, 128, 256, kernel_size=3,
                           stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),

            BinarizeConv2d(wb, 256, 256, kernel_size=3,
                           stride=1, padding=0, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(wb, self.features_out_dim, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),

            BinarizeLinear(wb, 512, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.Hardtanh(inplace=True),
            Quantizer(ab),

            BinarizeLinear(wb, 512, num_classes, bias=True),
            nn.BatchNorm1d(num_classes),

            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        self.features_out_dim = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, self.features_out_dim)
        x = self.classifier(x)
        return x

    def export(self):
        import numpy as np
        dic = {}
        i = 0

        # process conv and BN layers
        for k in range(len(self.features)):
            if hasattr(self.features[k], 'weight') and not hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_' +
                    str(i)] = self.features[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1. / \
                    np.sqrt(self.features[k].running_var.detach().numpy())
                i = i + 1

        # process linear and BN layers
        for k in range(len(self.classifier)):
            if hasattr(self.classifier[k], 'weight') and not hasattr(self.classifier[k], 'running_mean'):
                dic['arr_' +
                    str(i)] = np.transpose(self.classifier[k].weight.detach().numpy())
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].weight.detach().numpy()
                i = i + 1
                dic['arr_' +
                    str(i)] = self.classifier[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1. / \
                    np.sqrt(self.classifier[k].running_var.detach().numpy())
                i = i + 1

        save_file = 'results/cifar10-w{}a{}.npz'.format(args.wb, args.ab)
        np.savez(save_file, **dic)
        print("Model exported at: ", save_file)


def init_weights(m):
    if type(m) == BinarizeLinear or type(m) == BinarizeConv2d:
        torch.nn.init.uniform_(m.weight, -1, 1)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wb', type=float, default=1)
    parser.add_argument('--ab', type=float, default=1)
    args = parser.parse_args()

    model = cnv(wb=args.wb, ab=args.ab)
    model.export()
