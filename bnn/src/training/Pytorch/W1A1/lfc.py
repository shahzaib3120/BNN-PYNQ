
from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import  *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

class lfc(nn.Module):
    def __init__(self):
        super(lfc, self).__init__()

        self.classifier = nn.Sequential(
            BinarizeLinear(784, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(1024, 10, bias=True),
            nn.BatchNorm1d(10),
        )
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.classifier(x)
        return x

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        target=target.unsqueeze(1)
        if args.cuda:
            target_onehot = torch.cuda.FloatTensor(target.size(0), 10)
        else:
            target_onehot = torch.FloatTensor(target.size(0), 10)
        target_onehot.fill_(-1)
        target_onehot.scatter_(1, target, 1)
        target=target.squeeze()
        target_var = Variable(target_onehot)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_var)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    prev_acc = 0;
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        target=target.unsqueeze(1)
        if args.cuda:
            target_onehot = torch.cuda.FloatTensor(target.size(0), 10)
        else:
            target_onehot = torch.FloatTensor(target.size(0), 10)
        target_onehot.fill_(-1)
        target_onehot.scatter_(1, target, 1)
        target=target.squeeze()
        target_var = Variable(target_onehot)

        output = model(data)
        test_loss += criterion(output, target_var).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    new_acc = float(100 * correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset), new_acc))
    
    if new_acc > prev_acc:
        # save model
        torch.save(model, 'results/mnist-w1a1.pt')
        prev_acc = new_acc

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    if args.cuda:
            torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = lfc()
    if args.cuda:
            torch.cuda.set_device(0)
            model.cuda()

    criterion = HingeLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
            train(epoch)
            test()
