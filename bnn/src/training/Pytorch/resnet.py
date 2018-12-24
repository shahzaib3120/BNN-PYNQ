from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize #,Ternarize,Ternarize2,Ternarize3,Ternarize4,HingeLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ResNet Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


class block(nn.Module):
    def __init__(self):
	super(block, self).__init__()

        self.conv2 = BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
	self.bn2 = nn.BatchNorm2d(64)
        self.ac2 = nn.Hardtanh(inplace=True)

        self.conv3 = BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
	self.bn3 = nn.BatchNorm2d(64)
        self.ac3 = nn.Hardtanh(inplace=True)

    def forward(self, x):
		
	residual = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.ac2(out)

        out = self.conv3(out)

        out += residual

        out = self.bn3(out)
        out = self.ac3(out)

        return out

class resnet(nn.Module):
    def __init__(self):
	super(resnet, self).__init__()
		
	# conv 1
	self.conv1 = BinarizeConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
	self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
	self.bn1 = nn.BatchNorm2d(64)
	self.ac1 = nn.Hardtanh(inplace = True)


	# block conv 2
	self.conv2 = BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
	self.bn2 = nn.BatchNorm2d(64)
        self.ac2 = nn.Hardtanh(inplace=True)

        # block sp conv 3
        self.conv3 = BinarizeConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
	self.bn3 = nn.BatchNorm2d(64)
        self.ac3 = nn.Hardtanh(inplace=True)

	self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

	self.fc1 = BinarizeLinear(7*7*64, 1024, bias=True)
        self.bn4 = nn.BatchNorm1d(1024)
        self.ac4 = nn.Hardtanh(inplace=True)

        self.drop = nn.Dropout(0.5)
        
        self.fc2 = BinarizeLinear(1024, 10, bias=True)
        self.bn5 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
    	# conv 1
    	x = self.conv1(x)
    	x = self.max1(x)
    	x = self.bn1(x)
    	x = self.ac1(x)

    	# residual block
    	residual = x

    	# conv 2
    	x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)

        # sp conv 3
        x = self.conv3(x)
        x += residual
        x = self.bn3(x)
        x = self.ac3(x)

    	x = self.max2(x)

    	x = x.view(-1, 7*7*64)

    	# fc 1
    	x = self.fc1(x)
    	x = self.bn4(x)
    	x = self.ac4(x)
    	
    	x = self.drop(x)

    	# fc 2
    	x = self.fc2(x)
    	x = self.bn5(x)

    	return self.logsoftmax(x)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

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
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    prev_acc = 0;
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    new_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset), new_acc))
    
    if new_acc > prev_acc:
    	#save model
    	torch.save(model, 'res/params.pt')
    	prev_acc = new_acc

if __name__ == '__main__':
	torch.manual_seed(args.seed)
	if args.cuda:
    		torch.cuda.manual_seed(args.seed)


	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    	batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    	batch_size=args.test_batch_size, shuffle=True, **kwargs)

	model = resnet()
	if args.cuda:
    		torch.cuda.set_device(0)
    		model.cuda()


	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	
	for epoch in range(1, args.epochs + 1):
    		train(epoch)
    		test()

