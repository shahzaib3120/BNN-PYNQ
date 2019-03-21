# BNN.pytorch for w1a1 training

## For Traning cnv

```bash
	$ python cnv.py
```
To convert .pt to .npz run 

```bash
	$ python pt_to_npz_cnv.py
```
Copy the generated .npz

```bash
	$ cp results/cifar10-w1a1.npz ../../weights
```

Binarized Neural Network (BNN) for pytorch
This is the pytorch version for the BNN code, fro VGG and resnet models
Link to the paper: https://papers.nips.cc/paper/6573-binarized-neural-networks

The code is based on https://github.com/eladhoffer/convNet.pytorch
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10
