import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from spikingjelly.activation_based import functional, layer

from tqdm import tqdm
import math

parser = argparse.ArgumentParser(description='spikingjelly CIFAR10 Training')
parser.add_argument('--data_dir', metavar='DIR',
					help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
					metavar='N')
parser.add_argument('-T', '--timesteps', default=100, type=int,
					help='Simulation timesteps')
parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained parameters.')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')