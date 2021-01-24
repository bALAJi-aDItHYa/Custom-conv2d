import torch
import torchvision as tv
from torch import nn
import torch.nn.functional as F

def convolve(inp, kernel):
	
	result = 0.0
	for i in range(len(inp)):
		result += MBM_func(inp[i], kernel[i]) #Custom multiplication algorithm

	return result

def MBM_func(a, b):
	return a*b;