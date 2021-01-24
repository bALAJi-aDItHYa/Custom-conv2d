import torch
import numpy as np
import torchvision as tv
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold
from MBM import MBM_func, convolve

class MBM_conv2d(torch.autograd.Function):

	# @staticmethod
	# #inp and kernel must be a tensor with only one dimension and of same length
	
	# def convolve(inp, kernel):
		
	# 	result = 0.0
	# 	for i in range(len(inp)):
	# 		result += MBM_func(inp[i], kernel[i]) #Custom multiplication algorithm

	# 	return result

	@staticmethod
	#define the forward utility function - does the MBM convolution operation
	#ctx - context object used for storing necessary parameters in back_prop
	#kernel dimensions --> [out_channel, in_channel, kh, kw]

	def forward(ctx, in_feature, kernel, out_channel, bias=None):

		#Features to be later used in backward()
		ctx.save_for_backward(in_feature, kernel, bias)

		print(in_feature.size())
		print(kernel.size())

		batch_size = in_feature.size(0)
		in_channels = in_feature.size(1)
		orig_h, orig_w = in_feature.size(2), in_feature.size(3)

		#Kernel Dimenstions
		kh, kw = kernel.size(2), kernel.size(3)
		#Strides
		dh, dw = 1, 1

		#Padding --> o = [i+2p-k/s]+1 && o = i
		p = int((kh-1)/2)
		img = F.pad(input= in_feature, pad= (p, p, p, p), mode='constant', value= 0)

		#Image Dimenstions
		h, w = img.size(2), img.size(3)

		#Creating the patches - over which convolution is done
		patches = img.unfold(2, kh, dh).unfold(3, kw, dw).reshape(batch_size, -1, in_channels, kh, kw)
		#To parallelize the operation
		#[b,L,c,kh,kw] --> [b,L,c*kh*kw]
		patches = patches.reshape(patches.size(0), patches.size(1), -1) 

		#Reshaping the kernel for parallelization
		#[o,c,kh,kw] --> [o, c*kh*kw]
		k = kernel.reshape(out_channel, -1) 
		result = torch.zeros(batch_size, out_channel, orig_h, orig_w)

		patches, result = patches.type(torch.cuda.FloatTensor), result.type(torch.cuda.FloatTensor)

		#Convolution Operation
		#Actually it cross-correlation that is carried out!... 
		#x is a float val that is inserted in the appropriate position in output tensor --> result
		for b in range(batch_size):
			for o in range(out_channel):
				for L in range(patches.size(1)):
					x = convolve(patches[b][L], k[o])
					#print("this is L number - {}".format(L))
					#print("batch - {}".format(b))
					#print("channel - {}".format(o))
					#print("row pos - {}".format(L//orig_h))
					#print("col pos - {}".format(L%orig_w))
					result[b][o][L//orig_h][L%orig_w] = x

		#In case bias is also supposed to be added
		if bias is not None:
			result += bias.unsqueeze(0).expand_as(result)

		return result

	@staticmethod
	#Defining the gradient formula... done automatically
	# #arguments to backward() = #outputs from forward()
	# #outputs from backward() = #arguments to forward()
	
	def backward(ctx, grad_output):

		#Features from forward whose gradients are required
		# input --> in_feature, weight --> kernel, bias
		input, weight, bias = ctx.saved_tensors

		grad_input = grad_weight = grad_bias = None

		print("I'm here")
		if ctx.needs_input_grad[0]:
			grad_input = grad_output.mm(weight)
		if ctx.needs_input_grad[1]:
			grad_output = torch.transpose(grad_output,3,2)
			grad_weight = torch.matmul(input, grad_output)
			#grad_weight = grad_output.t().mm(input)
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0)

		return grad_input, grad_weight, None, None
        

class MBMconv2d(nn.Module):
	#Initialize the weight/ kernels
	#Call the custom functional API MBMconv2d

	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=None):
		super(MBMconv2d, self).__init__()
		
		#Initialize misc. variables
		self.in_channels = in_channels  
		self.out_channels = out_channels

		#Initialize weights/ kernels and make them parametrisable
		self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
		self.register_parameter('bias',None)

	def forward(self, x):
		# x = self.mbm_conv(x, self.kernel, self.out_channels, None)
		# return x
		return MBM_conv2d.apply(x, self.kernel, self.out_channels, None)