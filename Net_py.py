import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from custom_conv2d import MBMconv2d


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.MBMconv_1 = MBMconv2d(1, 32, kernel_size=3, stride=1, padding=1)
		self.batchnorm_1 = nn.BatchNorm2d(32)
		self.relu_1 = nn.ReLU(inplace=True)
		self.normalconv_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.batchnorm_2 = nn.BatchNorm2d(64)
		self.relu_2 = nn.ReLU(inplace=True)
		self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.normalconv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.batchnorm_3 = nn.BatchNorm2d(128)
		self.relu_3 = nn.ReLU(inplace=True)
		self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.linear_block = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(128*7*7, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )


	def forward(self, x):
		x = self.relu_1(self.batchnorm_1(self.MBMconv_1(x)))
		x = self.maxpool_1(self.relu_2(self.batchnorm_2(self.normalconv_1(x))))
		x = self.maxpool_2(self.relu_3(self.batchnorm_3(self.normalconv_2(x))))

		x = x.view(x.size(0), -1)

		x = self.linear_block(x)

		return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#         self.conv_block = nn.Sequential(
#             self.MBMconv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2) 
#         )
        
#         self.linear_block = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(128*7*7, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(64, 10)
#         )
        
#     def forward(self, x):
#         x = self.conv_block(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_block(x)
        
#         return x