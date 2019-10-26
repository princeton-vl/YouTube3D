import torch
from torch import nn
from torch.autograd import Variable

class inception(nn.Module):
	def __init__(self, input_size, config):
		self.config = config
		super(inception,self).__init__()
		self.convs = nn.ModuleList()

		# Base 1*1 conv layer
		self.convs.append(nn.Sequential(
			nn.Conv2d(input_size, config[0][0],1),
			nn.BatchNorm2d(config[0][0],affine=False),
			nn.ReLU(True),
		))

		# Additional layers
		for i in range(1, len(config)):
			filt = config[i][0]
			pad = int((filt-1)/2)
			out_a = config[i][1]
			out_b = config[i][2]
			conv = nn.Sequential(
				nn.Conv2d(input_size, out_a,1),
				nn.BatchNorm2d(out_a,affine=False),
				nn.ReLU(True),
				nn.Conv2d(out_a, out_b, filt,padding=pad),
				nn.BatchNorm2d(out_b,affine=False),
				nn.ReLU(True)
				)
			self.convs.append(conv)

	def __repr__(self):
		return "inception"+str(self.config)

	def forward(self, x):
		ret = []
		for conv in (self.convs):
			ret.append(conv(x))
		
		return torch.cat(ret,dim=1)

