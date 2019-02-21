import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed
import pdb

class ResBlock(nn.Module):
	def __init__(self, channel_input ,channel_num, channel_output):
		super(ResBlock, self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(channel_input, channel_num, kernel_size=1, stride=1,
					  padding=0, bias=False),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(True),
			nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1,
					  padding=1, bias=False),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(True),
			nn.Conv2d(channel_num, channel_output, kernel_size=3, stride=1,
					  padding=1, bias=False),
			nn.BatchNorm2d(channel_output))
		self.relu = nn.ReLU(inplace=True)
		if channel_input == channel_output:
			self.downsample = None
		else:
			self.downsample = nn.Sequential(
				nn.Conv2d(channel_input, channel_output, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(channel_output))

	def forward(self, x):
		residual = x
		if self.downsample is not None:
			residual = self.downsample(x)

		out = self.block(x)
		out += residual
		out = self.relu(out)
		return out


class generator(nn.Module):
	def __init__(self):
		super(generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		# self.embed_dim = 1024
		self.embed_dim = 4800
		self.projected_embed_dim = 2048 * 2
		self.latent_dim = self.noise_dim + self.embed_dim
		self.ngf = 64

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.latent_dim, out_features=self.projected_embed_dim),
			nn.BatchNorm1d(num_features=self.projected_embed_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)

		self.residual1 = ResBlock(self.ngf * 4, self.ngf * 2 , self.ngf * 8)
		self.block1 = nn.Sequential(
			nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True),
			nn.Conv2d(self.ngf * 4, self.ngf * 4, kernel_size=3, stride=1,padding=1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True)
		)

		self.residual2 = ResBlock(self.ngf * 4, self.ngf , self.ngf * 4)
		self.block2 = nn.Sequential(
			nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True),
			nn.Conv2d(self.ngf * 4, self.ngf * 2, kernel_size=3, stride=1, padding=1,bias=False),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
			nn.ConvTranspose2d(self.ngf * 2, self.ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
			nn.Conv2d(self.ngf * 2, self.ngf, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d(self.ngf, self.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			nn.Conv2d(self.ngf, self.num_channels, kernel_size=3, stride=1,padding=1, bias=False),
			nn.Tanh()
		)


	def forward(self, embed_vector, z):

		z = z.squeeze()
		latent_vector = torch.cat([embed_vector, z], 1)
		latent_vector = self.projection(latent_vector)
		latent_vector = torch.reshape(latent_vector,(-1,256,4,4))

		output = self.residual1(latent_vector)
		output = self.block1(output)
		output = self.residual2(output)
		output = self.block2(output)

		return output

class discriminator(nn.Module):
	def __init__(self):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.embed_dim = 4800
		self.projected_embed_dim = 128
		self.ndf = 64
		self.B_dim = 128
		self.C_dim = 16

		self.netD_1 = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

		self.netD_2 = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)	

	def forward(self, inp, embed):
		x_intermediate = self.netD_1(inp)
		x = self.projector(x_intermediate, embed)
		x = self.netD_2(x)

		return x.view(-1, 1).squeeze(1) , x_intermediate