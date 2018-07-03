import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class c_nn(nn.Module):

	def __init__(self, size, conv, tree, f_c, n_classes, n_children, feat_dropout):
		super(c_nn, self).__init__()
		self.tree_input_size = size
		self.n_children = n_children
		self.tree_depth = len(tree)
		self.dropout_layers = nn.ModuleList()
		self.batchnorm_layers = nn.ModuleList()

		# convolutional layers
		self.c_kernel_size = 3
		self.c_padding = 1
		self.p_kernel_size = 2

		self.conv_layers = nn.ModuleList()
		for i in range(len(conv)-1):
			self.conv_layers.append(nn.Conv2d(conv[i], conv[i+1], kernel_size=self.c_kernel_size, padding=self.c_padding))
			self.dropout_layers.append(nn.Dropout(feat_dropout))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))
			size = size + 2*self.c_padding - self.c_kernel_size + 1
			size = int(size/self.p_kernel_size)
		self.pool = nn.MaxPool2d(kernel_size=2)
		size = size*size*conv[-1]
		self.tree_input_size = size

		# tree layers
		current_leaves = 1
		tree = [self.tree_input_size] + tree

		self.tree_layer = nn.ModuleList()
		self.routers = nn.ModuleList()
		for i in range(len(tree)-1):
			for _ in range(current_leaves):
				self.routers.append(nn.Linear(tree[i], n_children))
			current_leaves = current_leaves*n_children
			for _ in range(current_leaves):
				self.tree_layer.append(nn.Linear(tree[i], tree[i+1]))
				self.dropout_layers.append(nn.Dropout(feat_dropout))
				self.batchnorm_layers.append(nn.BatchNorm1d(tree[i+1]))
				size = tree[i+1]
		size *= current_leaves

		# fully connected layers
		f_c = [size] + f_c

		self.fully_connected = nn.ModuleList()
		for i in range(len(f_c)-1):
			self.fully_connected.append(nn.Linear(f_c[i], f_c[i+1]))
			self.dropout_layers.append(nn.Dropout(feat_dropout))
			self.batchnorm_layers.append(nn.BatchNorm1d(f_c[i+1]))
			size = f_c[i+1]

		# output layer
		self.output_layer = nn.Linear(size, n_classes)

	def forward(self, x):

		do_bn_index = 0 # index of dropout and batchnorm layer arrays

		# convolutional layers
		for conv_layer in self.conv_layers:
			x = self.pool(F.relu(conv_layer(x)))
			x = self.dropout_layers[do_bn_index](x)
			x = self.batchnorm_layers[do_bn_index](x)
			do_bn_index += 1

		# tree layers
		x = [x.view(-1, self.tree_input_size)]
		current_index = 0
		for i in range(self.tree_depth):
			for j in range(len(x)):
				x_cummulative = []
				for _ in range(self.n_children):
					x_temp = F.relu(self.tree_layer[current_index](x[j]))
					x_temp = self.dropout_layers[do_bn_index](x_temp)
					x_temp = self.batchnorm_layers[do_bn_index](x_temp)
					current_index += 1
					do_bn_index += 1
					x_cummulative.append(x_temp)

				r = torch.t(F.softmax(self.routers[2**i + j - 1](x[j])))
				r_cummulative = []
				for _ in range(self.n_children):
					r_cummulative.append(torch.t(torch.stack([r[i]]*x_cummulative[i].size()[1])))
				for k in range(self.n_children):
					x_cummulative[k] *= r_cummulative[k]
				x[j] = x_cummulative

			x = [item for sublist in x for item in sublist]
		x = torch.cat(x,1)

		# fully connected layers
		for fc_layer in self.fully_connected:
			x = F.relu(fc_layer(x))
			x = self.dropout_layers[do_bn_index](x)
			x1 = self.batchnorm_layers[do_bn_index](x)
			do_bn_index += 1
		
		# output layer
		x = self.output_layer(x)

		return x