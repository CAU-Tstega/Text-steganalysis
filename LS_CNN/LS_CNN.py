import torch 
import numpy as np
import gensim
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class LS_CNN(nn.Module):
	def __init__(self, args, text_field=None):
		super(LS_CNN, self).__init__()
		self.args = args

		V = args.embed_num
		D = args.embed_dim
		C = args.class_num
		Ci = 1
		Co = args.kernel_num
		Ks = args.kernel_sizes

		self.embed_A = nn.Embedding(V, D)
		#self.embed_B = nn.Embedding(V, D)
		#self.embed_B.weight.data.copy_(text_field.vocab.vectors)
		#self.embed_B.weight.requires_grad = False

		self.conv_embed = nn.Conv2d(2, 1, (1, 1))
		self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

		self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(len(Ks)*Co, C)
	

	def forward(self, x):
		# x_A = self.embed_A(x) # [batch_size, sen_length, D]
		# x_B = self.embed_B(x)
		# #张量连接
		# x = torch.cat([x_A.unsqueeze(3), x_B.unsqueeze(3)], 3)
		# #将tensor的维度换位。
		# x = x.permute(0,3,1,2)
		# x = self.conv_embed(x)
		x=self.embed_A(x)

		if self.args.static:
			x = Varaible(x)

		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
		
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x, 1)

		x = self.dropout(x)
		logit = self.fc1(x)
		return logit
