import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer



class MyBert(nn.Module):
	def __init__(self, args):
		super(MyBert, self).__init__()
		self.args = args

		self.bert = args.model 
		for param in self.bert.parameters():
			param.requires_grad = True

		self.fc1 = nn.Linear(768, 2)
	

	def forward(self, x):
		context = x[0]
		mask = x[2]
		_, output = self.bert(context, attention_mask=mask)
		logit = self.fc1(output)
		return logit
