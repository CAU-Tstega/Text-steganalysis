import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer



class Bert_base(nn.Module):
	def __init__(self, args):
		super(Bert_base, self).__init__()
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


class Bilstm(nn.Module):
	def __init__(self, args):
		super(Bilstm, self).__init__()
		V = args.embed_num
		D = args.embed_dim
		H = args.hidden_size

		self.embed = nn.Embedding(V,D)
		self.lstm = nn.LSTM(D, H, num_layers=1,
							bidirectional = True,
							batch_first = True)
							# dropout = args.dropout)

		self.fc = nn.Linear(2*H, 2)

		# parameters initialization
		for key in self.state_dict():
			if key is "fc.weight":
				nn.init.normal_(self.state()[key])
			if key is "fc.bias":
				nn.init.constant_(self.state()[key], 0.1)

	def forward(self, x):
		x = self.embed(x[0])
		out, _ = self.lstm(x)
		logit = self.fc(out[:,-1,:])
		return logit



class CNN(nn.Module):
	def __init__(self, args):
		super(CNN, self).__init__()
		V = args.embed_num
		D = args.embed_dim
		
		self.embedding = nn.Embedding(V,D)
		self.conv = nn.Conv2d(1,100, (3,D))
		self.dropout = nn.Dropout(args.dropout)
		self.fc = nn.Linear(100, 2)

		# parameters initialization
		for key in self.state_dict():
			if key is "conv.weight":
				nn.init.xavier_normal_(self.state()[key])
			if key is "conv.bias":
				nn.init.constatn_(self.state()[key], 0.1)
			if key is "fc.weight":
				nn.init.normal_(self.state()[key])
			if key is "fc.bias":
				nn.init.constant_(self.state()[key], 0.1)

	def forward(self, x):
		x = self.embedding(x[0])
		x = x.unsqueeze(3).permute(0,3,1,2)
		x = self.conv(x)
		x = F.relu(x).squeeze(3)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		x = self.dropout(x)
		logit = self.fc(x)
		return logit
		

class Bilstm_C(nn.Module):
	def __init__(self, args):
		super(Bilstm_C, self).__init__()
		V = args.embed_num
		D = args.embed_dim
		H = args.hidden_size

		self.embed = nn.Embedding(V,D)
		self.embedding = nn.Embedding(V,D)
		self.lstm = nn.LSTM(D, H, num_layers=1,
							bidirectional = True,
							batch_first = True)
		self.conv = nn.Conv2d(1, 100, (3, D))
		self.dropout = nn.Dropout(args.dropout)
		self.fc1 = nn.Linear(500, 2)

		# parameters initialization
		for key in self.state_dict():
			if key is "fc1.weight":
				nn.init.normal_(self.state()[key])
			if key is "fc1.bias":
				nn.init.constant_(self.state()[key], 0.1)
	
	def forward(self, x):
		cnn_x = self.embedding(x[0])
		lstm_x = self.embed(x[0])
		lstm_out, _ = self.lstm(lstm_x)

		cnn_in = cnn_x.unsqueeze(3).permute(0,3,1,2)
		cnn_out = self.conv(cnn_in)
		cnn_out = F.relu(cnn_out).squeeze(3)
		cnn_out = F.max_pool1d(cnn_out, cnn_out.size(2)).squeeze(2)
		
		out = torch.cat([cnn_out, lstm_out[:,1,:]], 1)
		out = self.dropout(out)
		logit = self.fc1(out)
		return logit
