import torch 
import sys
import torch.nn as nn
from torch.autograd import Variable


class BiLSTM_Dense(nn.Module):
	def __init__(self, args):
		super(BiLSTM_Dense, self).__init__()
		self.args = args

		V = args.embed_num
		D = args.embed_dim
		C = args.class_num
		N = args.num_layers
		H = args.hidden_size

		self.embed = nn.Embedding(V, D)
		self.lstm1 = nn.LSTM(D, H, num_layers=N, \
							bidirectional = True,
							batch_first = True)

		self.lstm2 = nn.LSTM(2*H, H, num_layers=N, \
							bidirectional = True,
							batch_first = True)

		self.lstm3 = nn.LSTM(4*H, H, num_layers=N, \
							bidirectional = True,
							batch_first = True)

		self.fc1 = nn.Linear(2*H, C)

		

	def forward(self, x):

		x = self.embed(x)  # x [batch_size, sen_len, D]
		out1, _ = self.lstm1(x)
		out2, _ = self.lstm2(out1)
		out3, _ = self.lstm3(torch.cat([out1, out2], 2))
		out = torch.add(torch.add(out1, out2), out3)
		logit = self.fc1(out[:, -1, :])
		return logit
