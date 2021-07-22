import torch 
import torch.nn as nn
from torch.autograd import Variable


class TS_BiRNN(nn.Module):
	def __init__(self, args):
		super(TS_BiRNN, self).__init__()
		self.args = args

		V = args.embed_num
		D = args.embed_dim
		C = args.class_num
		N = args.num_layers
		H = args.hidden_size

		self.embed = nn.Embedding(V, D)
		self.lstm = nn.LSTM(D, H, num_layers=N, \
							bidirectional = True,
							batch_first = True,
							dropout=args.dropout)
		self.fc1 = nn.Linear(2*H, C)

		

	def forward(self, x):

		x = self.embed(x)  # x [batch_size, sen_len, D]
		out, _ = self.lstm(x)
		logit = self.fc1(out[:, -1, :])
		return logit
