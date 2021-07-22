import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class R_BI_C(nn.Module):
	def __init__(self, args, field=None):
		super(R_BI_C, self).__init__()
		self.args = args

		V = args.embed_num
		D = args.embed_dim
		C = args.class_num
		N = args.num_layers
		H = args.hidden_size
		Ci = 1
		Co = args.kernel_num
		Ks = args.kernel_sizes

		self.embed_A = nn.Embedding(V, D)
		self.embed_B = nn.Embedding(V, D)
		self.embed_B.weight.data.copy_(field.vocab.vectors)

		self.lstm = nn.LSTM(D, H, num_layers=N, \
							bidirectional = True,
							batch_first = True,
							dropout=args.LSTM_dropout)
		
		self.conv1_D = nn.Conv2d(Ci, Co, (1, 2*H))

		self.convK_1 = nn.ModuleList(
					[nn.Conv2d(Co, Co, (K, 1)) for K in Ks])

		self.conv3 = nn.Conv2d(Co, Co, (3, 1))

		self.conv4 = nn.Conv2d(Co, Co, (3, 1), padding=(1,0))

		self.CNN_dropout = nn.Dropout(args.CNN_dropout)
		self.fc1 = nn.Linear(len(Ks)*Co, C)

		

	def forward(self, x):

		x_A = self.embed_A(x)  # x [batch_size, sen_len, D]
		x_B = self.embed_B(x)
		x = torch.add(x_A, x_B)
		out, _ = self.lstm(x) # [batch_size, sen_len, H*2]
		x = out.unsqueeze(1)
		x = self.conv1_D(x)

		x = [F.relu(conv(x)) for conv in self.convK_1]
		x3 = [F.relu(self.conv3(i)) for i in x]
		x4 = [F.relu(self.conv4(i)) for i in x3]
		inception = []
		for i in range(len(x4)):
			res = torch.add(x3[i], x4[i])
			inception.append(res)

		x = [i.squeeze(3) for i in inception]
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x, 1)

		x = self.CNN_dropout(x)
		logit = self.fc1(x)
		return logit
