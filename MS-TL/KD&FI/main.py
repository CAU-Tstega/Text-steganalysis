import os
import sys
import argparse
import datetime
import torch
from transformers import BertModel, BertTokenizer
import Models 
import train 
from DataLoader import * 
import random
import numpy as np

parser = argparse.ArgumentParser(description='Models')

# learning
parser.add_argument('-batch-size', type=int, default=64, \
					help='batch size for training [default: 64]')
parser.add_argument('-lr', type=float, default=0.0001,\
					help='initial learning rate [default:5e-5]')
parser.add_argument('-epochs', type=int, default=30,\
					help='number of epochs for train [default:30]')
parser.add_argument('-log-interval', type=int, default=20, \
					help='how many steps to wait defore logging train status')
parser.add_argument('-test-interval', type=int, default=100, \
					help='how many steps to wait defore testing [default:100]')
parser.add_argument('-save-interval', type=int, default=500, \
					help='how many steps to wait defore saving [default:500]')
parser.add_argument('-early-stop', type=int, default=1000, \
					help='iteration numbers to stop without performace boost')
parser.add_argument('-save-best', type=bool, default=True,\
					help='whether to save when get best performance')
parser.add_argument('-save-dir', type=str, default='snapshot',
					help='where to save the snapshot')
parser.add_argument('-load-dir', type=str, default=None,
					help='where to loading the trained teacher model')

# data
parser.add_argument('-train-cover-dir', type=str, default='cover.txt',
					help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-train-stego-dir', type=str, default='1bpw.txt',
					help='the path of train stego data. [default:1bpw.txt]')
parser.add_argument('-test-cover-dir', type=str, default='cover.txt',
					help='the path of tset cover data. [default:cover.txt]')
parser.add_argument('-test-stego-dir', type=str, default='1bpw.txt',
					help='the path of test stego data. [default:1bpw.txt]')
parser.add_argument('-num-layers', type=int, default=1,
					help='the number of bilstm layers. [default:1]')

#device
parser.add_argument('-no-cuda', action='store_true', default=False, \
					help='disable the gpu [default:False]')
parser.add_argument('-device', type=str, default='cuda', \
					help='device to use for trianing [default:cuda]')
parser.add_argument('-idx-gpu', type=str, default='0',\
					help='the number of gpu for training [default:0]')

# option
parser.add_argument('-test', type=bool, default=False, \
					help='train or test [default:False]')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu


args.model = BertModel.from_pretrained('pretrained_BERT/base_uncased/')
args.tokenizer = BertTokenizer.from_pretrained('pretrained_BERT/base_uncased/')


# load data
print('\nLoading data...')
train_data, valid_data, test_data = build_dataset(args)
train_iter = build_iterator(train_data, args)
valid_iter = build_iterator(valid_data, args)
test_iter = build_iterator(test_data, args)

vocab = args.tokenizer.vocab	
print(vocab['get'])
sys.exit()
# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.embed_num =  len(args.tokenizer.vocab)
args.embed_dim = 300 
args.dropout = 0.5
args.hidden_size = 200

# Prepare seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1) 


if args.load_dir is not None:
	print("Knowledge Distilling...")
	teacher_model = Models.Bert_base(args)
	teacher_model.load_state_dict(torch.load(args.load_dir))
	# target_model = Models.Bilstm(args)
	target_model = Models.CNN(args)

	if args.cuda:
		torch.device(args.device)
		teacher_model = teacher_model.cuda()
		target_model = target_model.cuda()
	## Caculate the number of parameters of the loaded model
	teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
	target_total_params = sum(p.numel() for p in target_model.parameters())
	print('teacher: ', teacher_total_params)
	print('student: ', target_total_params)
	if not args.test:
		train.train(train_iter, valid_iter, args, target_model, teacher_model)
	
else:
	print("Fine-Turning Training...")
	pretrained_Bilstm_dir = 'snapshot/BERT-BiLSTM/movie_2bpw/best_steps_1300.pt'
	pretrained_CNN_dir = 'snapshot/BERT-CNN/movie_2bpw/best_steps_2400.pt'

	def transfer_state_dict(pretrained_dict, model_dict):
		state_dict = {}
		for k, v in pretrained_dict.items():
			if k in model_dict.keys():
				state_dict[k] = v
			else:
				print('Missing key(s) in state_dict: {}.'.format(k))
		return state_dict
	
	def transfer_model(args, model):
		pretrained_dict1 = torch.load(pretrained_CNN_dir)
		pretrained_dict2 = torch.load(pretrained_Bilstm_dir)
		model_dict = model.state_dict()
		pretrained_dict = {**pretrained_dict1, **pretrained_dict2}
		pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		return model

	target_model = Models.Bilstm_C(args)
	if args.cuda:
		target_model = target_model.cuda()


	if not args.test:
		pretrained_model = transfer_model(args, target_model)
		## Caculate the number of parameters of the loaded model
		target_total_params = sum(p.numel() for p in target_model.parameters())
		print(target_total_params)
		train.train(train_iter, valid_iter, args, pretrained_model)


if args.test:
	print('\n----------testing------------')
	print('Loading test model from {}...'.format(args.save_dir))
	models = []
	files = sorted(os.listdir(args.save_dir))
	for name in files:
		if name.endswith('.pt'):
			models.append(name)
	model_steps = sorted([int(m.split('_')[-1].split('.')[0]) for m in models])
	for step in model_steps[-3:]:
		best_model = 'best_steps_{}.pt'.format(step)
		m_path = os.path.join(args.save_dir, best_model)
		print('the {} model is loaded...'.format(m_path))
		target_model.load_state_dict(torch.load(m_path))
		train.data_eval(test_iter, target_model, args)

