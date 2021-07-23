import os
import sys
import argparse
import datetime
import torch
from transformers import BertModel, BertTokenizer
import MyBert
import train
from DataLoader import *


parser = argparse.ArgumentParser(description='MyBert')

# learning
parser.add_argument('-batch-size', type=int, default=32, \
					help='batch size for training [default: 128]')
parser.add_argument('-lr', type=float, default=5e-5,\
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
parser.add_argument('-load_dir', type=str, default=None,
					help='where to loading the trained model')

# data
parser.add_argument('-train-cover-dir', type=str, default='cover.txt',
					help='the path of train cover data. [default:cover.txt]')
parser.add_argument('-train-stego-dir', type=str, default='1bpw.txt',
					help='the path of train stego data. [default:1bpw.txt]')
parser.add_argument('-test-cover-dir', type=str, default='cover.txt',
					help='the path of tset cover data. [default:cover.txt]')
parser.add_argument('-test-stego-dir', type=str, default='1bpw.txt',
					help='the path of test stego data. [default:1bpw.txt]')

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

	
# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda

# print('\nParameters: ')
# for attr, value in sorted(args.__dict__.items()):
# 	print('\t{}={}'.format(attr.upper(), value))


# model
model = MyBert.MyBert(args)

if args.load_dir is not None:	
	print('\nLoading model from {}...'.format(args.load_dir))
	model.load_state_dict(torch.load(args.load_dir))


if args.cuda:
	torch.device(args.device)
	model = model.cuda()

if not args.test:
# training phase
	train.train(train_iter, valid_iter, model, args)


# testing phase
else:
	print('\n----------testing------------')
	print('Loading test model from {}...'.format(args.save_dir))
	models = []
	files = sorted(os.listdir(args.save_dir))
	for name in files:
		if name.endswith('.pt'):
			models.append(name)
	model_steps = sorted([int(m.split('_')[-1].split('.')[0]) for m in models])
	ACC, R, P, F1 = 0, 0, 0, 0
	for step in model_steps[-3:]:
		best_model = 'best_steps_{}.pt'.format(step)
		m_path = os.path.join(args.save_dir, best_model)
		print('the {} model is loaded...'.format(m_path))
		model.load_state_dict(torch.load(m_path))
		acc, r, p, f = train.data_eval(test_iter, model, args)
		ACC += acc
		R += r
		P += p
		F1 += f

	with open(os.path.join(args.save_dir, 'result.txt'), 'a') as f:
		f.write('The average testing accuracy: {:.4f} \n'.format(ACC/3))
		f.write('The average testing recall: {:.4f} \n'.format(R/3))
		f.write('The average testing precious: {:.4f} \n'.format(P/3))
		f.write('The average testing F1_sorce: {:.4f} \n'.format(F1/3))

