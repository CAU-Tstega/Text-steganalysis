import torch
import random
from tqdm import tqdm
#from transformers import BertModel, BertTokenizer

PAD, CLS = '[PAD]', '[CLS]'

def build_dataset(args):
	
	def load_dataset(paths, pad_size=32):
		'''
		paths: ['cover.txt', 'stego.txt']
		'''
		contents = []
		for path in paths:
			with open(path, 'r', errors='ignore') as f:
				for line in tqdm(f):
					lin = line.strip()
					if 'cover' in path:
						label = 0
					else:
						label = 1
					token = args.tokenizer.tokenize(lin)
					token = [CLS] + token
					seq_len = len(token)
					mask = []
					token_ids = args.tokenizer.convert_tokens_to_ids(token)

					if pad_size:
						if len(token) < pad_size:
							mask = [1] * len(token_ids) + \
								   [0] *(pad_size - len(token))
							token_ids += ([0] * (pad_size - len(token)))
						else:
							mask = [1] * pad_size
							token_ids = token_ids[:pad_size]
							seq_len = pad_size
					contents.append((token_ids, label, seq_len, mask))
		random.shuffle(contents)
		return contents

	valid_num = -2000
	training_dataset = load_dataset(
						[args.train_cover_dir, args.train_stego_dir])
	train_data = training_dataset[:valid_num]
	valid_data = training_dataset[valid_num:]
	test_data = load_dataset([args.test_cover_dir, args.test_stego_dir])
	
	return train_data, valid_data, test_data

class DatasetIterater(object):
	def __init__(self, batches, args):
		self.batch_size = args.batch_size
		self.batches = batches
		self.n_batches = len(batches) // args.batch_size
		self.residue = False
		if len(batches) % self.n_batches != 0:
			self.residue = True
		self.index = 0
		self.device = args.device

	def _to_tensor(self, datas):
		x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
		y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

		seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
		mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
		return (x, seq_len, mask), y

	def __next__(self):
		if self.residue and self.index == self.n_batches:
			batches = self.batches[self.index*self.batch_size:len(self.batches)]
			self.index += 1
			batches = self._to_tensor(batches)
			return batches

		elif self.index > self.n_batches:
			self.index = 0
			raise StopIteration

		else:
			batches = self.batches[
					  self.index*self.batch_size:(self.index+1)*self.batch_size]
			self.index += 1
			batches = self._to_tensor(batches)
			return batches
	
	def __iter__(self):
		return self
	
	def __len__(self):
		if self.residue:
			return self.n_batches + 1
		else:
			return self.n_batches


def build_iterator(dataset, args):
	iters = DatasetIterater(dataset, args)
	return iters



# Testing coding...

#if __name__ == '__main__':
#	import argparse
#	parser = argparse.ArgumentParser(description='data')
#	args = parser.parse_args()
#	args.train_cover_dir = '../../data/tina/coco/train_data/train_cover.txt'
#	args.train_stego_dir = '../../data/tina/coco/train_data/coco_1bpw.txt'
#	args.test_cover_dir = '../../data/tina/coco/test_data/test_cover.txt'
#	args.test_stego_dir = '../../data/tina/coco/test_data/coco_1bpw.txt'
#
#	args.model = BertModel.from_pretrained('bert_base_uncased/')
#	args.tokenizer = BertTokenizer.from_pretrained('bert_base_uncased/')
#	args.batch_size = 64
#	args.device = 'cuda'
#
#	train_data, valid_data, test_data = build_dataset(args) 
#
#	train_iter = build_iterator(train_data, args)
#	valid_iter = build_iterator(valid_data, args)
#	test_iter = build_iterator(test_data, args)
#	print(len(train_iter))
#	print(len(valid_iter))
#	print(len(test_iter))
	




