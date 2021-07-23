import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam


def train(train_iter, dev_iter, model, args):
	if args.cuda:
		model.cuda()
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer \
					if not any(nd in n for nd in no_decay)], 'weight':0.01},
		{'params': [p for n, p in param_optimizer \
					if any(nd in n for nd in no_decay)], 'weight':0.0}]

	optimizer = BertAdam(optimizer_grouped_parameters,
						 lr=args.lr,
						 warmup=0.05,
						 t_total=len(train_iter)*args.epochs)

	steps = 0
	best_acc = 0
	last_step = 0
	model.train()

	for epoch in range(1, args.epochs+1):
		print('\n--------training epochs: {}-----------'.format(epoch))
		for batch in train_iter:
			feature, target = batch[0], batch[1]

			optimizer.zero_grad()
			import time
			torch.cuda.synchronize()
			start = time.time()
			logit = model(feature)
			torch.cuda.synchronize()
			end = time.time()
			print(end - start)
			sys.exit()
			loss = F.cross_entropy(logit, target)
			loss.backward()
			optimizer.step()

			steps += 1
			if steps % args.log_interval == 0:
				corrects = (torch.max(logit, 1)[1].view(target.size()).data \
						 == target.data).sum()
				accuracy = corrects.item()/args.batch_size
				sys.stdout.write(
					'\rBatch[{}] - loss:{:.6f} acc:{:.4f}({}/{})'.format(
					steps, loss.item(), accuracy, corrects, args.batch_size))
		
			if steps % args.test_interval == 0:
				dev_acc, dev_loss = data_eval(dev_iter, model, args)
				if dev_acc > best_acc:
					best_acc = dev_acc
					last_step = steps
					if args.save_best:
						save(model, args.save_dir, 'best', steps)
				if epoch>10 and dev_loss > 0.9:
					print('\nthe validation is {}, training done...'.format(dev_loss))
					sys.exit(0)
				else:
					if steps - last_step >= args.early_stop:
						print('early stop by {} steps.'.format(args.early_stop))
				model.train()

			elif steps % args.save_interval == 0:
				save(model, args.save_dir, 'snapshot', steps)


def data_eval(data_iter, model, args):
	model.eval()
	corrects, avg_loss = 0, 0
	batch_num = 0
	logits = None
	targets = []
	for batch in data_iter:
		feature, target = batch[0], batch[1]
		with torch.no_grad():
			logit = model(feature)

		if logits is None:
			logits = logit
		else:
			logits = torch.cat([logits, logit], 0)

		targets.extend(target.tolist())

		loss = F.cross_entropy(logit, target)
		batch_num += 1
		avg_loss += loss.item() 
		corrects += (torch.max(logit, 1)[1].view(target.size()).data \
					 == target.data).sum()
	avg_loss /= batch_num
	size = 2000
	accuracy_ = corrects.item()/size
	if not args.test:   # validation phase
		print('\nValidation - loss:{:.6f} acc:{:.4f}({}/{})'.format(
			avg_loss, accuracy_, corrects, size))
		return accuracy_, avg_loss
	
	else:   # testing phase
	
		# mertrics
		from sklearn import metrics
		predictions = torch.max(logits, 1)[1].cpu().detach().numpy()
		labels = np.array(targets)
		accuracy = metrics.accuracy_score(labels, predictions)
		precious = metrics.precision_score(labels, predictions)
		recall = metrics.recall_score(labels, predictions)
		F1_score = metrics.f1_score(labels, predictions, average='weighted')
		TN = sum((predictions == 0) & (labels == 0))
		TP = sum((predictions == 1) & (labels == 1))
		FN = sum((predictions == 0) & (labels == 1))
		FP = sum((predictions == 1) & (labels == 0))
		print('\nTesting - loss:{:.6f} acc:{:.4f}({}/{})'.format(
			loss, accuracy, corrects, size))
		result_file = os.path.join(args.save_dir, 'result.txt')
		with open(result_file, 'a', errors='ignore') as f:
			f.write('The testing accuracy: {:.4f} \n'.format(accuracy))
			f.write('The testing precious: {:.4f} \n'.format(precious))
			f.write('The testing recall: {:.4f} \n'.format(recall))
			f.write('The testing F1_score: {:.4f} \n'.format(F1_score))
			f.write('The testing TN: {} \n'.format(TN))
			f.write('The testing TP: {} \n'.format(TP))
			f.write('The testing FN: {} \n'.format(FN))
			f.write('The testing FP: {} \n\n'.format(FP))
		return accuracy, recall, precious, F1_score


def save(model, save_dir, save_prefix, steps):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, save_prefix)
	save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
	torch.save(model.state_dict(), save_path)
			
