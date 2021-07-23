import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam
import torch.nn as nn
import torch.optim as optim


def train(train_iter, dev_iter, args, student_model, teacher_model=None):
	if not teacher_model == None:
		if args.cuda:
			teacher_model.cuda()
			student_model.cuda()
		criterion1 = nn.CrossEntropyLoss()
		criterion2 = nn.KLDivLoss(reduction='batchmean')
		optimizer = optim.Adam(student_model.parameters(), lr=0.001)
	else:
		if args.cuda:
			student_model.cuda()
		optimizer = optim.Adam(student_model.parameters(), lr=0.00005)

	steps = 0
	best_acc = 0
	last_step = 0
	student_model.train()

	for epoch in range(1, args.epochs+1):
		print('\n--------training epochs: {}-----------'.format(epoch))
		for batch in train_iter:
			feature, target = batch[0], batch[1]
			optimizer.zero_grad()
			if not teacher_model == None:
				student_out = student_model(feature)
				teacher_out = teacher_model(feature)
				loss1 = criterion1(student_out, target)

				T = 2 
				alpha = 0.95
				outputs_S = F.log_softmax(student_out/T, dim=1)
				outputs_T = F.softmax(teacher_out/T, dim=1)
				loss2 = criterion2(outputs_S, outputs_T)*T*T

				loss = loss1 * (1-alpha) + loss2 * alpha
			else:
				student_out = student_model(feature)
				loss = F.cross_entropy(student_out, target)
			loss.backward()
			optimizer.step()

			steps += 1
			if steps % args.log_interval == 0:
				corrects = (torch.max(student_out, 1)[1].view(target.size()).data \
						 == target.data).sum()
				accuracy = corrects.item()/args.batch_size
				sys.stdout.write(
					'\rBatch[{}] - loss:{:.6f} acc:{:.4f}({}/{})'.format(
					steps, loss.item(), accuracy, corrects, args.batch_size))
		
			if steps % args.test_interval == 0:
				dev_acc, dev_loss = data_eval(dev_iter, student_model, args)
				if dev_acc > best_acc:
					best_acc = dev_acc
					last_step = steps
					if args.save_best:
						save(student_model, args.save_dir, 'best', steps)
				#if epoch>10 and dev_loss > 0.9:
				#	print('\nthe validation is {}, training done...'.format(dev_loss))
				#	sys.exit(0)
				#else:
				if steps - last_step >= args.early_stop:
					print('early stop by {} steps.'.format(args.early_stop))
				student_model.train()

			elif steps % args.save_interval == 0:
				save(student_model, args.save_dir, 'snapshot', steps)


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
		print('Testing - loss:{:.6f} acc:{:.4f}({}/{})\n'.format(
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


def save(model, save_dir, save_prefix, steps):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_prefix = os.path.join(save_dir, save_prefix)
	save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
	torch.save(model.state_dict(), save_path)
			
