from omniglotNShot import OmniglotNShot
from meta import MetaLearner
from naive import Naive
from MiniImagenet import MiniImagenet

import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def main():
	meta_batchsz = 32
	n_way = 5
	k_shot = 1
	k_query = k_shot
	meta_lr = 1e-3
	num_updates = 5
	dataset = 'omniglot'



	if dataset == 'omniglot':
		imgsz = 28
		db = OmniglotNShot('dataset', batchsz=meta_batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=imgsz)

	elif dataset == 'mini-imagenet':
		imgsz = 84
		# the dataset loaders are different from omniglot to mini-imagenet. for omniglot, it just has one loader to use
		# get_batch(train or test) to get different batch.
		# for mini-imagenet, it should have two dataloader, one is train_loader and another is test_loader.
		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=imgsz)
		db = DataLoader(mini, meta_batchsz, shuffle=True, num_workers=4, pin_memory=True)
		mini_test = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=1000, resize=imgsz)
		db_test = DataLoader(mini_test, meta_batchsz, shuffle=True, num_workers=2, pin_memory=True)

	else:
		raise  NotImplementedError


	meta = MetaLearner(Naive, (n_way, imgsz), n_way=n_way, k_shot=k_shot, meta_batchsz=meta_batchsz, beta=meta_lr,
	                   num_updates=num_updates).cuda()

	tb = SummaryWriter('runs')


	# main loop
	for episode_num in range(200000):

		# 1. train
		if dataset == 'omniglot':
			support_x, support_y, query_x, query_y = db.get_batch('test')
			support_x = Variable( torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
			query_x = Variable( torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
			support_y = Variable(torch.from_numpy(support_y).long()).cuda()
			query_y = Variable(torch.from_numpy(query_y).long()).cuda()
		elif dataset == 'mini-imagenet':
			try:
				batch_test = iter(db).next()
			except StopIteration as err:
				mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
				                    batchsz=10000, resize=imgsz)
				db = DataLoader(mini, meta_batchsz, shuffle=True, num_workers=4, pin_memory=True)

			support_x = Variable(batch_test[0]).cuda()
			support_y = Variable(batch_test[1]).cuda()
			query_x = Variable(batch_test[2]).cuda()
			query_y = Variable(batch_test[3]).cuda()

		# backprop has been embeded in forward func.
		accs = meta(support_x, support_y, query_x, query_y)
		train_acc = np.array(accs).mean()

		# 2. test
		if episode_num % 30 == 0:
			test_accs = []
			for i in range(min(episode_num // 5000 + 3, 10)): # get average acc.
				if dataset == 'omniglot':
					support_x, support_y, query_x, query_y = db.get_batch('test')
					support_x = Variable( torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
					query_x = Variable( torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
					support_y = Variable(torch.from_numpy(support_y).long()).cuda()
					query_y = Variable(torch.from_numpy(query_y).long()).cuda()
				elif dataset == 'mini-imagenet':
					try:
						batch_test = iter(db_test).next()
					except StopIteration as err:
						mini_test = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot,
						                         k_query=k_query,
						                         batchsz=1000, resize=imgsz)
						db_test = DataLoader(mini_test, meta_batchsz, shuffle=True, num_workers=2, pin_memory=True)
					support_x = Variable(batch_test[0]).cuda()
					support_y = Variable(batch_test[1]).cuda()
					query_x = Variable(batch_test[2]).cuda()
					query_y = Variable(batch_test[3]).cuda()
 

				# get accuracy
				test_acc = meta.pred(support_x, support_y, query_x, query_y)
				test_accs.append(test_acc)

			test_acc = np.array(test_accs).mean()
			print('episode:', episode_num, '\tfinetune acc:%.6f' % train_acc, '\t\ttest acc:%.6f' % test_acc)
			tb.add_scalar('test-acc', test_acc)
			tb.add_scalar('finetune-acc', train_acc)


if __name__ == '__main__':
	main()
