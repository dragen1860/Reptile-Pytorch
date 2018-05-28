import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F


class Naive(nn.Module):
	"""
	Define your network here.
	"""
	def __init__(self, n_way, imgsz):
		super(Naive, self).__init__()

		if imgsz > 28: # for mini-imagenet
			self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.MaxPool2d(3,2)

			                         )
		else: # for omniglot
			self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.AvgPool2d(kernel_size=2),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True),

			                         nn.Conv2d(64, 64, kernel_size=3),
			                         nn.BatchNorm2d(64),
			                         nn.ReLU(inplace=True)

			                         )

		# dummy forward to get feature size
		dummy_img = Variable(torch.randn(2, 3, imgsz, imgsz))
		repsz = self.net(dummy_img).size()
		_, c, h, w = repsz
		self.fc_dim = c * h * w

		self.fc = nn.Sequential(nn.Linear(self.fc_dim, 64),
		                        nn.ReLU(inplace=True),
		                        nn.Linear(64, n_way))

		self.criteon = nn.CrossEntropyLoss()

		print(self)
		print('Naive repnet sz:', repsz)

	def forward(self, x, target):
		x = self.net(x)
		x = x.view(-1, self.fc_dim)
		pred = self.fc(x)
		loss = self.criteon(pred, target)

		return loss, pred
