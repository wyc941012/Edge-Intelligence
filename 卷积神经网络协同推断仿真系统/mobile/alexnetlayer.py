import torch.nn as nn

NUM_CLASSES = 10

class AlexNetLayer(nn.Module):
	def __init__(self, num_classes=NUM_CLASSES):
		super(AlexNetLayer, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1),
			nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=2.0),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
			nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1),
			nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=2.0),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=0),        
		)
		self.classifier = nn.Sequential(
			nn.Linear(2*2*128, 384),
			nn.Linear(384, 192),
			nn.Linear(192, NUM_CLASSES),
		)

	# def forward(self, x):
	# 	x = self.features(x)
	# 	x = x.view(x.size(0), 2*2*128)
	# 	x = self.classifier(x)
	# 	return x

	def forward(self, x, startLayer, endLayer, isTrain):
		if isTrain:
			x = self.features(x)
			x = x.view(x.size(0), 2*2*128)
			x = self.classifier(x)
		else:
			if startLayer==endLayer:
				if startLayer==10:
					x = x.view(x.size(0), 2*2*128)
					x = self.classifier[startLayer-10](x)
				elif startLayer<10:
					x = self.features[startLayer](x)
				else:
					x = self.classifier[startLayer-10](x)
			else:
				for i in range(startLayer, endLayer+1):
					if i<10:
						x = self.features[i](x)
					elif i==10:
						x = x.view(x.size(0), 2*2*128)
						x = self.classifier[i-10](x)
					else:
						x = self.classifier[i-10](x)
		return x