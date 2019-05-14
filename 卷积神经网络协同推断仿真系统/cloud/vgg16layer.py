import torch.nn as nn

NUM_CLASSES = 10

class VGG16Layer(nn.Module):
	def __init__(self, num_classes=NUM_CLASSES):
		super(VGG16Layer, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size=2, stride=2),      
		)
		self.classifier = nn.Sequential(
			nn.Linear(512, 4096),
			nn.Linear(4096, 4096),
			nn.Linear(4096, NUM_CLASSES),
		)

	# def forward(self, x):
	# 	x = self.features(x)
	# 	x = x.view(x.size(0), 2*2*128)
	# 	x = self.classifier(x)
	# 	return x

	def forward(self, x, startLayer, endLayer, isTrain):
		if isTrain:
			x = self.features(x)
			x = x.view(x.size(0), 512)
			x = self.classifier(x)
		else:
			if startLayer==endLayer:
				if startLayer==31:
					x = x.view(x.size(0), 512)
					x = self.classifier[startLayer-31](x)
				elif startLayer<31:
					x = self.features[startLayer](x)
				else:
					x = self.classifier[startLayer-31](x)
			else:
				for i in range(startLayer, endLayer+1):
					if i<31:
						x = self.features[i](x)
					elif i==31:
						x = x.view(x.size(0), 512)
						x = self.classifier[i-31](x)
					else:
						x = self.classifier[i-31](x)
		return x