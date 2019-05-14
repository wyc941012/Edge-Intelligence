import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys

def get_data_set(name="train", cifar=10):
	x = None
	y = None
	l = None

	folder_name = "cifar_10" if cifar == 10 else "cifar_100"

	f = open('datasets/'+folder_name+'/batches.meta', 'rb')
	datadict = pickle.load(f, encoding='latin1')
	f.close()
	l = datadict['label_names']

	if name is "train":
		for i in range(5):
			f = open('datasets/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
			datadict = pickle.load(f, encoding='latin1')
			f.close()

			_X = datadict["data"]
			_Y = datadict['labels']
			_X = np.array(_X, dtype=float) / 255.0

			_X = _X.reshape([-1, 3, 32, 32])

			_X = _X.transpose([0, 1, 2, 3])
                                           
			#_X = _X.reshape(-1, 32*32*3)

			if x is None:
				x = _X
				y = _Y
			else:
				x = np.concatenate((x, _X), axis=0)  
				y = np.concatenate((y, _Y), axis=0)

	elif name is "test":
		f = open('datasets/'+folder_name+'/test_batch', 'rb')
		datadict = pickle.load(f, encoding='latin1')
		f.close()

		x = datadict["data"]
		y = np.array(datadict['labels'])

		x = np.array(x, dtype=float) / 255.0
		x = x.reshape([-1, 3, 32, 32])
		x = x.transpose([0, 1, 2, 3])
		#x = x.reshape(-1, 32*32*3)

	def dense_to_one_hot(labels_dense, num_classes=10):
		num_labels = labels_dense.shape[0]
		index_offset = np.arange(num_labels) * num_classes
		labels_one_hot = np.zeros((num_labels, num_classes))
		labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

		return labels_one_hot

	return x, y, l

if __name__ == '__main__':
	print(get_data_set);

