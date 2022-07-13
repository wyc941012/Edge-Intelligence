import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
from data import get_data_set
import socket
import threading
import pickle
import io
import sys
import time

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10

ALEXNET_MODEL_PATH="model/alexnetlayermodel.pkl"
VGG16_MODEL_PATH="model/vgg16layermodel.pkl"

IP="192.168.123.10"
PORT=8081

class Data(object):

	def __init__(self, inputData, startLayer, endLayer):
		self.inputData=inputData
		self.startLayer=startLayer
		self.endLayer=endLayer

def run(model, inputData, startLayer, endLayer):
	print("Mobile running %d to %d layer" % (startLayer, endLayer))
	outputs = model(inputData, startLayer, endLayer, False)
	return outputs

def test(outputs, test_x, test_y):
	prediction = torch.max(outputs.data, 1)
	correct_classified += np.sum(prediction[1].numpy() == test_y.numpy())
	acc=(correct_classified/len(test_x))*100
	return acc

def sendData(client, inputData, startLayer, endLayer):
	data=Data(inputData, startLayer, endLayer)
	str=pickle.dumps(data)
	client.send(len(data).to_bytes(length=6, byteorder='big'))
	client.send(data)

def receiveData(client, model, x, test_x, test_y):
	while True:
		lengthData=client.recv(6)
		length=int.from_bytes(lengthData, byteorder='big')
		b=bytes()
		count=0
		while True:
			value=client.recv(length)
			b=b+value
			count+=len(value)
			if count>=length:
				break
		data=pickle.loads(b)
		if data.startLayer>=len(x):
			acc=test(outputs, test_x, test_y)
			end=time.time()
			print("Compute task completed with response time: %f, accuracy: %f" % (runtime, acc))
			client.close()
			break
		else:
			count=0
			for i in range(startLayer, len(x)):
				if x[i]==1:
					break
				count=i
			outputs=run(model, test_x, startLayer, count)
			if count==len(x)-1:
				acc=test(outputs, test_x, test_y)
				end=time.time()
				print("Compute task completed with response time: %f, accuracy: %f" % (runtime, acc))
				client.close()
				break
			else:
				endLayer=0
				for i in range(count+1, len(x)):
					if x[i]==0:
						break
					endLayer=i
				sendData(client, outputs, count+1, endLayer)

if __name__=="__main__":
	model=torch.load(ALEXNET_MODEL_PATH, map_location='cpu')
	device = torch.device("cpu")
	torch.set_num_threads(3)
	test_x,test_y,test_l=get_data_set("test")
	test_x=torch.from_numpy(test_x[0:100]).float()
	test_y=torch.from_numpy(test_y[0:100]).long()
	print("Model loaded successfully.")
	client=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client.connect((IP, PORT))
	print("Cloud connection successful, ready for computing mission")
	print("Task submitted. Unload decision made")
	x=[1,1,1,1,1,1,1,1,1,1,1,1,1]
	print("Start running computational tasks")
	start=time.time()
	if x[0]==1:
		count=0
		for i in range(1, len(x)):
			if x[i]==0:
				break
			count=i
		sendData(client, outputs, 0, count)
		t = threading.Thread(target=receiveData, name='receiveData', args=(client, model, x, test_x, test_y))
		t.start()

	else:
		count=0
		for i in range(1, len(x)):
			if x[i]==1:
				break
			count=i
		outputs=run(model, test_x, 0, count)
		if count==len(x)-1:
			acc=test(outputs, test_x, test_y)
			end=time.time()
			print("Compute task completed with response time: %f, accuracy: %f" % (runtime, acc))
			client.close()
		else:
			endLayer=0
			for i in range(count+1, len(x)):
				if x[i]==0:
					break
				endLayer=i
			sendData(client, outputs, count+1, endLayer)
			t = threading.Thread(target=receiveData, name='receiveData', args=(client, model, x, test_x, test_y))
			t.start()




