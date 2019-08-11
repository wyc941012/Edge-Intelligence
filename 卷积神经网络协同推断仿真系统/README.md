1.程序运行步骤：
==============
在树莓派和PC机上分别安装Python以及PyTorch，提供神经网络的运行环境。      

(1)将cloud文件夹放到PC机上，运行initCloud.py文件，启动服务端。   

![image](https://github.com/wyc941012/Edge-Intelligence/blob/master/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8D%8F%E5%90%8C%E6%8E%A8%E6%96%AD%E4%BB%BF%E7%9C%9F%E7%B3%BB%E7%BB%9F/image/cloud.jpg)  
(2)将mobile文件夹放到树莓派上，运行initMobile.py文件，传入参数分别为模型名称和最大可接受费用。程序将根据网速、费用等因素进行最优决策，得到神经网络模型每一层的最优运行位置(在移动端或者云端)，程序根据卸载决策分配计算任务，完成神经网络模型的推断计算。      

![image](https://github.com/wyc941012/Edge-Intelligence/blob/master/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8D%8F%E5%90%8C%E6%8E%A8%E6%96%AD%E4%BB%BF%E7%9C%9F%E7%B3%BB%E7%BB%9F/image/mobile.jpg)     

2.程序说明：
=============
cloud和mobile目录分别代表云端和移动端运行程序，使用PC机仿真云端，树莓派仿真移动端。      

本实验在cifar-10数据集上训练了AlexNet和VGG16模型。其中datasets目录存放cifar-10数据集，model目录存放训练好的CNN模型。程序借助PyTorch框架的特性，改写模型的forward方法，训练模型后，以层为粒度运行计算任务，实现模型的分层推断。  
~~~
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
~~~
