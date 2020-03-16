import csv
from numpy import *

import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in range(m):
        for j in range(n):
                newArray[i,j]=int(array[i,j])
    return newArray

def loadData(filename,train = 10000, test = 1000):
	l = []
	i = 0
	with open(filename) as file:
		lines = csv.reader(file)
		for line in lines:
			if i>train+test+5:
				break
			i+=1
			l.append(line)
	l.remove(l[0])
	l = array(l)
	label = l[:,0]
	data = l[:,1:]
	allData,allLabel = (toInt(data)),toInt(label)
	allLabel = allLabel.transpose()
	trainData = allData[:train][:]
	trainLabel = allLabel[:train]
	trainLabel = torch.from_numpy(trainLabel).long()
	trainData = torch.from_numpy(trainData).float()
	trainLabel = trainLabel.reshape(train)
	trainData = trainData.reshape(train,1,28,28)

	testData = allData[0-test:][:]
	testLabel = allLabel[0-test:]
	testLabel = torch.from_numpy(testLabel).long()
	testData = torch.from_numpy(testData).float()
	testLabel = testLabel.reshape(test)
	testData = testData.reshape(test,1,28,28)
	return trainData,trainLabel,testData,testLabel

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc1 = nn.Linear(320, 25)
        # self.fc2 = nn.Linear(100,10)


    def forward(self, x):
        in_size = x.size(0) # one batch
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc1(x)
        return F.log_softmax(x)

def train(epoch,data,target):
    for idx in range(data.shape[0]):
        data_1, target_1 = Variable(data[idx].reshape(1,1,28,28)), Variable(target[idx].reshape(1))
        optimizer.zero_grad()
        output = model(data_1)
        loss = F.nll_loss(output, target_1)
        train_loss.append(loss)
        loss.backward()
        optimizer.step()
        if idx % 500 == 0:
	        print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
	                epoch, idx, data.shape[0],
	                100.* idx / data.shape[0], loss.item()))

def loss(data,target):
    test_loss = 0
    correct = 0
    for idx in range(data.shape[0]):
        data_1, target_1 = Variable(data[idx].reshape(1,1,28,28)), Variable(target[idx].reshape(1))
        output = model(data_1)
        # sum up batch loss
        test_loss += F.nll_loss(output, target_1, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_1.data.view_as(pred)).cpu().sum()

    test_loss /= data.shape[0]
    test_loss_1.append(test_loss)
    test_accuracy_1.append(100.0*correct /data.shape[0])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, data.shape[0],
        100. * correct /data.shape[0] ))

if __name__ == "__main__":
	num_train = 10000
	num_test = 1000
	num_epoch = 5
	model = Net()
	print(model)
	data_path = './data/sign_mnist_train.csv'
	train_data,train_label,test_data,test_label = loadData(data_path,num_train,num_test)
	optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
	train_loss = []
	test_loss_1 = []
	test_accuracy_1 = []
	for epoch in range(1, num_epoch+1):
	    train(epoch,train_data,train_label)
	    print("**"*20)
	    loss(test_data,test_label)
	f1 = open('train_loss.txt','w+')
	f2 = open('test_loss.txt','w+')
	f3 = open('test_accuracy_1.txt','w+')
	for i in range(len(train_loss)):
		f1.write(str(train_loss[i])+'\n')
	for i in range(len(test_loss_1)):
		f2.write(str(test_loss_1[i])+'\n')
	for i in range(len(test_accuracy_1)):
		f3.write(str(test_accuracy_1[i])+'\n')
	f1.close()
	f2.close()
	f3.close()