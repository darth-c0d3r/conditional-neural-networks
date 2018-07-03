import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import c_nn_hard_routing
import c_nn_stochastic_routing
import time

# hyper-parameters
batch_size = 128
feat_dropout = 0.5
epochs = 20
report_every = 16
conv = [1,32,64]
tree = [60,20]
fc = []
n_children = 3
n_pass = 1

# change gpuid to use GPU
cuda = 0 
gpuid = -1
n_class = 10
size = 28

# return normalized dataset divided into two sets
def prepare_db():
	train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
												   torchvision.transforms.Normalize((0.1307,), (0.3081,))
											   ]))

	eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
												   torchvision.transforms.Normalize((0.1307,), (0.3081,))
											   ]))
	return {'train':train_dataset,'eval':eval_dataset}

# model = c_nn_stochastic_routing.c_nn(size, conv, tree, fc, n_class, n_children, feat_dropout)
model = c_nn_hard_routing.c_nn(size, conv, tree, fc, n_class, n_children, n_pass, feat_dropout)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

def train(model, optim, db, filename):

	time_start = time.time()

	for epoch in range(1, epochs+1):

		train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)

		# Update (Train)
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,target)
			loss.backward()
			optimizer.step()

			if batch_idx % report_every == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item()))


		# Evaluate
		model.eval()
		test_loss = float(0)
		correct = 0
		test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=batch_size, shuffle=True)
		for data, target in test_loader:
			with torch.no_grad():
				data= Variable(data)
			target = Variable(target)
			output = model(data)
			test_loss += criterion(output, target).item() # sum up batch loss
			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		# test_loss /= len(test_loader.dataset)
		accuracy = float(correct) / len(test_loader.dataset)

		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
			test_loss, correct, len(test_loader.dataset),
			accuracy))
		with open(filename, 'a+') as f:
			f.write(str(epoch)+" "+str(time.time()-time_start)+" "+str(test_loss)+" "+str(accuracy)+"\n")
					

def main():
	db = prepare_db()
	filename = "c_nn_hard_routing_3_1.dat"
	train(model, optim, db, filename)


if __name__ == '__main__':
	main()