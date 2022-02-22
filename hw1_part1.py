import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from network import Model_1, Model_2, MNIST_CNN, PCA_MNIST_CNN
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import random
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fix the random seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False


class SimulatedFunctionDataSet():
    def __init__(self):
        # Simulated function is y = sinx * .1*x
        # Create a dataset that maps x values to y values using the above function
        self.x = np.asarray([i for i in range(-100,101)])
        #self.y = np.asarray([math.sin(.1*i) for i in self.x])
        self.y = np.asarray([i**2 for i in self.x])

        self.x = self.x.reshape((len(self.x), 1))
        self.y = self.y.reshape((len(self.y), 1))

        self.n_samples = self.x.shape[0] 

        # Scale the values to be between 0 and 1
        scale_x = MinMaxScaler()
        self.x = scale_x.fit_transform(self.x)
        scale_y = MinMaxScaler()
        self.y = scale_y.fit_transform(self.y)
      
    # support indexing such that dataset[i] can 
    # be used to get i-th sample
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return torch.tensor(x).float(), torch.tensor(y).float()
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


def load_mnist_data(batch_size):
 
    train_mnist = dset.MNIST(
         root='data', 
         train=True, 
         download=True, 
         transform=transforms.ToTensor()
    )
    
    train_dataloader = DataLoader(
        train_mnist, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4
    )

    test_mnist = dset.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()
    )
    test_mnist
    test_dataloader = DataLoader(
        test_mnist, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4
    )
    return train_mnist, train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optimizer):
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x.float())
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        return loss, pred, x, y

def test(test_dataloader, model):
    with torch.no_grad():
        accu_number = 0.0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            predicted_class = torch.argmax(model(x),dim=1)
            accu_number += torch.sum(predicted_class == y)
        print('testing accuracy: %.4f' % (accu_number/len(test_dataloader.dataset)))
        return((accu_number/len(test_dataloader.dataset)))

def plot(loss_1, loss_2, epochs_1, epochs_2, path):
    plt.figure(0)
    epochs_1 = range(0,epochs_1)
    epochs_2 = range(0, epochs_2)
    plt.plot(epochs_1, loss_1, 'g', label='Deep Network')
    plt.plot(epochs_2, loss_2, 'b', label='Shallow Network')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/{}.png'.format(path))

def plot_predictions(predictions_1, predictions_2, x, y, path):
    plt.figure(1)
    plt.plot(x, predictions_1, 'g', label='Deep Network')
    plt.plot(x, predictions_2, 'b', label='Shallow Network')
    plt.plot(x, y, 'r', label= 'Groundtruth')
    plt.title('Prediction vs Groundtruth')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('plots/{}.png'.format(path))
  
    
def plot_mnist_loss(loss, epochs, path):
    plt.figure(2)
    epochs = range(0,epochs)
    print(loss)
    plt.plot(epochs, loss, 'g', label='CNN')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/{}.png'.format(path))
    
def plot_mnist_acc(accuracy, epochs, path):
    plt.figure(3)
    epochs = range(0,epochs)
    plt.plot(epochs, accuracy, 'g', label='CNN')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/{}.png'.format(path))
    
def count_parameters(model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    dataset = SimulatedFunctionDataSet()
    loss_fn = torch.nn.MSELoss()
    # # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.90)
   
    hyper_param = {
        'batch_size': 50,
        'n_epochs':200,
        'optimizer': 'Adam',
        'optim_param': {
            ## This dict should be changed according to the selection of optimizer ##
            'lr': 0.01,
        }
    }
    
    dataloader = DataLoader(dataset=dataset, batch_size=hyper_param['batch_size'], shuffle=True)

    model = Model_1().to(device)
    optimizer = optimizer = getattr(torch.optim, hyper_param['optimizer'])(
        model.parameters(), 
        **hyper_param['optim_param']
    )

    epochs = hyper_param['n_epochs']
    model1_loss = []
    model1_pred = []
    x_vals = []
    groundtruths = []
    for t in range(epochs):
        #print("Epoch {t}", end=' ')
        loss, prediction, x, groundtruth = train(dataloader, model, loss_fn, optimizer)
        model1_loss.append(loss)
       
    print(count_parameters(model))
    iter = range(0, hyper_param['n_epochs'])
    model_1_epochs= epochs
    for x in dataset.x:
      tensor = (torch.from_numpy(x))
      tensor = tensor.to('cuda:0')
      pred = model(tensor.float())
      model1_pred.append(pred)
    # Training neural network with less layers and same parameters

    hyper_param = {
        'batch_size': 10,
        'n_epochs':200,
        'optimizer': 'Adam',
        'optim_param': {
            'lr': 0.01,
        }
    }

    dataloader = DataLoader(dataset=dataset, batch_size=hyper_param['batch_size'], shuffle=True)
    
    model = Model_2().to(device)
    optimizer = getattr(torch.optim, hyper_param['optimizer'])(
        model.parameters(), 
        **hyper_param['optim_param']
    )
    
    count_parameters(model)
    
    epochs = hyper_param['n_epochs']
    model2_loss = []
    model2_pred = []
    
    for t in range(epochs):
        loss, prediction, x, y = train(dataloader, model, loss_fn, optimizer)
        model2_loss.append(loss)
    
    iter = range(0, hyper_param['n_epochs'])

    model2_pred = []
    x2_vals = []
    
    for x in dataset.x:
   
      tensor = (torch.from_numpy(x))
      tensor = tensor.to('cuda:0')
      pred = model(tensor.float())
   
      model2_pred.append(pred)
    plot(model1_loss, model2_loss, model_1_epochs, epochs, "Simulated_Function_Loss")
    plot_predictions(model1_pred, model2_pred, dataset.x, dataset.y, "Predicted_Values")  
    # Training Network for MNIST dataset

    hyper_param = {
        'batch_size': 20,
        'n_epochs':100,
        'optimizer': 'Adam',
        'optim_param': {
            ## This dict should be changed according to the selection of optimizer ##
            'lr': 0.001,
        }
    }

    model = MNIST_CNN().to(device)
    train_mnist, train_dataloader, test_dataloader = load_mnist_data(hyper_param['batch_size'])

    # Loss for a classifier
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, hyper_param['optimizer'])(
        model.parameters(), 
        **hyper_param['optim_param']
    )
    train_loss = []
    test_loss = []
    test_acc = []
    weights = []
  
    for t in range(hyper_param['n_epochs']):
        print(f"Epoch {t}", end=' ')
        loss, p,c,v = train(train_dataloader, model, loss_fn, optimizer)
        #for i in range(0, hyper_param['batch_size']):
        train_loss.append(loss)
        acc = test(test_dataloader, model)
      
        
    
        #test_loss.append(loss)
        test_acc.append(acc)
    iter = range(0, hyper_param['n_epochs'])
    path = "MNIST"
    
    #plot(train_loss, test_loss, hyper_param['n_epochs'])
    print("Done!")
    
    plot_mnist_loss(train_loss, hyper_param['n_epochs'], "mnist_cnn")
    plot_mnist_acc(test_acc, hyper_param['n_epochs'], "mnist_cnn_accuracy")

    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main()
