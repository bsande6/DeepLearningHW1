from fileinput import lineno
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
    # Simulated function is y = sinx * x^5
    def __init__(self):
        # Simulated function is y = sinx * x^5
        # Create a dataset that maps x values to y values using the above function
        self.x = np.asarray([i for i in range(-10,101)])
        #self.y = np.asarray([math.sin(.1*i) for i in self.x])
        self.y = np.asarray([math.sin(.1*i) for i in self.x])

        self.x = self.x.reshape((len(self.x), 1))
        self.y = self.y.reshape((len(self.y), 1))

        self.n_samples = self.x.shape[0] 

        # Scale the values to be between 0 and 1 due to the large values of the outputs
        scale_x = MinMaxScaler()
        self.x = scale_x.fit_transform(self.x)
        scale_y = MinMaxScaler()
        self.y = scale_y.fit_transform(self.y)
      
    # support indexing such that dataset[i] can 
    # be used to get i-th sample
    def __getitem__(self, index):
        return self.x[index], self.y[index]    
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class PcaMnistDataset():
    def __init__(self):
        self.train_mnist = dset.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
        X = self.train_mnist.data
        X = X.reshape((X.shape[0], -1))
        
        #standardized_data= StandardScaler().fit(X)
        standardized_data = StandardScaler().fit_transform(X)
        
        pca = PCA(n_components=2)
        pca.fit(standardized_data)
        self.data = pca.transform(standardized_data)
        #self.test_mnist = dset.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

    def __getitem__(self, index):
        return self.data[index], self.train_mnist.targets[index]
    
    def __len__(self):
        return self.data.shape[0]


def load_mnist_data(batch_size):
    train_mnist = dset.MNIST(
         root='data', 
         train=True, 
         download=True, 
         transform=transforms.ToTensor()
    )
   
    
    #train_mnist = PcaMnistDataset()
  
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
    #test_mnist
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
        loss = loss_fn(pred, y.float())
        #print('loss', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        return loss, pred, x, y
        
def train_categorical(dataloader, model, loss_fn, optimizer):
   
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        #y = torch.Tensor(y.cuda())
        #y = y.type(torch.cuda.FloatTensor)
        
        #hessian = torch.autograd.functional.hessian(loss_fn, (x.float(),y))
        #print("hess", hessian)
   
        pred = model(x.float())
        loss = loss_fn(pred, y)
        hessian = torch.autograd.functional.hessian(loss_fn, (pred, y))
        hessian = np.asarray(hessian)
        #hessian = torch.autograd.functional.hessian(loss, (x.float(),y.float()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, pred, x, y


def test(test_dataloader, model):
    with torch.no_grad():
        accu_number = 0.0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            
            predicted_class = torch.argmax(model(x),dim=1)
            accu_number += torch.sum(predicted_class == y)
        print('testing accuracy: %.4f' % (accu_number/len(test_dataloader.dataset)))
        return(accu_number)
        
def pca(matrix):
      X = matrix
      standardized_data = StandardScaler().fit_transform(X)
      pca = PCA(n_components=2)
      pca.fit(standardized_data)
      data = pca.transform(standardized_data)
      return data

def plot(loss_1, loss_2, epochs_1, epochs_2, path):
    plt.figure(0)
    epochs_1 = range(0,epochs_1)
    epochs_2 = range(0, epochs_2)
    plt.plot(epochs_1, loss_1, 'g', label='Model 1')
    plt.plot(epochs_2, loss_2, 'b', label='Model 2')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/{}.png'.format(path))

def plot_predictions(predictions_1, predictions_2, x, groundtruth, x2, groundtruths2, x1, y, path):
    plt.figure(1)
    plt.plot(x1, predictions_1, 'g', label='Model 1')
    plt.plot(x1, predictions_2, 'b', label='Model 2')
    plt.plot(x1, y, 'r', label='Model 3')
    plt.title('Prediction vs Groundtruth')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('plots/{}.png'.format(path))
  
    
def plot_mnist_loss(loss, epochs, path):
    plt.figure(2)
    #print(loss)
    epochs = range(0,epochs)
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

def plot_grad(x, y, loss, path, i):
    plt.figure(5+i)
    plt.plot(x, y, 'g', label='Gradient Norm')
    plt.plot(x, loss, 'b', label='Loss')
    plt.title(path)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.legend()
    #plt.show()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/{}.png'.format(path))
    
def plot_weights(data, path):
    plt.figure(4)
    #print(data.shape)
    for weight in data:
        #weight = np.asarray(weight)
        x_list = []
        y_list = []
        for tuple in weight:
            x_list.append(tuple[0])
            y_list.append(tuple[1])
        plt.title(path)
        plt.scatter(x_list, y_list)
        #plt.scatter(weight[:,0], weight[:,1])

        if not os.path.exists('plots'):
            os.makedirs('plots')
    plt.savefig('plots/{}.png'.format(path))

def plot_min_ratio(loss, ratios, path):
    plt.figure(4)
    loss = np.asarray(loss)
    plt.scatter(ratios, loss, c='g')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/{}.png'.format(path))


def main():
    dataset = SimulatedFunctionDataSet()
    loss_fn = torch.nn.MSELoss()
   
    hyper_param = {
        'batch_size': 1,
        'n_epochs':500,
        'optimizer': 'Adam',
        'optim_param': {
            ## This dict should be changed according to the selection of optimizer ##
            'lr': 0.01,
        }
    }
    min_ratios = []
    losses = []
    # Train 100 times
    weights_list = []
    layer0_weights = []
    #weights = np.array()
    grad_norms = []
    train_loss = []
    
    for i in range(0, 100):
        dataloader = DataLoader(dataset=dataset, batch_size=hyper_param['batch_size'], shuffle=True)

        model = Model_1().to(device)
        optimizer = getattr(torch.optim, hyper_param['optimizer'])(
            model.parameters(), 
            **hyper_param['optim_param']
        )

        epochs = hyper_param['n_epochs']
        
        layer0_params = []
        weights = []
    
        grad_min = 1000000000
        for t in range(epochs):
            grad_all = 0.0
            # print(f"Epoch {t}", end=' ')
            loss, prediction, x, y = train(dataloader, model, loss_fn, optimizer)
           
            param_list =[]
            for name, param in model.named_parameters():
                if 'weight' in name:      
                    parm=param.detach().cpu().clone().numpy()   
                    grad = 0.0
                    if param.grad is not None:
                        grad =(param.grad.cpu().data.numpy() ** 2).sum()
                    grad_all += grad
                if t % 3 == 0:
                    if 'layers.0.weight' in name:
                        layer0_params.append(np.ndarray.flatten(parm))
                    if 'weight' in name:
                        param_list.append(np.ndarray.flatten(parm))

            grad_norm = grad_all ** 0.5
            if i < 9:
                if len(param_list) > 1:
                    c = np.concatenate(param_list)
                elif len(param_list) == 1:
                    c = param_list[0]
                else:
                    c = None

                if c is not None:
                    weights.append(c)
            
            if i == 0:
                grad_norms.append(grad_norm)
                train_loss.append(loss)
                
            #print(grad_norm)
            if grad_norm < grad_min:
                        grad_min = grad_norm
                        state_dicts = model.state_dict()
            
        if i < 9:
            weights_list.append(weights)  
            layer0_weights.append(layer0_params)     

        grad_min_model = Model_1()
        grad_min_model.load_state_dict(state_dicts)
        grad_min_model.to(device)
        pred = grad_min_model(x.float())
        hessian = torch.autograd.functional.hessian(loss_fn, (pred, y.float()))
        print(pred, y.float())
        hessian = torch.autograd.functional.hessian(loss_fn, (pred, y.float()))
        hessian = np.asarray(hessian, dtype=float)
        hess_loss = loss_fn(pred, y)
        #print("loss", hess_loss)
        count = 0
        min_count = 0
        w, v = np.linalg.eig(hessian)
        #print(w)
        for eigenvalue in w:
            count+=1
            if eigenvalue > 0:
                min_count+=1

        min_ratio = min_count/count
        min_ratios.append(min_ratio)
        losses.append(hess_loss)
    
    path = "Minimum_ratios"
    plot_min_ratio(losses, min_ratios, path)
    
    for i in range (0, 8):
        weights_list[i] = np.vstack(weights_list[i])   
        weights_list[i]= pca(weights_list[i])
        layer0_weights[i] = np.vstack(layer0_weights[i])
        layer0_weights[i] = pca(layer0_weights[i])
    #weights = np.column_stack(len(weights))
    
    path = "X^2_weights"
    plot_weights(weights_list, path)
    #pca_data = pca(layer0_weights)
    path = "X^2_weights_layer0"
    plot_weights(layer0_weights, path)
    iter = range(0, hyper_param['n_epochs'])
    path = "X^2_grad"
   
    plot_grad(iter, grad_norms, train_loss, path, 3)

    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main()
