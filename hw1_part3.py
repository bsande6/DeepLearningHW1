from audioop import cross
from configparser import Interpolation
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from network import Model_1, Model_2, MNIST_CNN, MNIST_CNN2, MNIST_CNN3,MNIST_CNN4,MNIST_CNN5,MNIST_CNN6,MNIST_CNN7,MNIST_CNN8,MNIST_CNN9,MNIST_CNN10
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


def load_mnist_data(batch_size):
 
    train_mnist = dset.MNIST(
         root='data', 
         train=True, 
         download=True, 
         transform=transforms.ToTensor()
    )
    random.shuffle(train_mnist.targets)
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
    random.shuffle(test_mnist.targets)
    test_dataloader = DataLoader(
        test_mnist, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4
    )
    return train_mnist, train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optimizer):
    accu_number = 0
    for x, y in dataloader:
      
        x, y = x.to(device), y.to(device)
        pred = model(x.float())
        predicted_class = torch.argmax(model(x),dim=1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accu_number += torch.sum(predicted_class == y)
        return loss, pred, accu_number

def test(test_dataloader, model, loss_fn):
    with torch.no_grad():
        accu_number = 0.0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            predicted_class = torch.argmax(model(x),dim=1)
            pred = model(x.float())
            loss = loss_fn(pred, y)
            accu_number += torch.sum(predicted_class == y)
        print('testing accuracy: %.4f' % (accu_number/len(test_dataloader.dataset)))
        return(accu_number, loss)

def plot(loss_1, loss_2, epochs, path):
    plt.figure(0)
    epochs = range(0,epochs)
    plt.plot(epochs, loss_1, 'g', label='Train')
    plt.plot(epochs, loss_2, 'b', label='Test')
    plt.title('Random Shuffle Training vs Testing loss')
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
    print(loss)
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
    
def plot_param(params, test_loss, train_loss, test_acc, train_acc):
    plt.figure(20)
    print(params, test_loss)
    test_loss = list(test_loss)
    train_loss = list(train_loss)
    plt.scatter(params, test_loss, label='Test')
    plt.scatter(params, train_loss, label='Train')
    plt.title('Parameters vs Loss')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Loss')
    plt.legend()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/params_loss.png')
    plt.figure(21)
    plt.scatter(params, test_loss, label='Test')
    plt.scatter(params, train_loss, label='Train')
    plt.title('Parameters vs Accuracy')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.legend()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/params_accuracy.png')
    
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

def plot_interpolated_model_loss(loss, weights):
    plt.figure(4)
    plt.plot(weights,loss, 'g', label='Test')
    plt.title('Interpolation')
    plt.xlabel('Alpha')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.savefig('plots//interpolated_model.png')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    state_dicts = [] 
    for w in weight:
        state_dicts.append({key: (1 - w) * state_dict_1[key] + w * state_dict_2[key]
            for key in state_dict_1.keys()})
    return state_dicts

def plot_sensitivity(train_loss, test_loss, sensitivities, batch_sizes):
    plt.figure(5)
    ax1 = plt.plot(batch_sizes, train_loss, label='train')
    plt.plot(batch_sizes, test_loss, label='test')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Cross Entropy Loss')
    ax2 = ax1.twinx() 
    ax2.set_ylabel("Sensitivity")
    ax2.plot(batch_sizes, sensitivities, 'r', label='sensitivity')
    #axR = plt.subplot(1,1,1, sharex=axL, frameon=False)
    #ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

def main():
    hyper_param = {
        'batch_size': 2,
        'n_epochs':25,
        'optimizer': 'Adam',
        'optim_param': {
            ## This dict should be changed according to the selection of optimizer ##
            'lr': 0.01,
        }
    }
    models_list =  []
    # Training on 10 models
    model = MNIST_CNN().to(device)
    models_list.append(model)
    models_list.append(model)
    models_list.append(model)
    models_list.append(model)
    models_list.append(model)
    model = MNIST_CNN2().to(device)
    models_list.append(model)
    model = MNIST_CNN3().to(device)
    models_list.append(model)
    model = MNIST_CNN4().to(device)
    models_list.append(model)
    model = MNIST_CNN5().to(device)
    models_list.append(model)
    model = MNIST_CNN6().to(device)
    models_list.append(model)
    model = MNIST_CNN7().to(device)
    models_list.append(model)
    model = MNIST_CNN8().to(device)
    models_list.append(model)
    model = MNIST_CNN9().to(device)
    models_list.append(model)
    model = MNIST_CNN10().to(device)
    models_list.append(model)
    
    train_mnist, train_dataloader, test_dataloader = load_mnist_data(hyper_param['batch_size'])
    train_mnist1, train_dataloader1, test_dataloader1 = load_mnist_data(4)
    train_mnist1, train_dataloader2, test_dataloader2 = load_mnist_data(8)
    train_mnist, train_dataloader3, test_dataloader3 = load_mnist_data(10)
    train_mnist, train_dataloader4, test_dataloader4 = load_mnist_data(20)

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
    grad_norms = []

    for t in range(hyper_param['n_epochs']):
        print(f"Epoch {t}", end=' ')
        loss = train(train_dataloader, models_list[0], loss_fn, optimizer)
        #train_loss.append(loss[0])
        print("train_loss", train_loss)
        #acc, loss = test(test_dataloader, models_list[0], loss_fn)
        #test_loss.append(loss)

        loss = train(train_dataloader1, models_list[1], loss_fn, optimizer)
        #train_loss.append(loss[0])
        #acc, loss = test(test_dataloader1, models_list[1], loss_fn)
        #test_loss.append(loss)

        loss = train(train_dataloader2, models_list[2], loss_fn, optimizer)
        #acc, loss = test(test_dataloader2, models_list[2], loss_fn)
        
        loss = train(train_dataloader3, models_list[3], loss_fn, optimizer)
        #acc, loss = test(test_dataloader3, models_list[3], loss_fn)
        

        loss = train(train_dataloader4, models_list[4], loss_fn, optimizer)
        train_loss.append(loss[0])
        acc, loss = test(test_dataloader4, models_list[4], loss_fn)
        test_loss.append(loss)
        
        loss = train(train_dataloader, models_list[4], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[4], loss_fn)
        
        
        loss = train(train_dataloader, models_list[5], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[5], loss_fn)
        
        
        loss = train(train_dataloader, models_list[6], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[6], loss_fn)
     
        
        loss = train(train_dataloader, models_list[7], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[7], loss_fn)
       
        
        loss = train(train_dataloader, models_list[8], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[8], loss_fn)
       
        
        loss = train(train_dataloader, models_list[9], loss_fn, optimizer)  
        #acc, loss = test(test_dataloader, models_list[9], loss_fn)
        
        
        loss = train(train_dataloader, models_list[10], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[10], loss_fn)
    
        loss = train(train_dataloader, models_list[11], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[11], loss_fn)
        
        loss = train(train_dataloader, models_list[12], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[12], loss_fn)
        
        loss = train(train_dataloader, models_list[13], loss_fn, optimizer)
        #acc, loss = test(test_dataloader, models_list[13], loss_fn)
        
    grad_all =0
    grad = 0
    grad_list = []

    sensitivity_list = []
    path = "MNIST_random_loss"
    plot(train_loss, test_loss, hyper_param['n_epochs'], path)
    test_losses= [] 
    train_losses= []
    num_params = []
    test_loss_param = []
    train_loss_param = []
    train_acc = []
    test_acc = []
    for i in range(4, 14):
      num_params.append(count_parameters(models_list[i]))
      acc, loss = test(test_dataloader, models_list[i], loss_fn)
      test_acc.append(float(acc))
      
      test_loss_param.append(float(loss))
      print("loss", test_loss_param)
      loss, pred, acc = train(train_dataloader, models_list[i], loss_fn, optimizer)
      train_loss_param.append(float(loss))
      print(train_loss)
      test_acc.append(acc)
    print(num_params)
    print("test", test_loss)
    print("train", train_loss)  
    plot_param(num_params, test_loss_param, train_loss_param, test_acc, train_acc)
    for i in range(0, 5):
        grad_array = []
        for name, param in models_list[i].named_parameters():
            loss = train(train_dataloader, models_list[0], loss_fn, optimizer)
            
            print(name)
            if param.grad is not None:
                grad = (param.grad.cpu().data.numpy() ** 2).sum()
                if 'weight' in name:
                    grad_list.append((np.ndarray.flatten(param.grad.cpu().data.numpy())))
                    

            grad_all += grad
   
        train_losses.append(loss[0])
     
        if len(grad_list) > 1:
                    c = np.concatenate(grad_list)
        elif len(grad_list) == 1:
                    c = grad_list[0]
        else:
                    c = None
       
        grad_array.append(c)
        grad_norm = grad_all ** 0.5
        grad_norms.append(grad_norm)
        iter = range(0, hyper_param['n_epochs'])
        
        acc, loss = test(test_dataloader, models_list[i], loss_fn)
        test_losses.append(loss)
        print(loss)
        # models_list[0].save_state_dict()
        # models_list[1].save_state_dict()
        # for name, param in models_list[0]:
        #     new_param = ((1-alpha) * param) + (alpha*param_2)
        alphas = (x/100 for x in range(-100, 200, 1))
        print(alphas)
        new_state_dicts = interpolate_state_dicts(models_list[0].state_dict(), models_list[1].state_dict(), alphas)
        interpolated_model = MNIST_CNN()
        cross_entropy = []
        for new_state_dict in new_state_dicts: 
            interpolated_model.load_state_dict(new_state_dict)
            interpolated_model.cuda()
            acc, loss = test(test_dataloader, interpolated_model, loss_fn)
            cross_entropy.append(loss)
        alpha = []
        for x in range(-100, 200, 1):
            alpha.append(x/100)
        
        grad_array = np.vstack(grad_array) 
        matrix = np.ones((2,2))
        sensitivity = np.linalg.norm(grad_array, ord='fro')
        sensitivity_list.append(sensitivity)
    batch_sizes = [4, 10, 25, 40, 50]
  
    plot_interpolated_model_loss(cross_entropy, alpha)
    plot_sensitivity(train_losses, test_losses, sensitivity_list, batch_sizes)
    

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main()
