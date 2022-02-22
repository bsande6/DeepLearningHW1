import torch

class Model_1(torch.nn.Module):
    # deep network
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
            torch.nn.ReLU(),

        )
        
    def forward(self, x):
        pred = self.layers(x)
         
        return pred

class Model_2(torch.nn.Module):
    # shallow network
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            #torch.nn.Flatten(start_dim=1, end_dim=-1),

            torch.nn.Linear(1, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 1),
            torch.nn.ReLU(),
            
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred

class MNIST_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.Conv2d(128, 256, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred
        
class PCA_MNIST_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 5),
            torch.nn.Linear(5, 10),
            #torch.nn.Linear(10, 10),
            #torch.nn.Linear(10, 10),
            #torch.nn.Linear(10, 10),
            #torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 32),
            torch.nn.Linear(32, 10),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred
        
class MNIST_CNN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(5, 10, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(10, 16, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.Conv2d(16, 32, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            
            torch.nn.Flatten(), 
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred

class MNIST_CNN3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
             torch.nn.Conv2d(5, 10, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(10, 16, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(16, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
           
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred
    
class MNIST_CNN4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.Conv2d(128, 128, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred

class MNIST_CNN5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(4, 16, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(16, 32, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.Conv2d(32, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 10),
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred
        
class MNIST_CNN6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(16, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.Conv2d(128, 256, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(256, 64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
            # The sigmoid function returns multiple labels rather than just selecting one
            #torch.nn.Sigmoid()
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred
        
class MNIST_CNN7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.Conv2d(128, 256, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(256, 64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
            # The sigmoid function returns multiple labels rather than just selecting one
            #torch.nn.Sigmoid()
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred
        
class MNIST_CNN8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(64, 200, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(200, 256, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(256, 64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred
        
class MNIST_CNN9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 15, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
           torch.nn.Conv2d(15, 32, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
            # The sigmoid function returns multiple labels rather than just selecting one
            #torch.nn.Sigmoid()
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred

class MNIST_CNN10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 15, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
           
            
            torch.nn.Conv2d(15, 64, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64, 100, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(100, 128, kernel_size=3, stride =2, padding=1),
            torch.nn.ReLU(),
     
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(), 
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
            # The sigmoid function returns multiple labels rather than just selecting one
            #torch.nn.Sigmoid()
     
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        pred = self.layers(x)
        return pred        
