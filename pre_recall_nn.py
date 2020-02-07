import numpy as np
import random 
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from numpy import save
from numpy import load


# this is one way to define a network

    
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2,2)
#         self.out = torch.nn.Linear(n_hidden2, n_output)
#         self.out_pred = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
#         x = F.tanh(x)
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
#         x = self.out(x)  
#         y = self.out(x)              # binary output
        return x
    
    def predict(self,x):
        #Apply softmax to output. 
        pred = self.forward(x)
#         print(self.forward(x))
        ans = []
#         print(pred[0])
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
                
#         print(self.forward(x).size())
#         return torch.tensor(ans).unsqueeze(1)
        return torch.tensor(ans)

    
prediction_y = []   
for i in range(2):
    
    for j in range(21):
        Xtrain = Variable(torch.Tensor(X_train_datasets_5d_resampled[j]))
        ytrain = Variable(torch.Tensor(y_train_datasets_5d_resampled[j]))
        if(len(ytrain.size()) > 1):
            ytrain = ytrain.squeeze(1)
        Xtest = Variable(torch.Tensor(X_test_datasets_5d_resampled[j]))


        net = Net(n_feature=5, n_hidden1=20 , n_hidden2=20, n_output=1)     # define the network

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss()  # this is for regression mean squared loss


        # train the network
        for epoch in range(1000):

            
            optimizer.zero_grad()   # clear gradients for next train
            
            prediction = net.forward(Xtrain)     # input x and predict based on x
            
#             print(ytrain.long().squeeze(1).size())
#             print(prediction.size())
            loss = loss_func(prediction, ytrain.long())     # must be (1. nn output, 2. target)

            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            if epoch % 100 == 0:
                print('epoch {}, loss {}'.format(epoch, loss.data)) 



        pred_y = net.predict(Xtest) 
        prediction_y.append(pred_y.detach().numpy())
