#!/usr/bin/env python
# coding: utf-8

# # Deep Learning in Biomedical Engineering

# ## Assignment 2
# ### Due date:<font color='red'> 11:59 pm, Febraury 25, 2021</font>

# ## [Please enter your full name and UNI here]

# ## Introduction
# In this assignment, you will implement, train, and test [LeNet](https://en.wikipedia.org/wiki/LeNet) 
# model to classify [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset.
# 
# ## Instructions
# Depending on each question, there are empty blocks you need to fill. The code blocks are only for Python 
# codes and comments and currently have a comment like ```# [Your code here]```. The markdown blocks 
# are only for plain or laTex text that you may use for answering descriptive questions. Currently, the 
# markdown blocks have a comment like <font color='red'>[your answer here]</font>. Please remove the 
# comments before filling the blocks. You can always add more blocks if you need to, or it just makes 
# your answers well-organized.
# 
# Although you may use other available online resources (such as GitHub, Kaggle Notebooks), it is highly 
# recommended to try your best to do it yourself. If you are using an online code or paper, make sure you 
# cite their work properly to avoid plagiarism. Using other students' works is absolutely forbidden and 
# considered cheating.
# 
# Write comments for your codes in the big picture mode. One comment line at the top of each code block is 
# necessary to explain what you did in that block. Don't comment on every detail.
# 
# Name your variables properly that represent the data they are holding, such as ``` test_set = ..., 
# learning_rate = ...``` not ```a = ..., b = ...```.
# 
# Implementing and reporting results using other architectures than LeNet will grant you an extra 20% on grade.
# 
# In this [Kaggle Notebook](https://www.kaggle.com/roblexnana/cifar10-with-cnn-for-beginer), you may find useful information about how your outputs must look. Just remember, they are using a different architecture, and they are using TensorFlow for implementations.
# 
# 
# ## How to submit:
# After you have completed the assignment: 
# 1. Save a version (You can use 'Quick Save' to avoid re-running the whole notebook)
# 2. Make the saved version public (Share -> Public)
# 3. Copy the 'Public URL'
# 4. Download your completed notebook as a '.ipynb' file (File -> Download)
# 5. Upload the 'Public URL' and the '.ipynb' files on the [CourseWorks](https://courseworks2.columbia.edu/).

# ### 1. (5 pts.) According to the [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset descriptions and other online resources, please identify the quantities below:
# 
# a) Total Number of samples \
# b) Number of classes \
# c) Image size \
# d) Write class names and their corresponding label index (e.g., 0. cats, 1. dogs, ...) \
# e) Intensity range

# <font color='red'>[your answers here]</font>

# * a) There are 60000 images. 
# * b) There are 10 classes. 
# * c) These images are 32X32X3. 
# * d) 0.airplane; 1.automobile;2.bird;3.cat;4.deer;5.dog;6.frog;7.horse;8.ship;9.truck. 
# * e) The intensity ranges from 0 to 255. 

# ### 2. (0 pts.) Import the required packages in the following code block. 
# You can always come back here and import another package; please keep them all in the following block to make your code well-organized.

# In[ ]:


# [Import packages here]
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader 
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import numpy as np

import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torch.optim as optim
from tensorflow.keras.utils import plot_model, to_categorical


# ### 3. (5 pts.) Load train and test sets using Pytorch datasets functions.
# Make sure the intensity range is between $[0, 1]$ and images are stored as 'tensor' type. 
# You may use transformers while downloading the dataset to do the job. Look at this tutorial: [Pytorch CIFAR10](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html))

# In[ ]:


# [Your code here]

BATCH_size = 32
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# training set 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_size,shuffle=True, num_workers=8,pin_memory = True)
# testing set 
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
print(type(testset))
testloader = DataLoader(testset, batch_size=BATCH_size,shuffle=False, num_workers=8,pin_memory = True)
# image classes 
classes = ('airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print((len(trainset)))
print(len(testset))


# ### 4. (10 pts.) Using the 'matplotlib' library, make a figure with $N\times4$ grid cells where $N$ is 
# the number of classes. Display one random sample of each class pulled from the train set in its corresponding 
# row that depends on its class index and the first column and its histogram in the second column. Repeat this 
# for the third and fourth columns but pull images from the test set.

# get examples 
examples = {i: [] for i in range(len(classes))}
#for x, i in trainset:
   # img = trainset[i]
   # imshow(img)
#plt.show()
    
    


# ### 5. (5 pts.) Split up the train set into new train and validation sets so that the number of samples in the 
# validation set is equal to the number of samples in the test set. Then create a 
# [```DataLoader```](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for each set, 
# including the test set, with a batch size of 32.
# The [Pytorch CIFAR10](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) tutorial will also help you here.
# Make sure none of the samples in the validation set exists in the new train set.

torch.manual_seed(44)
val_size = 10000
train_size = len(trainset) - val_size
print(train_size)

trainset, validationset = random_split(trainset, [train_size, val_size])
len(trainset), len(validationset)

# dataloader 
train_loader = DataLoader(trainset, batch_size = 32, shuffle=True, num_workers=4)
val_loader = DataLoader(validationset, batch_size = 32, shuffle=False, num_workers=4)
test_loader = DataLoader(testset, batch_size = 32, shuffle=False, num_workers=4)


# ### 6. (5 pts.) Display the number of samples for each class in the train, validation, and test sets as a stacked bar plot similar to the '[FirstTutorial](https://www.kaggle.com/soroush361/deeplearninginbme-firsttutorial)'.

# print shape 


# ### 7. (10 pts.) According to the LeNet architecture below, create a fully-connected model. Also, 
# identify the architeture's hyper-parameters, activation functions, and tensor shapes.
# Architecture hyper-parameter includes the number of layers, number of kernels in each layer, size of the kernels,
#  stride, zero-padding size. Just by looking at the architecture itself, you should be able to identify the hyper-parameters. 
#  Keep in mind that you identified the $W, H$, and $N$ (Which refers to the number of classes) in the first question.
# For more help, look at this [implementation](https://github.com/icpm/pytorch-cifar10/blob/master/models/LeNet.py).
# ![LeNet Architecture](https://raw.githubusercontent.com/soroush361/DeepLearningInBME/main/Ass1_Arch1.png)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)


# learning rate = 0.001

# ### 8. (5 pts.) Create an instance of [ADAM optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) 
# with an initial learning rate of $0.0001$ and an instance of 
# [Mean Squared Error (MSE)](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) loss function. 
# Briefly explain the ADAM optimizer algorithm and the MSE loss function.
# For ADAM optimizer, keep other arguments as default. 
# 
# For your information, here is the mathematics behind the ADAM optimizer: \
# For each parameter $w^j$
# $$
# v_t = \beta_1v_{t-1}-(1-\beta_1)g_t \\
# s_t = \beta_2s_{t-1}-(1-\beta_2)g_t^2 \\
# \Delta w^j = -\eta\frac{v_t}{\sqrt{s_t+\epsilon}}g_t \\
# w^j_{t+1} = w^j_t+\Delta w^j
# $$
# Where $\eta$ is the initial learning rate, $g_t$ is the gradient at time $t$ along $w^j$, $v_t$ is the exponential average of gradients along $w^j$, $s_t$ is the exponential average of squares of gradients along $w^j$, $\beta_1, \beta_2$ are the hyper-parameters, and $\epsilon$ is a small number to avoid dividing by zero.
# 
# The MSE loss function is:
# $$
# L(y,\hat{y}) = \frac{1}{N}\sum_{i=1}^N{(y_i-\hat{y}_i)^2}
# $$
# Where $y$ is the true value, $\hat{y}$ is the predicted value, $N$ is the number of classes. Keep in mind that $y$ is a one-hot vector like $y=\begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \end{bmatrix}$ (This example of $y$ indicates that the sample belongs to class ID 2, remember it is zero-indexed) and $\hat{y}=\begin{bmatrix} 0.1 \\ 0.03 \\ 0.8 \\ \vdots \end{bmatrix}$ shows the probability of belonging to each class for the same sample and predicted by the model.
# 
# For other optimization algorithms and loss functions, check the links below:
# 
# [Optimizers list](https://pytorch.org/docs/stable/optim.html#algorithms) 
# 
# [Loss function list](https://pytorch.org/docs/stable/nn.html#loss-functions)

# [Create your optimiser and loss function here]

# add optimizer 
optimizer = optim.Adam(net.parameters(), lr = 0.001)
# add loss function 
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()


# ADAM is an optimization algorithm with stochastic gradient descent. It uses adaptive learning rate and momentums to converge faster. 
# MSE takes the difference of predicted values and the actural values, then square it. It puts more weight on the outliers. 

# ### 9. (15 pts.) Train the model for 200 epochs using the train set and validate your model's 
# performance at the end of each epoch using the validation sets.
# To increase the training speed, use the GPU accelerator.
# Look at the '[FirstTutorial](https://www.kaggle.com/soroush361/deeplearninginbme-firsttutorial)' for more help.
# Do not forget to save the model at the end of each epoch.

# [Training and validation procedure in here]

epochs = 1
lowest_val_loss = float('inf')
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs): 
    
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    
    ### Train ###
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        
        one_labels = F.one_hot(labels,num_classes=10).to(torch.float)  #one hot format 
        outputs.type(torch.FloatTensor)
        loss = criterion(outputs,one_labels)
            
        optimizer.zero_grad() # set gradient to 0      
        loss.backward() # backpropagate the loss
        optimizer.step() # update the weights and bias values 
        
        ## accuracy ## 
        _, preds = torch.max(outputs, 1) # taking the highest value of prediction.
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data) # calculating te accuracy by summing correct predictions in a batch.
    
    ### Validation ###
    else:
        with torch.no_grad(): # we do not need gradient for validation.
            for val_inputs, val_labels in val_loader:
                
                val_inputs, val_labels = inputs.to(device), labels.to(device)
                val_outputs = net(val_inputs)    
                val_one_labels = F.one_hot(val_labels,num_classes=10).to(torch.float) #one hot format
                val_loss = criterion(val_outputs, val_one_labels)
        
                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_corrects += torch.sum(val_preds == val_labels.data)
            
            # selecting best epoch 
                if val_running_loss < lowest_val_loss:
                    
                    lowest_val_loss = val_running_loss 
                    best_epoch = e 
                    PATH = './cifar_net.pth'
                    torch.save(net.state_dict(),PATH)
                
        epoch_loss = running_loss/len(train_loader)  #loss per epoch 
        epoch_acc = running_corrects.float()/len(train_loader) #accuracy 
        running_loss_history.append(epoch_loss) #appending to display
        running_corrects_history.append(epoch_acc)
        
        val_epoch_loss = val_running_loss/len(val_loader)
        val_epoch_acc = val_running_corrects.float()/len(val_loader)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)
        # print results 
        print('epoch:',(e+1))
        print('training loss: {:.4f}, acc {:.4f}'.format(epoch_loss,epoch_acc.item())) 
        print('validation loss: {:.4f}, validation acc {:.4f}'.format(val_epoch_loss, val_epoch_acc.item()))
              
              


# ### 10. (5 pts.) Display the learning curve and illustrate the best epoch. Explain your criteria for the best epoch.
# The learning curve shows the model's loss and accuracy at the end of each epoch for all epochs (200 epochs). 
# The criteria for the best epoch can be the minimum loss or maximum accuracy or other criteria.

plt.figure(figsize = (12, 8))
plt.plot(running_loss_history, '-o', label = 'training loss', markersize = 3)
plt.plot(val_running_loss_history, '-o', label = 'validation loss', markersize = 3)
plt.legend(loc = 'upper right');

# best epoch = min loss 
best_epoch = np.argmin(val_running_loss_history)
print('best epoch: ', best_epoch)


# <font color='red'>[Describe your criteria for choosing the best epoch]</font>

# ### 11. (10 pts.) Load the model's weights at the best epoch 
# and test your model performance on the test set. 
# Display the confusion matrix and classification report.


# [Your code here for loading the best model, testing on the test set, and displaying the confusion matrix as well as the classification report]


# ### 12. (5 pts.) Display five random samples of each class titled with the true label and the predicted label. Comment on your model's performance.
# Samples must be pulled from the test set.


# [Your code here]

# ### 13. (20 pts.) Repeat the training, validation, and testing with the 
# [Cross-Entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) 
# loss function and initial learning rate of $0.005$. Explain how the model's performance changed.
# Essentially you need to copy all the codes above and just change the loss function and edit the learning rate. 
# Obviously, you don't need to re-import the dataset and the libraries. However, you need to create a 
# new instance of the architecture. Otherwise, the weights would be the same as the last epoch (or the best epoch) 
# in the last part. To avoid overwriting your previously trained model, change the save directories in the training loop.

# [Your code here]
