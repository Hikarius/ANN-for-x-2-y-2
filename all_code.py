#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[27]:


g = np.random.rand(50,2)
cikti = []
tmp = []
girdi = []
for each in g:
    tmp.append(each[0]*10-5)
    tmp.append(each[1]*10-5)
    girdi.append(tmp)
    tmp = [tmp[0]*tmp[0]+tmp[1]*tmp[1]]
    cikti.append(tmp)
    tmp=[]

    


# In[28]:


girdi =np.array(girdi)
cikti =np.array(cikti)


# In[29]:


tmp_max = np.amax(cikti)
cikti = cikti/tmp_max


# In[30]:


tmp_max


# In[31]:


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
x = np.outer(np.linspace(-5, 5, 30), np.ones(30))
y = x.copy().T # transpose
z = np.array(x ** 2 + y ** 2)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()


# In[32]:


def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# In[33]:


def relu(t):
    if t < 0:
        return 0
    return t


# In[34]:


def relu_derivative(p):
    if p<0 :
        return 0
    return 1


# In[35]:


class NeuralNetwork:

    def __init__(self, x, y):

        self.input      = x

        self.weights1   = np.random.rand(self.input.shape[1],6) 

        self.weights2   = np.random.rand(6,4)    
        
        self.weights3   = np.random.rand(4,1) 

        self.y          = y

        self.output     = np.zeros(self.y.shape)



    def feedforward(self):

        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))

        self.output = sigmoid(np.dot(self.layer2, self.weights3))
        
        return self.output 
        
    def backprop(self):
        
        d_errors3 = self.y - self.output
        d_weights3 = d_errors3 * sigmoid_derivative(self.output)

        
        d_errors2 = np.dot(d_weights3,self.weights3.T)
        d_weights2 = d_errors2 * sigmoid_derivative(self.layer2)
        
        
        d_errors1 = np.dot(d_weights2,self.weights2.T)
        d_weights1 = d_errors1 * sigmoid_derivative(self.layer1)
        
        self.weights1 = self.weights1 + np.dot(self.input.T,d_weights1)      

        self.weights2 = self.weights2 + np.dot(self.layer1.T,d_weights2)  
        
        self.weights3 = self.weights3 + np.dot(self.layer2.T,d_weights3)     
    
        
    def train(self):
        self.output = self.feedforward()
        self.backprop()


# In[43]:


NN = NeuralNetwork(x = girdi,y= cikti)


# In[44]:


error_rate = 10
e_l=[]
i=0
while(error_rate>0.05):
    NN.train()
    error_rate=np.mean(np.square(cikti - NN.feedforward()))
    e_l.append(error_rate)
    i = i+1
print("iteration "+str(i))
print("Actual Output: \n" + str(cikti*tmp_max)) 
print("Predicted Output: \n" + str(NN.feedforward() * tmp_max) )
print(error_rate)  
