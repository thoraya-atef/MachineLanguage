# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 19:05:12 2021

@author: pc
"""
        
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#cost function
def costFun(y,x,theta,n):
    h=x.dot(theta)
    j=(np.sum((h-y)**2))*(1/(2*n))
    
    return j
#-------------------

def gradientDescent(y,x,theta,learningCurve,iterate,n):    
    for iterate in range(iterate):
        h=x.dot(theta)

        dj=np.sum(x.T.dot(h-y))/n

        theta=theta-(learningCurve*dj)
        error=costFun(y,x,theta,n)
        print("Error in iterate ",iterate," : ", (error))
    return theta

#Read Dataset from file
dataSet=pd.read_csv("house_data.csv")

    #to x0
dataSet.insert(1,'x0',1)
    
    # Define a size for your train set
train_size = int(0.7 * len(dataSet))
    # Split your dataset
train_set = dataSet[:train_size]
    #print(train_set)
test_set =dataSet[train_size:]

    #to x1(predictor)-->sqft_living  , y(target value)--->price 
x_train=train_set.iloc[ : ,[1,6]]
y_train=train_set.iloc[ : ,3]

x_test=test_set.iloc[ : ,[1,6]]
y_test=test_set.iloc[ : ,3]
    #normalizing
min_x=x_train.iloc[ : ,1].min()
max_x=x_train.iloc[ : ,1].max()
x_train.iloc[ : ,1]=(x_train.iloc[ : ,1]-min_x)/(max_x-min_x)
    
min_x_test=x_test.iloc[ : ,1].min()
max_x_test=x_test.iloc[ : ,1].max()
x_test.iloc[ : ,1]=(x_test.iloc[ : ,1]-min_x_test)/(max_x_test-min_x_test)
    #to convert x0 ,x1 to matrices
x_train=np.array(x_train)
x_test=np.array(x_test)
    #to make y vector
y_train=np.array(y_train).flatten() 
y_test=np.array(y_test).flatten()
    #---------------------------
    
#intial value to theta0 ,theta1
theta=np.array([0,0])
#number of data points
n=len(y_train)
iterate=1000
learningCurve=.001
t=gradientDescent(y_train,x_train,theta,learningCurve,iterate,n)
print("Theta0 :  ", (t[0]),"Theta1 :  ", (t[1])) 
y_pred=t[0]+t[1]*x_test[:,1]
correctly_classified = 0
count = 0
for count in range(np.size(y_pred)):
    if y_test[count] == y_pred[count]:
       correctly_classified = correctly_classified + 1
    count=count+1
print("Accuracy on test set by our model       :  ", (
                correctly_classified / count) * 100)  

    
    




