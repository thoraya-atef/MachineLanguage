import numpy as np
import pandas as pd

# get data from excel sheet
Data = pd.read_csv("house_data.csv")

# put new colum after first colum =1
Data.insert(1, "ones", 1)
#Data = Data.head(17291)  # it's the training sets 80% from original data

# select our features [bathrooms,view,grade,lat,sqft_living15]
X = Data.iloc[:, [5, 10, 12, 18, 20]]
Y = Data.iloc[:, 3]  # price
learningRate = 0.000037
Iterations = 10000
S = Data.iloc[:, [1]]

# Normalization
Min = X.min()
Max = X.max()
X = (X - Min) / (Max - Min)

# convert to arrays by using np
X = np.array(X)
Y = np.array(Y).flatten()
S = np.array(S)
X = np.hstack((X, S))

Theta = np.array([0, 0, 0, 0, 0, 0])

# Cost function
def CostFunction(X, Y, Theta):
    hypothesis = X.dot(Theta)
    C = np.sum((hypothesis - Y) ** 2) / 2 / len(Y)
    return C


def GradientDescent(X, Y, Theta, learningRate, Iterations):

    for i in range(Iterations):
        hypothesis = X.dot(Theta)
        hypothesis_minus_Y = hypothesis - Y
        G = X.T.dot(hypothesis_minus_Y) / len(Y)
        Theta = Theta - (learningRate * G)
        Cost = CostFunction(X, Y, Theta)
        print('iteration (',(i+1),') =',(Cost*2))


    return Theta


ValuesOfTheta = GradientDescent(X, Y, Theta, learningRate, Iterations)
print('By using GradientDescent function ,this is our thetas')
print('theta  0 =',ValuesOfTheta[5],'\n')
for i in range(5):
    print('theta ',(i+1),'=',ValuesOfTheta[i],'\n')

Y_predict =X.dot(ValuesOfTheta)
error=Y_predict-Y

MAE = ((np.sum(abs( (Y_predict- Y)/Y_predict)))/len(Y))*100
print('Mean abs error = ',MAE,'%')

accurecy = np.sum(np.equal(Y,Y_predict))/len(Y)

print('Accurecy = ',accurecy)
print('Accuracy by using MAE = ', (100-MAE),'%')
print('column  of prediction (Price)  ', Y_predict)
