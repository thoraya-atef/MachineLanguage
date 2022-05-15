# Press the green button in the gutter to run the script.

import numpy as np
# for reading Dataset
from pandas import read_csv


class logisticRegression:
    # num iteration for gradient descent
    def __init__(self, lr=0.01, num_iter=1500):
        self.lr = lr
        self.num_iter = num_iter

    # trestbps chol thalach oldpeak
    # itr take training example and value of traing label
    def fit(self, X, Y):
        # init of parameters
        # 1d number of examples
        # 2d number of feature
        self.num_samples, self.num_feature = X.shape
        self.weights = np.zeros(self.num_feature)
        # given features
        self.X = X
        # target
        self.Y = Y
        # gredient decent
        for _ in range(self.num_iter):
            print("Error: ")
            self.gradientDecentEq()
            return self

    # for test sample
    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X @ self.weights) ))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

    def gradientDecentEq(self):
        sigma = self.sigmoid(self.X @ self.weights )
        loss=self.cost(sigma)
        # calculate gradients
        tmp = (sigma - self.Y)
        print(tmp)
        #tmp = np.reshape(tmp, self.num_samples)
        derviative_weight = tmp.T @ self.X / self.num_samples
        # update weights
        self.weights = self.weights - self.lr * derviative_weight
        return self

    def sigmoid(self,input):
        output = 1 / (1 + np.exp(-input))
        return output
    def cost(self,sigma):
        error = -1 / self.num_samples * np.sum(self.Y * np.log(sigma)) + (1 - self.Y) * np.log(1 - sigma)
        return error

def accuracy(given_y, pred_y):
    accuracy = (np.sum(given_y == pred_y) / len(given_y)) * 100
    return accuracy


if __name__ == '__main__':
    path = "heart.csv"
    Features = ['trestbps', 'chol', 'thalach', 'oldpeak', 'target']
    dataset = read_csv(path, skipinitialspace=True, usecols=Features)
    dataset=dataset.assign(bias=1)
    # print(dataset)
    shuffle_ds = dataset.sample(frac=1)
    # Define a size for your train set
    train_size = int(0.9 * 303)
    # Split your dataset
    train_set = shuffle_ds[:train_size]
    # print(train_set)
    test_set = shuffle_ds[train_size:]
    # print(test_set)
    x_train = train_set.filter(['trestbps', 'chol', 'thalach', 'oldpeak','bias'], axis=1)
    y_train = train_set.target


    x_test = test_set.filter(['trestbps', 'chol', 'thalach', 'oldpeak','bias'], axis=1)
    y_test = test_set.target

    reg = logisticRegression(lr=0.22, num_iter=1500)
    reg.fit(x_train, y_train)
    P = reg.predict(x_test)
    print("Accuracy",accuracy(y_test, P))
